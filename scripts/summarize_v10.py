"""v10 汇总（2026-10-22 晚）：VCC 2026 六项 + 综合分（基线锚 identity、重复锚 benchmark，MSE 截断 ≥ 0；用户 10-21 的规则）。

逐扰动结果（eval_bulk_h1.py / eval_model_h1.py 的 _per_pert.parquet）可在任意扰动子集上重算六项：
PDS、fidelity、reach、sig_jacc、lfc_nmae 取逐扰动均值（跳过 NaN），MSE_norm = Σ expr_mse_unbiased_capped / Σ expr_distance_unbiased。

用法：
  python summarize_v10.py --pick <臂>      交叉拟合选幅度：每折用其他 4 折扰动（第 0、1 次重复）上综合分最高的 s，打印 "s0,s1,s2,s3,s4"
  python summarize_v10.py                  汇总表（扫 s 的表、交叉拟合后 5 次重复的六项与综合分、逐扰动 bootstrap）→ report/summary_v10.csv
"""
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path("/cluster2/home/hanwen/vcc_26/pertfold/runs3")
V = R / "h1_v10/n32"
META = json.loads(Path("/cluster2/home/hanwen/vcc_26/pertfold/data/h1/meta.json").read_text())
FOLD = {META["perts"][p]: k for k, fo in enumerate(META["folds"]) for p in fo}
CK = "stage2_ft_lr0.0001_best_val_mse"
ARMS = {"A_mse 连续头": f"pred_A_mse_{CK}_cont", "A_mse nbw": f"pred_A_mse_{CK}_nbw",
        "A_zinb 连续头": f"pred_A_zinb_{CK}_cont", "E0 并集均匀 kNN": "pred_E0_knnbulk"}
GRID = [0.5, 1, 1.5, 2, 3]
REF = {"identity": str(R / "h1_v3/n32/pred_string_msa_stage2_ft_lr0.0001_step500_rep{r}_identity_per_pert.parquet"),
       "训练平均": str(R / "h1_v3/n32/pred_train_mean_rep{r}_per_pert.parquet"),
       "STRING kNN": str(R / "h1_v3/n32/pred_string_knn_rep{r}_per_pert.parquet"),
       "r4-v2 完整": str(R / "h1_v7k/n32/pred_K_S1c修正_stage2_ft_lr0.0001_best_pearson_exp_exp_rep{r}_per_pert.parquet"),
       "r4-v2 nbw": str(R / "h1_v7k/n32/pred_K_S1c修正_stage2_ft_lr0.0001_best_pearson_exp_nbw_rep{r}_per_pert.parquet")}
BENCH = "/cluster2/home/hanwen/vcc_26/pertfold/analysis/single_line/results/n32/rep*_per_pert.parquet"
M = {"pds_cosine": "PDS", "de_wilcoxon_direction_fidelity_yield_raw": "fid", "de_wilcoxon_direction_reach_raw": "reach",
     "de_wilcoxon_sig_jaccard": "sjac", "de_wilcoxon_lfc_nmae": "nmae"}
SIX = ["PDS", "MSE", "fid", "reach", "sjac", "nmae"]


def wide(path):
    d = pd.read_parquet(path)
    return d.pivot_table(index="perturbation", columns="metric", values="value")


def six(w, perts=None):
    """逐扰动宽表 → 六项（可限定扰动子集）"""
    if perts is not None:
        w = w.loc[w.index.intersection(perts)]
    out = {v: w[k].mean() for k, v in M.items()}
    out["MSE"] = w["expr_mse_unbiased_capped"].sum() / w["expr_distance_unbiased"].sum()
    return out


def mean_reps(paths, perts=None):
    vals = [six(wide(p), perts) for p in paths]
    return {k: float(np.mean([v[k] for v in vals])) for k in SIX}


IDN = mean_reps([REF["identity"].format(r=r) for r in range(5)])
BEN = mean_reps(sorted(glob.glob(BENCH)))


def composite(u):
    s = {}
    for k in SIX:
        b, r = IDN[k], BEN[k]
        v = (u[k] - b) / (r - b)
        s[k] = min(max(v, 0.0), 1.0) if k == "MSE" else v
    return float(np.mean(list(s.values()))), s


def grid_paths(stem, s, reps=(0, 1)):
    return [str(V / f"{stem}_rep{r}_s{s:g}_per_pert.parquet") for r in reps]


def pick(arm):
    stem = ARMS[arm]
    fs = []
    for k in range(5):
        other = [p for p, f in FOLD.items() if f != k]
        best = max(GRID, key=lambda s: composite(mean_reps(grid_paths(stem, s), other))[0])
        fs.append(best)
    return fs


def boot(pa, pb, n=2000, seed=0):
    """逐扰动 bootstrap：两组（各自 5 次重复的逐扰动宽表列表）在同一批重抽的扰动上算综合分与 PDS 之差"""
    wa = [wide(p) for p in pa]
    wb = [wide(p) for p in pb]
    perts = sorted(set.intersection(*[set(w.index) for w in wa + wb]))
    rng = np.random.default_rng(seed)
    dc, dp = [], []
    for _ in range(n):
        idx = rng.choice(perts, len(perts))
        ua = {k: np.mean([six(w.loc[idx])[k] for w in wa]) for k in SIX}
        ub = {k: np.mean([six(w.loc[idx])[k] for w in wb]) for k in SIX}
        dc.append(composite(ua)[0] - composite(ub)[0])
        dp.append(ua["PDS"] - ub["PDS"])
    q = lambda v: (np.mean(v), np.percentile(v, 2.5), np.percentile(v, 97.5))  # noqa: E731
    return q(dc), q(dp)


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--pick":
        print(",".join(f"{s:g}" for s in pick(sys.argv[2])))
        return
    pd.set_option("display.width", 250)
    rows = []
    for arm, stem in ARMS.items():                                               # 扫 s（第 0、1 次重复，全部扰动）
        for s in GRID:
            ps = grid_paths(stem, s)
            if all(Path(p).exists() for p in ps):
                u = mean_reps(ps)
                rows.append({"臂": arm, "s": s, **{k: round(v, 3) for k, v in u.items()}, "综合分": round(composite(u)[0], 3)})
    if rows:
        print("扫幅度 s（第 0、1 次重复，147 个扰动；仅供参考，选 s 用交叉拟合）")
        print(pd.DataFrame(rows).to_string(index=False))
    final, paths = [], {}
    for name, pat in REF.items():
        paths[name] = [pat.format(r=r) for r in range(5)]
    for arm, stem in ARMS.items():                                               # 交叉拟合后的 5 次重复
        fs = sorted(glob.glob(str(V / f"{stem}_rep?_fs*_per_pert.parquet")))
        if len(fs) == 5:
            paths[arm + "（交叉拟合 s=" + re.search(r"_fs(.*)_per_pert", fs[0]).group(1) + "）"] = fs
    for name, ps in paths.items():
        if not all(Path(p).exists() for p in ps):
            print("缺", name)
            continue
        u = mean_reps(ps)
        c, s = composite(u)
        final.append({"配置": name, **{k: round(v, 3) for k, v in u.items()}, "综合分": round(c, 3),
                      **{"s_" + k: round(v, 3) for k, v in s.items()}})
    T = pd.DataFrame(final)
    print("\n5 次重复（147 个扰动）")
    print(T.to_string(index=False))
    T.to_csv("/cluster2/home/hanwen/vcc_26/pertfold/analysis/single_line/report/summary_v10.csv", index=False)
    print("\n逐扰动 bootstrap（2000 次；综合分与 PDS 之差，95% 区间；* 为不含 0）")
    for a in [k for k in paths if k.startswith(("A_", "E0"))]:
        for b in ("r4-v2 完整", "r4-v2 nbw"):
            (mc, lc, hc), (mp, lp, hp) = boot(paths[a], paths[b])
            sc = "*" if lc > 0 or hc < 0 else ""
            sp_ = "*" if lp > 0 or hp < 0 else ""
            print(f"{a} − {b}：综合分 {mc:+.3f} [{lc:+.3f}, {hc:+.3f}]{sc}  PDS {mp:+.3f} [{lp:+.3f}, {hp:+.3f}]{sp_}")


if __name__ == "__main__":
    main()
