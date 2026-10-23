"""benchmark 汇总（2026-10-22 晚，用户定：强基线 = kNN + A-E 线性；全部表达基因与 2,125 个基因两套都报）。
六项与综合分的算法同 summarize_v10.py（identity 为基线锚、真实数据对半自比为重复锚、MSE 截断 ≥ 0）；
s 为每折交叉拟合选出的输出幅度（其他 4 折扰动上综合分最高的网格值）。

用法：
  python summarize_bench.py --pick <目录> <npz 名字前缀>   打印每折的 s（"s0,...,s4"）
  python summarize_bench.py                                 两张总表 + 相对"较好的强基线"的逐扰动 bootstrap → report/summary_bench.csv
"""
import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from summarize_v10 import FOLD, REF, SIX, boot, composite, mean_reps

R3 = Path("/cluster2/home/hanwen/vcc_26/pertfold/runs3")
GRID = [0.5, 1, 1.5, 2, 3, 4, 6, 8]                                         # 10-23 夜：加 6、8（模型输出收缩）
CK10 = "stage2_ft_lr0.0001_best_val_mse"
TABLES = {
    "全部表达基因（12,710）": [
        ("强基线 kNN", R3 / "h1_bench/n32", "pred_BL_knn_expr"),
        ("强基线 A-E 线性（K562 表征）", R3 / "h1_bench/n32", "pred_BL_ae_expr"),
        ("kNN 来源加权（12 个来源）", R3 / "h1_bench/n32", "pred_BL_srcw_expr"),
        ("kNN − 0.5·训练平均", R3 / "h1_bench/n32", "pred_BL_knnc05_expr"),
        ("kNN − 1.0·训练平均", R3 / "h1_bench/n32", "pred_BL_knnc10_expr"),
        ("v12 Z0 底座（ZINB + 弱 DE + nb，PDS 选模）", R3 / "h1_v12/n32", "pred_Z0_s0_k"),
        ("v12 Z0 − 0.25·训练平均", R3 / "h1_v12/n32", "pred_Z0c025_s0_k"),
        ("v12 Z0 − 0.5·训练平均", R3 / "h1_v12/n32", "pred_Z0c050_s0_k"),
        ("v12 Z1 + 对照重构", R3 / "h1_v12/n32", "pred_Z1_s0_k"),
        ("v12 Z1 − 0.25·训练平均", R3 / "h1_v12/n32", "pred_Z1c025_s0_k"),
        ("v12 Z1 − 0.5·训练平均", R3 / "h1_v12/n32", "pred_Z1c050_s0_k"),
        ("v12 Z1 种子 1", R3 / "h1_v12/n32", "pred_Z1_s1_k"),
        ("v12 Z1b 对照重构权重 2", R3 / "h1_v12/n32", "pred_Z1b_s0_k"),
        ("v12 Z1 三种子集成", R3 / "h1_v12/n32", "pred_Z1ens3_k"),
        ("v12 Z1 五种子集成", R3 / "h1_v12/n32", "pred_Z1ens5_k"),
        ("v12 七模型集成（Z0×2、Z1×3、Z1b、Z1d）", R3 / "h1_v12/n32", "pred_Zmix7_k"),
        ("v12 Z1d 弱 DE 权重 0.5", R3 / "h1_v12/n32", "pred_Z1d_s0_k"),
        ("混合 0.5·kNN + 0.5·Z1", R3 / "h1_v12/n32", "pred_Bk50Z1_k"),
        ("混合 0.5·kNN + 0.5·(Z1 − 0.25·平均)", R3 / "h1_v12/n32", "pred_Bk50Z1c025_k"),
        ("混合 0.5·kNN + 0.5·(Z1 五种子集成 − 0.25·平均)", R3 / "h1_v12/n32", "pred_Bk50Z1ens5c025_k"),
        ("混合 0.3·kNN + 0.7·(Z1 − 0.25·平均)", R3 / "h1_v12/n32", "pred_Bk30Z1c025_k"),
        ("混合 0.7·kNN + 0.3·(Z1 − 0.25·平均)", R3 / "h1_v12/n32", "pred_Bk70Z1c025_k"),
        ("v12 Z1e（Z1 + 权重滑动平均 0.98）", R3 / "h1_v12/n32", "pred_Z1e_s0_k"),
        ("v12 Z1L（Z1，学习率 3e-5）", R3 / "h1_v12/n32", "pred_Z1L_s0_k"),
        ("v12 Z1S（Z1，训练 300 步）", R3 / "h1_v12/n32", "pred_Z1S_s0_k"),
        ("v12 Z1F（不留内部 val，全部非测试扰动训练，固定 250 步）", R3 / "h1_v12/n32", "pred_Z1F_s0_k"),
        ("混合 0.5·kNN + 0.5·(Z1F 三种子集成 − 0.25·平均)", R3 / "h1_v12/n32", "pred_Bk50Z1Fens3c025_k"),
        ("v12 Z2 + 能量距离", R3 / "h1_v12/n32", "pred_Z2_s0_k"),
        ("v12 Z3 + 判别器", R3 / "h1_v12/n32", "pred_Z3_s0_k"),
        ("v11 base（kNN，严格供体 94 个）", R3 / "h1_v11/n32", "pred_pass_base"),
        ("v11 直通 decoder（严格供体）", R3 / "h1_v11/n32", "pred_pass_full"),
    ],
    "原 2,125 个基因": [
        ("强基线 kNN", R3 / "h1_v10/n32", "pred_E0_knnbulk"),
        ("强基线 A-E 线性（K562 表征）", R3 / "h1_bench/n32", "pred_BL_ae_p2125"),
        ("v10 MSE + nb loss", R3 / "h1_v10/n32", f"pred_A_mse_{CK10}_cont"),
        ("v10 ZINB + nb loss", R3 / "h1_v10/n32", f"pred_A_zinb_{CK10}_cont"),
        ("v10 学到权重的 nbw", R3 / "h1_v10/n32", f"pred_A_mse_{CK10}_nbw"),
    ],
}


def pick(d, stem):
    fs = []
    for k in range(5):
        other = [p for p, f in FOLD.items() if f != k]
        reps = lambda s: [r for r in (0, 1) if (d / f"{stem}_rep{r}_s{s:g}_per_pert.parquet").exists()]   # 10-22 晚：扫描可只用第 0 次重复
        ok = [s for s in GRID if reps(s)]
        fs.append(max(ok, key=lambda s: composite(mean_reps([str(d / f"{stem}_rep{r}_s{s:g}_per_pert.parquet") for r in reps(s)], other))[0]))
    return fs


def fs_paths(d, stem):
    """只取与当前交叉拟合选择一致的 5 次重复（网格扩展后可能存在多套 fs 文件）"""
    try:
        fs_now = "_".join(f"{v:g}" for v in pick(d, stem))
    except ValueError:
        fs_now = None
    fs = sorted(glob.glob(str(d / f"{stem}_rep?_fs{fs_now}_per_pert.parquet"))) if fs_now else []
    return fs if len(fs) == 5 else None


def main():
    if len(sys.argv) > 3 and sys.argv[1] == "--pick":
        print(",".join(f"{s:g}" for s in pick(Path(sys.argv[2]), sys.argv[3])))
        return
    pd.set_option("display.width", 250)
    out = []
    for title, arms in TABLES.items():
        paths = {k: [v.format(r=r) for r in range(5)] for k, v in REF.items() if k in ("identity", "训练平均", "STRING kNN", "r4-v2 完整", "r4-v2 nbw")}
        if title.startswith("全部"):
            paths = {k: v for k, v in paths.items() if k in ("identity", "训练平均")}
        for name, d, stem in arms:
            ps = fs_paths(d, stem)
            if ps:
                paths[f"{name}（s={re.search(r'_fs(.*)_per_pert', ps[0]).group(1)}）"] = ps
            else:
                print(f"[{title}] 缺：{name}（{stem}）")
        rows = []
        for name, ps in paths.items():
            u = mean_reps(ps)
            c, s = composite(u)
            rows.append({"基因范围": title, "配置": name, **{k: round(v, 3) for k, v in u.items()}, "综合分": round(c, 3)})
        T = pd.DataFrame(rows)
        print(f"\n===== {title}（5 次重复，147 个扰动）")
        print(T.drop(columns="基因范围").to_string(index=False))
        out.append(T)
        base = [k for k in paths if k.startswith("强基线")]
        if base:
            best = max(base, key=lambda k: composite(mean_reps(paths[k]))[0])
            print(f"较好的强基线：{best}；逐扰动 bootstrap（2000 次，综合分 / PDS 之差，95% 区间，* 为不含 0）")
            for a in paths:
                if a == best or a in ("identity", "训练平均"):
                    continue
                (mc, lc, hc), (mp, lp, hp) = boot(paths[a], paths[best])
                print(f"  {a} − 较好基线：综合分 {mc:+.3f} [{lc:+.3f}, {hc:+.3f}]{'*' if lc > 0 or hc < 0 else ''}  "
                      f"PDS {mp:+.3f} [{lp:+.3f}, {hp:+.3f}]{'*' if lp > 0 or hp < 0 else ''}")
    pd.concat(out).to_csv("/cluster2/home/hanwen/vcc_26/pertfold/analysis/single_line/report/summary_bench.csv", index=False)


if __name__ == "__main__":
    main()
