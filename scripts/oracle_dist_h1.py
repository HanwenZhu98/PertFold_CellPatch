"""真值上限与生成方式对比（2026-10-22，用户要求：fold change 转换是否有问题；用分布（ZINB / NB）生成细胞是否更好）。
所有读出用同一套抽样（STRING kNN rep<r> 的 x_idx / real_idx / ctrl_pred / ctrl_real），真值侧与 eval_model_h1.py 相同。
--mode：
  tmpl_dp    ŷ = xbar + 群体 delta_p（log 的平均），走 eval_model_h1 的模板倍数转换（现在的流程，给完美均值）
  tmpl_bulk  r = 群体 bulk 倍数（打分空间 bulk_lognorm：组内计数求和 → CPM 5e4 → log1p，对 ctrl_pred 的 bulk）直接作用于模板
  nb_pert    NB 抽样：均值 = 模板细胞文库 × 该扰动的群体表达比例；离散度 = 该扰动细胞（去掉 real_idx）矩估计
  nb_ctrl    同上，离散度用对照细胞（ctrl_pred）矩估计 —— 模型只预测均值时的分布生成
  cal:<npz>  模型预测 δ̂（log 的平均）按基因换算到 bulk 空间（δ̂ / k_g，k_g = 本折训练扰动上 delta_p 对 bulk δ 的过原点斜率，
             截断到 [0.2, 1.2]），再按 tmpl_bulk 的方式作用于模板；npz 为该模型的预测文件（rep 与 --rep 相同）
  add_lin    加法（计数空间）：Δf = 扰动群体表达比例 − 对照（全部对照细胞）群体表达比例（bulk 均值之差）；
             每个模板细胞 c' = c + Poisson(L·Δf)（Δf > 0）或 max(c − Poisson(L·|Δf|), 0)（Δf < 0），L 为该细胞文库大小
  add_log    加法（log 空间）：每个模板细胞 x' = max(log1p(CP10k) + 群体 delta_p, 0) → CP10k' = expm1(x') → 计数 = CP10k'·L/1e4，随机取整
只有模型基因被改变 / 重新抽样，其他基因保持模板计数（与 eval_model_h1 相同）。
用法：python oracle_dist_h1.py <mode> [--rep 0] → runs3/h1_oracle/<名字>_rep<r>_agg.csv、_per_pert.parquet
"""
import argparse
import json
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from benchmark_h1 import CFG, CTRL, SIX
from cell_eval2.anchor import _inner_config
from cell_eval2.catalog import resolve_metrics
from cell_eval2.run import aggregate_metrics, compute_metrics
from eval_model_h1 import EPS, R_MAX, acquire_slot, adata, make_pred_cells

H1 = Path("/cluster2/home/hanwen/vcc_26/splithalf/data/H1.h5ad")
D = Path("/cluster2/home/hanwen/vcc_26/pertfold/data/h1")
RUNS = Path("/cluster2/home/hanwen/vcc_26/pertfold/runs3")
OUT = RUNS / "h1_oracle"
BULK_SUM = 5e4
EPS_B = 0.1          # bulk CPM（5e4 尺度）的伪计数，避免 0 表达基因上的倍数发散


def bulk_cpm(C, idx, col):
    s = np.asarray(C[idx].sum(0)).ravel()
    return (s / s.sum() * BULK_SUM)[col]


def cells_from_r(C, x_idx, r, col, pbar, rng):
    """与 make_pred_cells 相同的抽稀 / 泊松添加，只是直接给倍数 r [P, G]"""
    P, n = x_idx.shape
    tmpl = C[x_idx.ravel()].toarray().astype(np.float64)
    r = np.repeat(np.clip(r, 0.0, R_MAX), n, axis=0)
    L = tmpl.sum(1, keepdims=True)
    c = tmpl[:, col]
    down = rng.binomial(c.astype(np.int64), np.minimum(r, 1.0))
    up = c + rng.poisson(L * pbar[col] * np.maximum(r - 1.0, 0.0))
    tmpl[:, col] = np.where(r < 1.0, down, up)
    return sp.csr_matrix(tmpl.astype(np.float32))


def nb_moments(C, idx, col):
    """组内 NB 矩估计：f = Σc / ΣL（表达比例），1/θ = max(Σ[(c − Lf)² − Lf], 0) / Σ(Lf)²"""
    X = C[idx][:, col].toarray().astype(np.float64)
    L = np.asarray(C[idx].sum(1)).ravel()[:, None]
    f = X.sum(0) / L.sum()
    mu = L * f
    inv_theta = np.maximum(((X - mu) ** 2 - mu).sum(0), 0) / np.maximum((mu ** 2).sum(0), 1e-12)
    return f, inv_theta


def nb_cells(C, x_idx, F, INV, col, rng):
    """模板细胞的文库大小 L_i；模型基因 c ~ NB(均值 L_i·f, 1/θ)（Gamma-Poisson），其他基因保持模板计数"""
    P, n = x_idx.shape
    tmpl = C[x_idx.ravel()].toarray().astype(np.float64)
    L = tmpl.sum(1, keepdims=True)
    mu = L * np.repeat(F, n, axis=0)
    inv = np.repeat(INV, n, axis=0)
    shape = np.where(inv > 1e-8, 1.0 / np.maximum(inv, 1e-8), 1e8)
    lam = rng.gamma(shape, mu / shape)
    tmpl[:, col] = rng.poisson(lam)
    return sp.csr_matrix(tmpl.astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode")
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--name", default="")
    args = ap.parse_args()
    slot = acquire_slot()  # noqa: F841（与 eval_model_h1 共用的全局限流）
    t0 = time.time()
    meta = json.loads((D / "meta.json").read_text())
    perts, col = meta["perts"], np.array(meta["gene_col"])
    Lb = np.load(D / "labels.npz")
    dp = Lb["delta_p"]
    z = np.load(RUNS / f"h1_v3/n32/pred_string_knn_rep{args.rep}.npz")
    a = ad.read_h5ad(H1)
    C = a.X.tocsr()
    lab = a.obs[CFG.pert_col].astype(str).values
    P, n = z["x_idx"].shape
    lib = np.asarray(C[z["ctrl_pred"]].sum(1)).ravel()
    pbar = np.asarray((sp.diags(1.0 / lib) @ C[z["ctrl_pred"]]).mean(0)).ravel()
    rng = np.random.default_rng(1000 + args.rep)
    pert_cells = [np.where(lab == p)[0] for p in perts]
    ctrl_all = np.where(lab == CTRL)[0]
    bc_all = bulk_cpm(C, ctrl_all, col)
    pop = np.stack([np.log1p(bulk_cpm(C, pc, col)) - np.log1p(bc_all) for pc in pert_cells])   # 群体 bulk δ [P, G]
    bc = bulk_cpm(C, z["ctrl_pred"], col)                                       # 预测侧对照 bulk CPM

    def r_from_bulk_delta(db):
        return (np.expm1(np.log1p(bc) + db) + EPS_B) / (bc + EPS_B)

    mode = args.mode
    if mode == "tmpl_dp":
        cells = make_pred_cells(C, z["x_idx"], z["xbar"], z["xbar"] + dp, col, pbar, False, rng)
    elif mode == "tmpl_bulk":
        cells = cells_from_r(C, z["x_idx"], r_from_bulk_delta(pop), col, pbar, rng)
    elif mode in ("nb_pert", "nb_ctrl"):
        F = np.zeros((P, len(col)))
        INV = np.zeros((P, len(col)))
        if mode == "nb_ctrl":
            _, inv_c = nb_moments(C, z["ctrl_pred"], col)
        for i, pc in enumerate(pert_cells):
            fit = np.setdiff1d(pc, z["real_idx"][i])                            # 拟合用的细胞不含打分真值侧
            f, inv = nb_moments(C, fit, col)
            F[i] = f
            INV[i] = inv if mode == "nb_pert" else inv_c
        cells = nb_cells(C, z["x_idx"], F, INV, col, rng)
    elif mode == "add_lin":                                                     # 用户 10-22：真值 bulk 均值之差，加法
        tmpl = C[z["x_idx"].ravel()].toarray().astype(np.float64)
        L = tmpl.sum(1, keepdims=True)
        lib_c = np.asarray(C[ctrl_all].sum(1)).ravel()
        f_ctrl = np.asarray(C[ctrl_all][:, col].sum(0)).ravel() / lib_c.sum()
        df = np.stack([np.asarray(C[pc][:, col].sum(0)).ravel() / np.asarray(C[pc].sum(1)).sum() - f_ctrl for pc in pert_cells])
        df = np.repeat(df, n, axis=0)
        c = tmpl[:, col]
        k = rng.poisson(L * np.abs(df))
        tmpl[:, col] = np.where(df >= 0, c + k, np.maximum(c - k, 0))
        cells = sp.csr_matrix(tmpl.astype(np.float32))
    elif mode == "add_log":                                                     # 用户 10-22：真值 delta_p，log 空间加法
        tmpl = C[z["x_idx"].ravel()].toarray().astype(np.float64)
        L = tmpl.sum(1, keepdims=True)
        x = np.log1p(tmpl[:, col] / L * 1e4)
        v = np.expm1(np.maximum(x + np.repeat(dp, n, axis=0), 0.0)) * L / 1e4
        fl = np.floor(v)
        tmpl[:, col] = fl + (rng.random(v.shape) < (v - fl))
        cells = sp.csr_matrix(tmpl.astype(np.float32))
    elif mode.startswith("cal:"):
        pz = np.load(mode[4:])
        assert (pz["x_idx"] == z["x_idx"]).all(), "预测文件的抽样与 kNN rep 不同"
        dh = pz["yhat"] - pz["xbar"]
        db = np.zeros_like(dh)
        for f in meta["folds"]:                                                 # k_g 只用本折训练扰动
            tr = np.setdiff1d(np.arange(P), f)
            k = (dp[tr] * pop[tr]).sum(0) / np.maximum((pop[tr] ** 2).sum(0), 1e-12)
            db[f] = dh[f] / np.clip(k, 0.2, 1.2)
        cells = cells_from_r(C, z["x_idx"], r_from_bulk_delta(db), col, pbar, rng)
    else:
        raise SystemExit(f"未知 mode {mode}")
    tg = np.repeat(np.array(perts), n)
    pred = adata(sp.vstack([cells, C[z["ctrl_pred"]]]), list(tg) + [CTRL] * len(z["ctrl_pred"]))
    real = adata(sp.vstack([C[z["real_idx"].ravel()], C[z["ctrl_real"]]]), list(tg) + [CTRL] * len(z["ctrl_real"]))
    pred.var_names = real.var_names = a.var_names
    print(f"{mode}：pred {pred.shape}（{time.time() - t0:.0f}s）", flush=True)
    tidy = compute_metrics(pred, real, config=_inner_config(CFG))
    metrics = list(resolve_metrics(CFG.metrics, version=CFG.version)[0])
    agg = aggregate_metrics(tidy, metrics=metrics)
    vals = dict(zip(agg["metric"].to_list(), agg["mean"].to_list()))
    OUT.mkdir(parents=True, exist_ok=True)
    name = args.name or mode.replace(":", "_").replace("/", "_")
    stem = OUT / f"{name}_rep{args.rep}"
    pd.DataFrame([{"mode": mode, "rep": args.rep, **vals}]).to_csv(f"{stem}_agg.csv", index=False)
    tidy.write_parquet(f"{stem}_per_pert.parquet")
    print(f"{name} " + "  ".join(f"{m}={vals.get(m, np.nan):.4f}" for m in SIX) + f"  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
