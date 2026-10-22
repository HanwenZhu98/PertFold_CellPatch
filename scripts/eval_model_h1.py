"""把 predict_h1.py 的预测变成整数计数的细胞，用 cell-eval2 VCC2026 六项打分（.venv-eval）。

预测细胞：以模型输入用的 n 个真实对照细胞（x_idx）为模板，按预测的倍数变化调整计数
  r_g = (expm1(yhat_g) + ε) / (expm1(xbar_g) + ε)        模型基因；其余基因 r = 1
  r < 1：Binomial(c, r)（抽稀）
  r > 1：c + Poisson(L · p̄_g · (r − 1))                   L 为该细胞文库大小，p̄ 为 ctrl_pred 的平均表达比例
  直接从平均表达泊松采样的细胞没有真实细胞的过度离散，Wilcoxon 会把分布形状差异当成 DE；以真实细胞为模板可避免。
  --identity：r = 1（预测 = 对照细胞原样），即"把对照当预测"的基线。
预测侧 = 预测细胞 + ctrl_pred；真值侧 = real_idx 细胞 + ctrl_real；两侧各用各的对照（与 benchmark 相同）。

npz 有 perts 字段（扰动编号）时只评这些扰动，否则按 meta.json 的全部扰动。

用法：python eval_model_h1.py <pred.npz> [--identity]
输出：同目录 <stem>[_identity]_agg.csv、_per_pert.parquet
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

H1 = Path("/cluster2/home/hanwen/vcc_26/splithalf/data/H1.h5ad")
META = Path("/cluster2/home/hanwen/vcc_26/pertfold/data/h1/meta.json")
EPS = 0.1          # CP10k 单位
R_MAX = 50.0


def make_pred_cells(C, x_idx, xbar, yhat, col, pbar, identity, rng):
    """C：H1 计数（csr）。→ 预测细胞 csr [P·n, 全部基因]"""
    P, n = x_idx.shape
    tmpl = C[x_idx.ravel()].toarray().astype(np.float64)                          # [P·n, Gall]
    if identity:
        return sp.csr_matrix(tmpl.astype(np.float32))
    r = np.clip((np.expm1(yhat) + EPS) / (np.expm1(xbar) + EPS), 0.0, R_MAX)       # [P, G]
    r = np.repeat(r, n, axis=0)                                                    # [P·n, G]
    L = tmpl.sum(1, keepdims=True)
    c = tmpl[:, col]
    down = rng.binomial(c.astype(np.int64), np.minimum(r, 1.0))
    up = c + rng.poisson(L * pbar[col] * np.maximum(r - 1.0, 0.0))
    tmpl[:, col] = np.where(r < 1.0, down, up)
    return sp.csr_matrix(tmpl.astype(np.float32))


def adata(X, targets):
    return ad.AnnData(X=X.tocsr(), obs=pd.DataFrame({CFG.pert_col: pd.Categorical(targets)},
                                                    index=[f"c{i}" for i in range(X.shape[0])]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred")
    ap.add_argument("--identity", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    meta = json.loads(META.read_text())
    perts, col = meta["perts"], np.array(meta["gene_col"])
    z = np.load(args.pred)
    a = ad.read_h5ad(H1)
    C = a.X.tocsr()
    lib = np.asarray(C[z["ctrl_pred"]].sum(1)).ravel()
    pbar = np.asarray((sp.diags(1.0 / lib) @ C[z["ctrl_pred"]]).mean(0)).ravel()  # 对照平均表达比例 [Gall]
    rng = np.random.default_rng(int(Path(args.pred).stem.split("rep")[-1]))
    n = z["x_idx"].shape[1]
    pred_cells = make_pred_cells(C, z["x_idx"], z["xbar"], z["yhat"], col, pbar, args.identity, rng)
    pidx = z["perts"] if "perts" in z.files else np.arange(len(perts))
    tg = np.repeat(np.array(perts)[pidx], n)
    pred = adata(sp.vstack([pred_cells, C[z["ctrl_pred"]]]), list(tg) + [CTRL] * len(z["ctrl_pred"]))
    real = adata(sp.vstack([C[z["real_idx"].ravel()], C[z["ctrl_real"]]]), list(tg) + [CTRL] * len(z["ctrl_real"]))
    pred.var_names = real.var_names = a.var_names
    print(f"pred {pred.shape} real {real.shape}（{time.time() - t0:.0f}s）", flush=True)

    tidy = compute_metrics(pred, real, config=_inner_config(CFG))
    metrics = list(resolve_metrics(CFG.metrics, version=CFG.version)[0])
    try:
        agg = aggregate_metrics(tidy, metrics=metrics)
    except ValueError as err:
        if "expr_mse_unbiased_capped_norm" not in str(err):
            raise
        agg = aggregate_metrics(tidy, metrics=[m for m in metrics if m != "expr_mse_unbiased_capped_norm"])
    vals = dict(zip(agg["metric"].to_list(), agg["mean"].to_list()))
    stem = Path(args.pred).with_suffix("").as_posix() + ("_identity" if args.identity else "")
    pd.DataFrame([{"pred": args.pred, "identity": args.identity, "n_cells_per_bulk": n, **vals}]).to_csv(
        stem + "_agg.csv", index=False)
    tidy.write_parquet(stem + "_per_pert.parquet")
    print(("identity " if args.identity else "model ") + "  ".join(f"{m}={vals.get(m, np.nan):.4f}" for m in SIX)
          + f"  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
