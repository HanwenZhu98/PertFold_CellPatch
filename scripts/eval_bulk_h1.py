"""v10（2026-10-22 晚）：打分空间预测的 cell-eval2 VCC 2026 六项打分（.venv-eval）。

npz 里的 dhat 是打分空间的 δ̂（log1p(5e4 · 伪 bulk 表达比例) 之差，predict_h1_v5.py 对 --space bulk 模型输出，
或 make_v10_inputs.py 的 E0）。换成每个基因的倍数，再作用到模板细胞上（作用方式与 eval_model_h1.py 相同）：
  b_c = 预测侧对照（ctrl_pred）的打分空间伪 bulk
  r_g = expm1(b_c,g + s · δ̂_g) / expm1(b_c,g)
  打分空间群体伪 bulk ≤ EXPR_MIN 的基因（不表达）与本扰动靶基因 r = 1；b_c,g = 0 时 r = 1
  --scale s：输出幅度（v10 的交叉拟合在这里扫）；--fold-scale a,b,c,d,e：每折各自的 s（5 折合并预测，按 meta.json 的折）
npz 里有 dhat_all（[扰动, 全部 18,080 个基因]，make_v10_inputs.py --all-genes）时改用全部基因：
  b_c、表达掩码（data/h1/bulk_sums_all.npz）、靶基因位置都在全部基因上算，模板细胞的全部基因都按 r 改动。
其余与 eval_model_h1.py 相同：预测侧 = 预测细胞 + ctrl_pred，真值侧 = real_idx 细胞 + ctrl_real。

用法：python eval_bulk_h1.py <pred.npz> [--scale 1.0 | --fold-scale 1,1,1,1,1] [--label 名字]
输出：同目录 <stem>_<label 或 s<scale>>_agg.csv、_per_pert.parquet
"""
import argparse
import dataclasses
import json
import os
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

import eval_model_h1 as E
from benchmark_h1 import CFG, CTRL, SIX
from cell_eval2.anchor import _inner_config
from cell_eval2.catalog import resolve_metrics
from cell_eval2.run import aggregate_metrics, compute_metrics
from oracle_dist_h1 import bulk_cpm, cells_from_r

D = Path("/cluster2/home/hanwen/vcc_26/pertfold/data/h1")
EXPR_MIN = 0.03          # 与 train_h1_v5.py 相同


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--fold-scale", default="", help="每折的 s，逗号分隔（覆盖 --scale）")
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    E.START_HOLD, E.SLOTS = 40, int(os.environ.get("EVAL_SLOTS", "10"))      # 启动间隔 40 s、最多 10 个同时打分（原 180 s、8 个）
    host = os.uname().nodename.split(".")[0]                                     # 10-23：槽按节点分开（锁文件在共享盘上），node5 沿用原目录
    E.SLOT_DIR = Path(os.environ.get("EVAL_SLOT_DIR", str(E.SLOT_DIR) + ("" if host == "node5" else f"_{host}")))
    slot = E.acquire_slot()  # noqa: F841（持有到进程结束）
    t0 = time.time()
    meta = json.loads((D / "meta.json").read_text())
    perts, col = meta["perts"], np.array(meta["gene_col"])
    z = np.load(args.pred)
    pidx = z["perts"] if "perts" in z.files else np.arange(len(perts))
    allg = "dhat_all" in z.files                                                 # 全部基因输出
    dhat = (z["dhat_all"] if allg else z["dhat"]).astype(np.float64)
    if args.fold_scale:
        fs = [float(v) for v in args.fold_scale.split(",")]
        fold_of = {p: k for k, fo in enumerate(meta["folds"]) for p in fo}
        s = np.array([fs[fold_of[int(p)]] for p in pidx])[:, None]
        label = args.label or "fs" + "_".join(f"{v:g}" for v in fs)
    else:
        s = args.scale
        label = args.label or f"s{args.scale:g}"
    if allg:
        a0 = ad.read_h5ad(E.H1, backed="r")
        vi = {g: i for i, g in enumerate(a0.var_names)}
        col = np.arange(len(vi))
        expr = np.load(D / "bulk_sums_all.npz")["B"][-1] > EXPR_MIN
        tgt = np.array([vi.get(perts[int(p)], -1) for p in pidx])
    else:
        expr = np.load(D / "bulk_sums.npz")["B"][-1] > EXPR_MIN
        tgt = np.load(D / "labels.npz")["target"][pidx]
    keep = np.broadcast_to(expr, dhat.shape).copy()
    keep[np.arange(len(pidx))[tgt >= 0], tgt[tgt >= 0]] = False
    a = ad.read_h5ad(E.H1)
    C = a.X.tocsr()
    lib = np.asarray(C[z["ctrl_pred"]].sum(1)).ravel()
    pbar = np.asarray((sp.diags(1.0 / lib) @ C[z["ctrl_pred"]]).mean(0)).ravel()
    bc = np.log1p(bulk_cpm(C, z["ctrl_pred"], col))                               # 预测侧对照的打分空间伪 bulk
    ok = keep & (bc[None, :] > 0)
    r = np.ones_like(dhat)
    r[ok] = (np.expm1(bc[None, :] + s * dhat) / np.expm1(np.maximum(bc, 1e-12))[None, :])[ok]
    rng = np.random.default_rng(int(Path(args.pred).stem.split("rep")[-1]))
    n = z["x_idx"].shape[1]
    pred_cells = cells_from_r(C, z["x_idx"], r, col, pbar, rng)
    tg = np.repeat(np.array(perts)[pidx], n)
    pred = E.adata(sp.vstack([pred_cells, C[z["ctrl_pred"]]]), list(tg) + [CTRL] * len(z["ctrl_pred"]))
    real = E.adata(sp.vstack([C[z["real_idx"].ravel()], C[z["ctrl_real"]]]), list(tg) + [CTRL] * len(z["ctrl_real"]))
    pred.var_names = real.var_names = a.var_names
    print(f"pred {pred.shape} real {real.shape}；r>2 {np.mean(r > 2):.4f} r<0.5 {np.mean(r < 0.5):.4f}（{time.time() - t0:.0f}s）", flush=True)

    nt = int(os.environ.get("EVAL_THREADS", "10"))                               # 10-22 晚：每个打分进程限 10 线程（原 -1 = 全部核，10 个并行时严重超订）
    cfg = dataclasses.replace(CFG, num_threads=nt, gather_threads=nt)
    tidy = compute_metrics(pred, real, config=_inner_config(cfg))
    metrics = list(resolve_metrics(CFG.metrics, version=CFG.version)[0])
    try:
        agg = aggregate_metrics(tidy, metrics=metrics)
    except ValueError as err:
        if "expr_mse_unbiased_capped_norm" not in str(err):
            raise
        agg = aggregate_metrics(tidy, metrics=[m for m in metrics if m != "expr_mse_unbiased_capped_norm"])
    vals = dict(zip(agg["metric"].to_list(), agg["mean"].to_list()))
    stem = Path(args.pred).with_suffix("").as_posix() + "_" + label
    pd.DataFrame([{"pred": args.pred, "label": label, "n_cells_per_bulk": n, **vals}]).to_csv(stem + "_agg.csv", index=False)
    tidy.write_parquet(stem + "_per_pert.parquet")
    print(f"{label}  " + "  ".join(f"{m}={vals.get(m, np.nan):.4f}" for m in SIX) + f"  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
