"""H1 单细胞系基准：真实数据自己和自己比，按 n_cells_per_bulk 分组，重复 20 次。

每次重复：
  每个扰动（只用 ≥ min_cells 个细胞的扰动，三个 n 共用同一扰动集合）不放回抽 2n 个细胞，
  对半分成两组各 n 个：一组当"预测"，另一组当"真值"。对照细胞全部保留并对半分，两组各用各的对照。
  用 cell-eval2 0.18.0 的 VCC2026 六项指标打分（与 splithalf.py 相同的调用方式）。
这是模型在该 n 下应当接近的水平，而不是满分。

用法（.venv-eval）：
  python benchmark_h1.py <n_cells_per_bulk> [n_reps=20] [min_cells=64] [划分文件]
输出：results/n<N>/rep<i>_agg.csv（六项均值）、rep<i>_per_pert.parquet（逐扰动）
      给出划分文件时只用其 test 扰动（加全部对照），输出到 results/n<N>_<划分文件名>/
"""
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl

from cell_eval2.anchor import _derive_seeds, _inner_config, _lfc_nmae_names, _lfc_nmae_raw
from cell_eval2.catalog import resolve_metrics
from cell_eval2.ceiling import _disjoint_halves
from cell_eval2.config import EvalConfig
from cell_eval2.run import (_compute_de_side, _resolve_target_sum_from_control, aggregate_metrics,
                            compute_metrics, metric_output_names)

ROOT = Path(__file__).parent
DATA = Path("/cluster2/home/hanwen/vcc_26/splithalf/data/H1.h5ad")      # 整数计数，VCC 2025 H1
CFG = EvalConfig.from_yaml("/cluster2/home/hanwen/vcc_26/backbone/cell_eval_vcc2026_base.yaml")
CTRL = CFG.control
SIX = ["pds_cosine", "expr_mse_unbiased_capped_norm", "de_wilcoxon_direction_fidelity_yield_raw",
       "de_wilcoxon_direction_reach_raw", "de_wilcoxon_sig_jaccard", "de_wilcoxon_lfc_nmae"]


def subsample(a, k, min_cells, seed):
    """每个扰动抽 k 个细胞（只保留原本 ≥ min_cells 的扰动）；对照全部保留。"""
    rng = np.random.default_rng(seed)
    keep = []
    for p, idx in a.obs.groupby(CFG.pert_col, observed=True).indices.items():
        if p == CTRL:
            keep.append(idx)
        elif idx.size >= min_cells:
            keep.append(rng.choice(idx, k, replace=False))
    return a[np.sort(np.concatenate(keep))].copy()


def score_once(sub, seed):
    """sub 对半分：half_b 当预测、half_a 当真值。返回 (六项等指标的均值 dict, 逐扰动 tidy 表)。"""
    metrics = list(resolve_metrics(CFG.metrics, version=CFG.version)[0])
    tidy = compute_metrics(*_disjoint_halves(sub, CFG.pert_col, CTRL, seed)[::-1], config=_inner_config(CFG))
    try:
        agg = aggregate_metrics(tidy, metrics=metrics)
    except ValueError as err:
        # 细胞太少时整个面板的 expr_distance_unbiased 之和 ≤ 0，MSE_norm 无定义，记为 NaN
        if "expr_mse_unbiased_capped_norm" not in str(err):
            raise
        agg = aggregate_metrics(tidy, metrics=[m for m in metrics if m != "expr_mse_unbiased_capped_norm"])
    vals = dict(zip(agg["metric"].to_list(), agg["mean"].to_list()))
    de_full = _compute_de_side(sub, cfg=_resolve_target_sum_from_control(CFG, sub), fp=None, store=None, side="real")
    for m in _lfc_nmae_names(metric_output_names(CFG)):
        try:
            vals[m] = _lfc_nmae_raw(sub, CFG, seed, de_full)[0]
        except ValueError as err:
            # 细胞太少（如 n = 1）时没有基因通过真值侧的显著性门控，lfc_nmae 无定义，记为 NaN
            if "scored no perturbation" not in str(err):
                raise
            vals[m] = np.nan
    return vals, tidy


def main():
    n = int(sys.argv[1])
    n_reps = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    min_cells = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    if 2 * n > min_cells:
        raise ValueError(f"需要 min_cells ≥ 2n（{2 * n}）")
    split = Path(sys.argv[4]) if len(sys.argv) > 4 else None
    out = ROOT / "results" / (f"n{n}_{split.stem}" if split else f"n{n}")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    a = ad.read_h5ad(DATA)
    if split:
        import json
        perts = json.loads((Path("/cluster2/home/hanwen/vcc_26/pertfold/data/h1/meta.json")).read_text())["perts"]
        keep = [perts[i] for i in json.loads(split.read_text())["test"]] + [CTRL]
        a = a[a.obs[CFG.pert_col].isin(keep).to_numpy()].copy()
    cnt = a.obs[CFG.pert_col].value_counts()
    print(f"H1 读入 {a.shape} ({time.time() - t0:.0f}s)；扰动 ≥{min_cells} 细胞：{(cnt[cnt.index != CTRL] >= min_cells).sum()}，"
          f"对照 {cnt[CTRL]}", flush=True)
    for i, s in enumerate(_derive_seeds(0, n_reps)):
        if (out / f"rep{i}_agg.csv").exists():
            continue
        t0 = time.time()
        vals, tidy = score_once(subsample(a, 2 * n, min_cells, seed=1000 + i), s)
        pd.DataFrame([{"n_cells_per_bulk": n, "rep": i, "seed": s, **vals}]).to_csv(out / f"rep{i}_agg.csv", index=False)
        tidy.with_columns(pl.lit(i).alias("rep")).write_parquet(out / f"rep{i}_per_pert.parquet")
        print(f"n={n} rep {i}: " + "  ".join(f"{m}={vals.get(m, np.nan):.4f}" for m in SIX)
              + f"  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
