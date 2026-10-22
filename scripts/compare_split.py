"""在某个划分的 test 扰动上汇总 cell-eval 六项（.venv-eval）：各方法 5 次重复的均值 ± 标准差。

每一行由逐扰动表（*_per_pert.parquet）筛出 test 扰动后汇总。注意 PDS 依赖参与评分的扰动集合，
各方法都应直接在 test 扰动上评分（--split 预测、subset_preds.py、benchmark_h1.py 的划分参数），筛选只作保险。
"理论最好"一行用 benchmark 在 test 扰动上的结果（results/n32_<划分文件名>）。
用法：python compare_split.py <划分文件> <输出名> "<方法名>=<per_pert.parquet 通配符>" ...
      → 打印表格，写 report/<输出名>.csv
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from benchmark_h1 import CFG
from cell_eval2.catalog import resolve_metrics
from cell_eval2.run import aggregate_metrics

SIX = {"pds_cosine": "PDS↑", "expr_mse_unbiased_capped_norm": "MSE_norm↓", "de_wilcoxon_direction_fidelity_yield_raw": "fidelity↑",
       "de_wilcoxon_direction_reach_raw": "reach↑", "de_wilcoxon_sig_jaccard": "sig_jacc↑", "de_wilcoxon_lfc_nmae": "lfc_nmae↓"}
META = Path("/cluster2/home/hanwen/vcc_26/pertfold/data/h1/meta.json")


def agg(f, names, metrics):
    t = pl.read_parquet(f)
    t = t.drop("rep") if "rep" in t.columns else t
    t = t.filter(pl.col("perturbation").is_in(names))
    try:
        a = aggregate_metrics(t, metrics=metrics)
    except ValueError:
        a = aggregate_metrics(t, metrics=[m for m in metrics if m != "expr_mse_unbiased_capped_norm"])
    return dict(zip(a["metric"].to_list(), a["mean"].to_list())), t["perturbation"].n_unique()


def main():
    perts = json.loads(META.read_text())["perts"]
    names = [perts[i] for i in json.loads(Path(sys.argv[1]).read_text())["test"]]
    metrics = list(resolve_metrics(CFG.metrics, version=CFG.version)[0])
    rows = []
    bench = str(Path(__file__).parent / f"results/n32_{Path(sys.argv[1]).stem}/rep*_per_pert.parquet")
    for spec in [f"理论最好：真实 32 细胞（benchmark）={bench}"] + sys.argv[3:]:
        name, pat = spec.split("=", 1)
        fs = sorted(glob.glob(pat))
        if not fs:
            print(f"（{name}：没有文件 {pat}）")
            continue
        res = [agg(f, names, metrics) for f in fs]
        row = {"方法": name, "次数": len(fs), "扰动数": res[0][1]}
        for m, lab in SIX.items():
            v = np.array([r[0].get(m, np.nan) for r in res], float)
            row[lab] = f"{np.nanmean(v):.3f} ± {np.nanstd(v):.3f}"
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(Path(__file__).parent / f"report/{sys.argv[2]}.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
