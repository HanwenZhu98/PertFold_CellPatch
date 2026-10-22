"""5 折（147 个扰动）汇总：各方法六项（5 次重复平均）、相对 STRING kNN 的配对 ΔPDS、逐扰动 bootstrap 95% 置信区间。
用法（.venv-eval）：python summarize_kfold.py <输出名> "名称=预测前缀" ...
  预测前缀 = pred_*.npz 去掉 _rep<i>.npz 的部分；参照为 pertfold/runs3/h1_v3 的 5 折 STRING kNN（同一套折）。
输出 scripts/report/<输出名>.csv
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

KNN = "/cluster2/home/hanwen/vcc_26/pertfold/runs3/h1_v3/n32/pred_string_knn"
M = {"pds_cosine": "PDS", "expr_mse_unbiased_capped_norm": "MSE", "de_wilcoxon_direction_fidelity_yield_raw": "fidelity",
     "de_wilcoxon_direction_reach_raw": "reach", "de_wilcoxon_sig_jaccard": "sig_jacc", "de_wilcoxon_lfc_nmae": "lfc_nmae"}


def load(prefix):
    fs = sorted(glob.glob(prefix + "_rep?_agg.csv"))
    if not fs:
        raise SystemExit(f"找不到 {prefix}_rep?_agg.csv")
    agg = pd.concat([pd.read_csv(f) for f in fs])[list(M)].rename(columns=M).reset_index(drop=True)
    pp = pd.concat([pd.read_parquet(f.replace("_agg.csv", "_per_pert.parquet")) for f in fs])
    return agg, pp.groupby(["perturbation", "metric"]).value.mean().unstack()


def main():
    name, arms = sys.argv[1], dict(a.split("=", 1) for a in sys.argv[2:])
    k_agg, k_pp = load(KNN)
    rng = np.random.default_rng(0)
    rows = [{"方法": "STRING kNN", **k_agg.mean().round(3).to_dict()}]
    for a, p in arms.items():
        agg, pp = load(p)
        x = (pp["pds_cosine"] - k_pp["pds_cosine"]).dropna().values
        bs = [rng.choice(x, len(x)).mean() for _ in range(5000)]
        rows.append({"方法": a, **agg.mean().round(3).to_dict(),
                     "ΔPDS 配对 kNN": round(float((agg.PDS.values - k_agg.PDS.values).mean()), 3),
                     "逐扰动 95% 置信区间": f"[{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]"})
    t = pd.DataFrame(rows)
    print(t.to_string(index=False))
    out = Path(__file__).parent / "report"
    out.mkdir(exist_ok=True)
    t.to_csv(out / f"{name}.csv", index=False)


if __name__ == "__main__":
    main()
