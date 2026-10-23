"""多个预测的平均（集成，2026-10-23 夜）：用法 python make_ensemble.py <输出 stem> <输入 stem1> <输入 stem2> ...（同一目录 runs3/h1_v12/n32，各 5 次重复）"""
import sys
from pathlib import Path
import numpy as np
V = Path("/cluster2/home/hanwen/vcc_26/pertfold/runs3/h1_v12/n32")
dst, srcs = sys.argv[1], sys.argv[2:]
for r in range(5):
    zs = [dict(np.load(V / f"{s}_rep{r}.npz")) for s in srcs]
    assert all(np.array_equal(z["perts"], zs[0]["perts"]) and np.array_equal(z["x_idx"], zs[0]["x_idx"]) for z in zs)
    out = dict(zs[0]); out["dhat_all"] = np.mean([z["dhat_all"] for z in zs], 0).astype(np.float32)
    np.savez(V / f"{dst}_rep{r}.npz", **out)
print("ok", dst, "←", srcs)
