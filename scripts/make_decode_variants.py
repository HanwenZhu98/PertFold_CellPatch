"""解码变体（2026-10-23 夜，自主优化）：δ_out = δ̂ − λ·训练平均（本折全部非测试扰动的打分空间群体 δ，在输出基因上）。
s 之后照常由交叉拟合选。只对有预测（非全 0）的扰动扣除。
用法：python make_decode_variants.py <输入 stem（不含 _rep）> <λ> <输出 stem> [目录，默认 runs3/h1_v12/n32]"""
import json, sys
from pathlib import Path
import numpy as np
D = Path("/cluster2/home/hanwen/vcc_26/pertfold")
src, lam, dst = sys.argv[1], float(sys.argv[2]), sys.argv[3]
V = Path(sys.argv[4]) if len(sys.argv) > 4 else D / "runs3/h1_v12/n32"
meta = json.loads((D / "data/h1/meta.json").read_text()); P = len(meta["perts"])
fold_of = np.zeros(P, int)
for k, fo in enumerate(meta["folds"]): fold_of[fo] = k
dB = np.load(D / "data/h1/bulk_sums_all.npz")["dB"]
MU = {k: dB[fold_of != k].mean(0) for k in range(5)}
for r in range(5):
    z = dict(np.load(V / f"{src}_rep{r}.npz")); full = z["dhat_all"].copy()
    nz = np.abs(full) > 0                                                      # 只在模型的输出基因上扣
    for i, p in enumerate(z["perts"]):
        if nz[i].any():
            full[i] = np.where(nz[i], full[i] - lam * MU[fold_of[p]], 0.0)
    z["dhat_all"] = full.astype(np.float32)
    np.savez(V / f"{dst}_rep{r}.npz", **z)
print("ok", dst)
