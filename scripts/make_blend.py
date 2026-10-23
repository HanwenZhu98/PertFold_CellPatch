"""预测混合（2026-10-23 夜，自主优化）：δ_blend = a·A + (1 − a)·c·B，c = 所有扰动上 ‖A‖ / ‖B‖ 的中位数（只用预测本身，不用真值）。
A、B 为同一套抽样的 5 折合并预测（dhat_all）。s 之后照常交叉拟合。
用法：python make_blend.py <A 路径 stem> <B 路径 stem> <a> <输出 stem（写到 runs3/h1_v12/n32）>"""
import sys
from pathlib import Path
import numpy as np
A, B, a, dst = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
V = Path("/cluster2/home/hanwen/vcc_26/pertfold/runs3/h1_v12/n32")
for r in range(5):
    za, zb = dict(np.load(f"{A}_rep{r}.npz")), np.load(f"{B}_rep{r}.npz")
    assert np.array_equal(za["perts"], zb["perts"]) and np.array_equal(za["x_idx"], zb["x_idx"])
    xa, xb = za["dhat_all"], zb["dhat_all"]
    ok = (np.abs(xa).sum(1) > 0) & (np.abs(xb).sum(1) > 0)
    c = np.median(np.linalg.norm(xa[ok], axis=1) / np.linalg.norm(xb[ok], axis=1))
    za["dhat_all"] = (a * xa + (1 - a) * c * xb).astype(np.float32)
    np.savez(V / f"{dst}_rep{r}.npz", **za)
print("ok", dst, "c =", round(float(c), 3))
