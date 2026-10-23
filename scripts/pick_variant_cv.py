"""在多个预测变体 × s 网格上做 5 折交叉拟合选择（2026-10-23 上午）：
每折 k 在其余 4 折的扰动上（第 0 次重复的 s 网格文件）选综合分最高的（变体, s），用到第 k 折。
然后把各折选中的变体拼成一份预测（同一套抽样，逐扰动取对应变体的 dhat_all），写出 5 次重复的 npz 与每折 s，
供 eval_bulk_h1.py --fold-scale 打分。这样从多个变体里挑最好的也不会用到测试折。
用法：python pick_variant_cv.py <输出 stem> <变体 stem 1> <变体 stem 2> ...（都在 runs3/h1_v12/n32）
打印每折的选择；最后一行是 --fold-scale 的参数。
"""
import json
import sys
from pathlib import Path

import numpy as np

import summarize_v10 as S

V = Path("/cluster2/home/hanwen/vcc_26/pertfold/runs3/h1_v12/n32")
GRID = [1, 1.5, 2, 3, 4, 6, 8]
meta = json.loads(Path("/cluster2/home/hanwen/vcc_26/pertfold/data/h1/meta.json").read_text())
perts, folds = meta["perts"], meta["folds"]


def main():
    out, stems = sys.argv[1], sys.argv[2:]
    W = {}
    for st in stems:
        for s in GRID:
            f = V / f"{st}_rep0_s{s:g}_per_pert.parquet"
            if f.exists():
                W[(st, s)] = S.wide(str(f))
    print("可选（变体, s）：", len(W), flush=True)
    pick = []
    for k in range(5):
        other = [perts[i] for kk, fo in enumerate(folds) if kk != k for i in fo]
        best = max(W, key=lambda key: S.composite(S.six(W[key], other))[0])
        pick.append(best)
        print(f"第 {k} 折：{best[0]}，s = {best[1]:g}（其余 4 折 {S.composite(S.six(W[best], other))[0]:.3f}）", flush=True)
    own = [None] * len(perts)
    for k, fo in enumerate(folds):
        for i in fo:
            own[i] = pick[k][0]
    for r in range(5):
        zs = {st: np.load(V / f"{st}_rep{r}.npz") for st in set(p[0] for p in pick)}
        base = dict(next(iter(zs.values())))
        for z in zs.values():
            assert np.array_equal(z["perts"], base["perts"]) and np.array_equal(z["x_idx"], base["x_idx"])
        x = np.stack([zs[own[int(p)]]["dhat_all"][j] for j, p in enumerate(base["perts"])])
        base["dhat_all"] = x.astype(np.float32)
        np.savez(V / f"{out}_rep{r}.npz", **base)
    print(",".join(f"{p[1]:g}" for p in pick))


if __name__ == "__main__":
    main()
