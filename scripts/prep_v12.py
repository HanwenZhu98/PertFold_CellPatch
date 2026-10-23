"""v12 数据准备（2026-10-23，用户要求：r4-v2 主体 + 修好的内容 + 全基因 ZINB + 弱 DE + 对照重构 / MMD / 判别器；split_811 两个种子）。

输出基因 = 对照 CPM ≥ 1 的基因 ∪ 原模型的 2,125 个基因（列号按 H1.h5ad 的 var 顺序）。只用对照细胞决定，不涉及任何扰动细胞。

写出 data/h1/v12/：
  genes.npy        输出基因在全部 18,080 个基因中的列号 [G_out]
  counts.npy       int16 [N, G_out]：每个扰动随机最多 256 个细胞 + 6,000 个对照细胞的原始计数（只取输出基因）
  cells.npz        group [N]（扰动编号，对照 = -1）、lib [N]（全部基因的总计数）、cell [N]（H1.h5ad 行号）
  labels.npz       dB [P, G_out]（打分空间群体 δ）、bc [G_out]（全部对照的打分空间伪 bulk）、
                   de / tested / lfc [P, G_out]（de_full.parquet：全部细胞的 Wilcoxon，p_adj < 0.05；log2FC）
  pca.npz          只用缓存的对照细胞拟合的 PCA（log1p(1e4 · 计数 / 文库)，输出基因）：mean [G_out]、comp [G_out, 50]
"""
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

D = Path("/cluster2/home/hanwen/vcc_26/pertfold/data/h1")
OUT = D / "v12"
N_PER, N_CTRL, N_PC, SEED = 256, 6000, 50, 0


def main():
    OUT.mkdir(exist_ok=True)
    meta = json.loads((D / "meta.json").read_text())
    perts, P = meta["perts"], len(meta["perts"])
    z = np.load(D / "bulk_sums_all.npz")
    cpm = z["S"][-1] / z["tot"][-1] * 1e6
    genes = np.union1d(np.where(cpm >= 1.0)[0], np.array(meta["gene_col"]))
    np.save(OUT / "genes.npy", genes)
    a = ad.read_h5ad("/cluster2/home/hanwen/vcc_26/splithalf/data/H1.h5ad")
    C = a.X.tocsr()
    var = list(a.var_names)
    tgt = np.array(a.obs["target"].astype(str))
    pid = {p: i for i, p in enumerate(perts)}
    rng = np.random.default_rng(SEED)
    rows, group = [], []
    for p in perts:
        idx = np.where(tgt == p)[0]
        idx = np.sort(rng.choice(idx, min(N_PER, idx.size), replace=False))
        rows.append(idx); group += [pid[p]] * idx.size
    cidx = np.sort(rng.choice(np.where(tgt == "non-targeting")[0], N_CTRL, replace=False))
    rows.append(cidx); group += [-1] * cidx.size
    rows = np.concatenate(rows)
    lib = np.asarray(C[rows].sum(1)).ravel().astype(np.float32)
    X = C[rows][:, genes].toarray()
    assert X.max() < 32767
    np.save(OUT / "counts.npy", X.astype(np.int16))
    np.savez(OUT / "cells.npz", group=np.array(group, np.int32), lib=lib, cell=rows)
    print(f"输出基因 {genes.size}；缓存细胞 {rows.size}（对照 {cidx.size}）；counts {X.shape} int16 {X.nbytes / 2**30 * 0.5:.2f} GB", flush=True)
    # 标签
    gpos = {var[g]: j for j, g in enumerate(genes)}
    de = np.zeros((P, genes.size), bool); tested = np.zeros_like(de); lfc = np.full((P, genes.size), np.nan, np.float32)
    df = pd.read_parquet(D / "de_full.parquet")
    df = df[df.target.isin(pid) & df.feature.isin(gpos)]
    r = df.target.map(pid).to_numpy(); c = df.feature.map(gpos).to_numpy()
    pa = df.p_adj.to_numpy(np.float64); ok = np.isfinite(pa)
    tested[r[ok], c[ok]] = True; de[r[ok], c[ok]] = pa[ok] < 0.05; lfc[r[ok], c[ok]] = df.log2_fold_change.to_numpy()[ok]
    np.savez(OUT / "labels.npz", dB=z["dB"][:, genes].astype(np.float32), bc=z["B"][-1, genes].astype(np.float32),
             de=de, tested=tested, lfc=lfc)
    print(f"DE 标签：已检验 {tested.mean():.3f}，每个扰动 DE 中位 {np.median(de.sum(1)):.0f}", flush=True)
    # 只用对照细胞拟合 PCA
    ci = np.array(group) == -1
    L = np.log1p(1e4 * X[ci].astype(np.float64) / lib[ci, None])
    mu = L.mean(0)
    _, S, Vt = np.linalg.svd(L - mu, full_matrices=False)
    np.savez(OUT / "pca.npz", mean=mu.astype(np.float32), comp=Vt[:N_PC].T.astype(np.float32),
             var_ratio=(S[:N_PC] ** 2 / (S ** 2).sum()).astype(np.float32))
    print(f"PCA（对照）：前 {N_PC} 个成分解释方差 {(S[:N_PC] ** 2).sum() / (S ** 2).sum():.3f}", flush=True)


if __name__ == "__main__":
    main()
