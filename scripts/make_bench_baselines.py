"""benchmark 基线（2026-10-22 晚，用户定：强基线 = kNN + A-E 线性；基因范围两套都报）。打分空间 δ，5 折，
供体 / 训练扰动 = 本折全部非测试扰动（两个基线都没有要在内部验证集上选的超参数）。抽样同 K_S1c修正 的预测文件第 r 次重复。

  BL_knn   并集邻居（STRING ≥ 0.7 + K562 Perturb-seq + DepMap，各前 5，轮流取，最多 10 个）的打分空间群体 δ 均匀平均
  BL_ae    Ahlmann-Eltze 等（Nat Methods 2025）的线性基线，照 benchmark/src/run_linear_pretrained_model.R：
             Y（基因 × 训练扰动的 δ）− 行均值 = G K P，K = (GᵀG + λI)⁻¹ Gᵀ Yc Pᵀ (PPᵀ + λI)⁻¹，λ = 0.1；
             G = 训练扰动伪 bulk 表达（基因 × 扰动，基因作观测）的 PCA 10 维得分；
             P = K562 全基因组 Perturb-seq（Replogle GWPS 归一化 bulk）扰动 × 基因的 PCA 10 维得分（论文的跨体系预训练表征）；
             没有 K562 表征的扰动（训练时剔除，预测时只给行均值 = 训练平均）。
基因范围：expr = 对照 CPM ≥ 1（12,710 个，字段 dhat_all）；p2125 = 原模型基因（字段 dhat）。

用法：python make_bench_baselines.py → runs3/h1_bench/n32/pred_BL_{knn,ae}_{expr,p2125}_rep<r>.npz
"""
import json
from pathlib import Path

import anndata as ad
import numpy as np

D = Path("/cluster2/home/hanwen/vcc_26/pertfold/data")
SRC = "/cluster2/home/hanwen/vcc_26/pertfold/runs3/h1_v7k/n32/pred_K_S1c修正_stage2_ft_lr0.0001_best_pearson_exp_exp_rep{r}.npz"
OUT = Path("/cluster2/home/hanwen/vcc_26/pertfold/runs3/h1_bench/n32")
K562 = D / "relations/raw/K562_gwps_normalized_bulk_01.h5ad"
STRING_MIN, K_PER, NB_K = 0.7, 5, 10
PCA_DIM, RIDGE = 10, 0.1


def union_pool(P, train, srcs, p):
    tops = []
    for sc in srcs:
        v = np.full(P, -np.inf)
        v[train] = sc[p, train]
        v[p] = -np.inf
        o = np.argsort(-v, kind="stable")[:K_PER]
        tops.append([j for j in o if np.isfinite(v[j])])
    pool = []
    for r in range(K_PER):
        for t in tops:
            if r < len(t) and t[r] not in pool and len(pool) < NB_K:
                pool.append(t[r])
    return pool


def pca_scores(X, k, oversample=20, iters=4, seed=0):
    """X [n, m]，按列中心化，返回前 k 个主成分得分 [n, k]（同 prcomp 的 x）；随机截断 SVD（Halko 等），只算前 k 维"""
    Xc = X - X.mean(0, keepdims=True)
    rng = np.random.default_rng(seed)
    Q = np.linalg.qr(Xc @ rng.standard_normal((Xc.shape[1], k + oversample)))[0]
    for _ in range(iters):                                                       # 幂迭代，提高前几个成分的精度
        Q = np.linalg.qr(Xc @ (Xc.T @ Q))[0]
    U, S, _ = np.linalg.svd(Q.T @ Xc, full_matrices=False)
    return (Q @ U)[:, :k] * S[:k]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = json.loads((D / "h1/meta.json").read_text())
    perts, P = meta["perts"], len(meta["perts"])
    z = np.load(D / "h1/bulk_sums_all.npz")
    B, dB = z["B"], z["dB"]                                                       # [P+1, 18080]、[P, 18080]
    cpm = z["S"][-1] / z["tot"][-1] * 1e6
    sets = {"expr": np.where(cpm >= 1.0)[0], "p2125": np.array(meta["gene_col"])}
    fold_of = np.zeros(P, int)
    for k, fo in enumerate(meta["folds"]):
        fold_of[fo] = k
    direct = np.load(D / "string/h1_graph.npz")["direct"]
    srcs = [np.where(direct >= STRING_MIN, direct, -np.inf)]
    for s in ("perturbseq_k562", "depmap"):
        r = np.load(D / f"relations/h1_{s}.npz")
        sc = np.nan_to_num(r["score"].astype(float), nan=-np.inf)
        srcs.append(np.where(sc >= float(r["min"]), sc, -np.inf))
    # K562 扰动表征（全部 K562 扰动上做 PCA，再取 H1 扰动对应的行）
    a = ad.read_h5ad(K562)
    Xk = np.clip(np.nan_to_num(np.asarray(a.X, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0), -10, 10)
    Pk = pca_scores(Xk, PCA_DIM)
    kg = np.array([s.split("_")[1] for s in a.obs.index])
    first = {}
    for i, g in enumerate(kg):
        first.setdefault(g, i)
    emb = np.zeros((P, PCA_DIM))
    has = np.zeros(P, bool)
    for p, g in enumerate(perts):
        if g in first:
            emb[p], has[p] = Pk[first[g]], True
    print(f"K562 表征：H1 扰动覆盖 {has.sum()}/{P}", flush=True)
    preds = {(m, gs): np.zeros((P, len(idx))) for m in ("knn", "ae") for gs, idx in sets.items()}
    for k in range(5):
        test = np.where(fold_of == k)[0]
        train = np.where(fold_of != k)[0]
        for gs, idx in sets.items():
            Y = dB[:, idx]
            for p in test:                                                           # kNN
                pool = union_pool(P, train, srcs, p)
                if pool:
                    preds[("knn", gs)][p] = Y[pool].mean(0)
            tr = train[has[train]]                                                   # A-E 线性
            G = pca_scores(B[tr][:, idx].T, PCA_DIM)                                 # 基因 × 10
            Yt = Y[tr].T                                                             # 基因 × 训练扰动
            center = Yt.mean(1, keepdims=True)
            Pt = emb[tr].T                                                           # 10 × 训练扰动
            K = np.linalg.solve(G.T @ G + RIDGE * np.eye(PCA_DIM), G.T @ (Yt - center) @ Pt.T) @ np.linalg.inv(Pt @ Pt.T + RIDGE * np.eye(PCA_DIM))
            for p in test:
                preds[("ae", gs)][p] = (center[:, 0] + (G @ K @ emb[p] if has[p] else 0.0))
        print(f"第 {k} 折完成", flush=True)
    n_var = dB.shape[1]
    for r in range(5):
        zz = np.load(SRC.format(r=r))
        keep = {kk: zz[kk] for kk in ("perts", "x_idx", "real_idx", "ctrl_pred", "ctrl_real", "xbar")}
        pi = zz["perts"]
        for (m, gs), pr in preds.items():
            if gs == "expr":
                full = np.zeros((len(pi), n_var), np.float32)
                full[:, sets["expr"]] = pr[pi]
                np.savez(OUT / f"pred_BL_{m}_{gs}_rep{r}.npz", **keep, dhat_all=full, space="bulk")
            else:
                np.savez(OUT / f"pred_BL_{m}_{gs}_rep{r}.npz", **keep, dhat=pr[pi].astype(np.float32), space="bulk")
    # 诊断：逐扰动相关（去本扰动靶基因）与 Systema 式中心化相关（预测与真值都减去本折训练平均）
    tg = {g: i for i, g in enumerate(json.loads((D / "h1/var_names.json").read_text()))}
    for gs, idx in sets.items():
        pos = {g: j for j, g in enumerate(idx)}
        for m in ("knn", "ae"):
            r_raw, r_cen, ratio = [], [], []
            for p in range(P):
                y = dB[p, idx].copy()
                yh = preds[(m, gs)][p].copy()
                keepg = np.ones(len(idx), bool)
                t = tg.get(perts[p])
                if t is not None and t in pos:
                    keepg[pos[t]] = False
                mu = dB[np.where(fold_of != fold_of[p])[0]][:, idx].mean(0)
                if np.abs(yh[keepg]).sum() == 0:
                    continue
                r_raw.append(np.corrcoef(yh[keepg], y[keepg])[0, 1])
                r_cen.append(np.corrcoef((yh - mu)[keepg], (y - mu)[keepg])[0, 1])
                ratio.append(np.linalg.norm(y[keepg]) / max(np.linalg.norm(yh[keepg]), 1e-12))
            print(f"{gs:5s} {m:3s}：逐扰动 pearson {np.nanmean(r_raw):.3f}，中心化（去训练平均）{np.nanmean(r_cen):.3f}，"
                  f"真值/预测范数比中位 {np.median(ratio):.2f}（n={len(r_raw)}）", flush=True)


if __name__ == "__main__":
    main()
