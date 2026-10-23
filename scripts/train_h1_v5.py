"""v10 参数（2026-10-22 晚，用户要求：统一到打分空间；loss 先 MSE + nb loss，再试 ZINB；默认 = v9）：
  --space bulk         标签、nbw 证据（邻居 δ）、c_k、输出都改用打分空间群体 δ（data/h1/bulk_sums.npz：全部细胞计数和 →
                       log1p(5e4 · 比例)，扰动 − 对照）；encoder 输入 x 不变。对照伪 bulk ≤ EXPR_MIN 的基因（约 25%）与本扰动靶基因
                       不进 loss（labels["gmask"]）。默认 dp（原 delta_p）
  --recon-loss mse|zinb  阶段 2 的 loss（自动打开 --cont-head）：mse = 打分空间 MSE；zinb = 该扰动 n 个细胞计数的 ZINB 似然
                       （θ 按对照细胞矩估计初始化）。都加原 nb 打分 loss；DE / LFC / 方向 / 分箱不再进 loss
  --select mse         阶段 2 按评估扰动上打分空间 MSE（gmask 格子，除以 s_dp²）最低保存 best（stage2_*_best_val_mse.pt）；
                       默认 pearson_exp（原做法）
  从 --from-stage1 载入时，stage1.pt 里没有的连续头 / zi 头 / log_theta 用新初始化（阶段 1 不训练 decoder），s_dp 按本空间重设
v6 参数（2026-10-20，用户要求：N1u + mul_std 底座上比较阶段 1 的预训练目标；默认全关 = v5）：
  --value-enc mul      token = 基因 ⊙ LN(AutoBin(x))；配 --value-norm --norm-bulk 即 mul_std
  --norm-bulk          A1 的标准化统计量改用 bulk：512 个对照 bulk + 512 个训练扰动 bulk（各 n 个细胞）的每基因均值 / 标准差
                       （下限 NORM_BULK_SD_MIN）；encoder 单独测试按"encoder 输入的总体"标准化，这里输入是 bulk。MAE 的重建目标也用它
  --jepa-mode sg       阶段 1 目标 = 在线 CellPatch(y) 停梯度 + MSE（arXiv 2610.07540 官方代码）；默认 ema（原 jepa）
  --idm λ              EP-IDM：D([池化 CellPatch(x), 池化 CellPatch(y)]) → 扰动类别（全部扰动编号，只有训练扰动出现），CE
  --jepa-sigreg λ      阶段 1 SIGReg（池化在线 latent，x 与 y 两端）
  --mask-targets       阶段 1（jepa）中 y 送入 encoder 前，全部 147 个靶基因列换成对照细胞均值（防止 IDM 只读靶基因认扰动）
  --mae [--mae-ratio 0.3]  阶段 1 改为 MAE：x（对照 bulk）与 y（扰动 bulk）各 B 个，遮住 30% 基因，重建被遮基因的标准化表达
  v7（PLAN_v7_1021.md）：--dec-type xattn|xattn_sa|affine  --ev-drop 可含 ctrl  --ctrl-jepa λ [--ctrl-jepa-ratio 0.3]  --input-mask r
  消融（R4 配方重训）：--no-encoder（需 --s1 0）--no-cell-msa --no-pert-msa --ev-drop ctx,str,nbw|all --const-z（见 model_v5.py）
  --init-cellpatch 路径  从 Tahoe 预训练（analysis/tahoe_pretrain）载入 CellPatch 权重（不含标准化缓冲区）
以下为 v5 说明。
v5 训练脚本（2026-10-18）：train_h1_v3.py 的副本，模型用 pertfold3/model_v5.py。新增（默认全关 = v3x）：
  --relation a+b+c   候选邻居池 = 各来源（阈值同 h1_<来源>.npz / STRING 0.7）前 5 个的并集，最多 NB_K 个（按来源轮流取）；
                     MSA 相关行与 str_* 证据（等权）都用这个池。单一来源时与 v3 相同（前 5 个）。
  --nb-learn         N1：邻居学习权重；关联特征 = 每个来源的 [归一化得分, 是否为该来源的邻居]
                     c_k = 查询扰动与邻居真实 delta_p 的 pearson（去掉靶基因列），只在训练 batch 中给（val / test 不用标签）
  --cont-head        N2：连续 δ 头 + MSE；diag 另记 pearson_cont，并按它另存 best_pearson_cont 权重
  --nb-direct        N3：连续头 + α · 加权邻居 δ
  --pert-var λ       P1：扰动一路跨样本 VICReg
  --aug-sigma σ      B1（2026-10-19）批次增强：每个训练样本抽一个"批次"，即每个基因的计数缩放 s_g，log s_g ~ N(0, σ²)（截断在 ±3），
                     每个细胞变换为 log1p(s ⊙ expm1(X))；x（对照）、y（扰动细胞）、标签 delta_p（群体均值之差，查表 + 线性插值，精确）
                     和对照证据（该批次的对照群体均值）都在同一批次空间中；其余证据（ctx / str / nbw）与邻居行保持原空间（"来自别的批次"）。
                     de_mask / lfc 不变（按基因缩放计数时，均值的 fold change 不变）。val 选权重在原空间进行。
  --slot-attn / --sigreg λ / --disc λ   塌缩修复 S / J / D（见 model_v5.py）；每次评估另记 RankMe（CellPatch 输出与扰动一路的有效秩）
  --ev-mode fc       B2：证据的 *_dp 块与邻居输入改用 fold change：log2((μ_p + ε) / (μ_ctrl + ε))，μ 为计数（CP10k）群体均值，ε = 0.05；
                     它在按基因缩放计数时不变。默认 delta（原做法）。邻居目标 c_k 仍用 delta_p 的相关。
以下为 v3 脚本说明。
v3 训练脚本（恢复版，2026-10-16）：2026-10-15 10:38 跑 runs3/h1_v3 时的 train_h1.py，原文从会话记录中恢复
（scratchpad 中的 train_h1_v3_original.py），模型用冻结的 pertfold3/model_v3.py（decoder 含 PIE 证据）。
相对原文只加了：--split（固定划分：train 训练、val 评估与选权重）、--eval-every、--seed、--relation（邻居来源，
同时决定 MSA 相关行与 str_* 证据，规则同 train_h1.set_neighbors）、阶段 2 按 val pearson_exp 保存 best 权重、data.json。
默认参数下与原 v3 完全相同（batch 32、有放回抽扰动、EVAL_EVERY 500）。
2026-10-16 起模型改用 pertfold3/model_v3x.py（model_v3 + 修复开关，默认全关 = 原 v3），并加了修复实验的参数：
  --value-norm（A1）--value-enc linear（A2）--patch-groups（A3）--freeze-genes（A5）--token-ln（A6）
  --var-weight λ（T2，jepa 的 VICReg 方差项）--relate-grad（T5）--fp32（T3，不用 bf16 autocast）--wd（T4）
  --sample-loss λ（T1：x 的对照细胞按测序深度与 y 的细胞配对，recon 再加 λ·MSE(期望值读出 δ̂, y − x)）
2026-10-18：--recon-weight λ（E1：encoder 重建对照 bulk）、--patch-var λ [--patch-var-gamma γ]（E2：CellPatch 输出跨样本防塌缩）。
模型开关写入 grid.json（predict_h1_v3.py 据此建模型），--fp32 写入 data.json。

H1 单细胞系、未见扰动：训练 PertFold3（vcc26 环境）。

一个样本（扰动 p，n = n_cells_per_bulk）：
  x            n 个随机对照细胞的平均（log1p CP10k）              [G]
  y            n 个扰动 p 的细胞的平均                              [G]
  relate_x     p 的 STRING 邻居扰动（本折训练扰动，直接分 ≥ 0.7，最多 5 个）各 n 个真实细胞的平均 [K, G]
  pert_gene    p 的靶基因
  relate_genes 同一批 STRING 邻居扰动的靶基因；没有邻居的位置屏蔽
训练：阶段 1 jepa（S1 步）→ 阶段 2 recon（PIE 的四项 loss，标签来自 label_h1.py；decoder 默认使用 PIE 的 context 证据），两个版本：freeze（只训 decoder）与 ft（全部微调），共用阶段 1 权重。
评估：每 EVAL_EVERY 步在评估扰动（--inner-val 时为验证集，否则为本折测试扰动）上算 pearson_delta 与 DE AUPRC。

用法：python train_h1.py <n_cells_per_bulk> <fold> [--gpu 0] [--s1 3000 --lr1 3e-4] [--s2 3000 --stage2 freeze:3e-4,ft:1e-4]
      [--inner-val 23（调参：在本折训练扰动里留出验证集）] [--tag 名字] [--save-steps 1000,2000]
输出：pertfold/runs3/h1/n<N>/fold<k>/{stage1.pt, stage2_freeze.pt, stage2_ft.pt, log.jsonl}
"""
import argparse
import dataclasses
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/cluster2/home/hanwen/vcc_26/pertfold_r4")
from pertfold3.model_v5 import Config3, PertFold3, dp_centers  # noqa: E402
import torch.nn.functional as F  # noqa: E402

DATA = Path("/cluster2/home/hanwen/vcc_26/pertfold/data/h1")
RUNS = Path("/cluster2/home/hanwen/vcc_26/pertfold_r4/runs")
ESM = Path("/cluster2/home/hanwen/vcc_26/pertfold/data/esm/esm2_t33_650M_human_all.pt")
BATCH = 32
S1, S2, WARMUP = 3000, 3000, 200
LR1, LR2_FREEZE, LR2_FT = 3e-4, 3e-4, 1e-4
EVAL_EVERY = 500
DP_BIN_FC = 1.1          # PIE replogle_wdataset：delta_p 箱宽 ln(1.1)
STRING_GRAPH = Path("/cluster2/home/hanwen/vcc_26/pertfold/data/string/h1_graph.npz")   # string_graph.py
RELATIONS = Path("/cluster2/home/hanwen/vcc_26/pertfold/data/relations")   # build_relations.py：h1_<名字>.npz
AMP = True               # bf16 autocast；--fp32 时关
WD = 0.01                # AdamW weight decay；--wd 改
SAMPLE_LOSS = 0.0        # T1：逐样本 loss 的权重（> 0 时 batch 按测序深度配对 x）
CTRL_SD_MIN = 0.05       # A1：对照标准差的下限（log1p 单位），避免几乎不表达的基因被放大
NB_K = 10                # 并集候选池的最大邻居数
STRING_MIN, STRING_K = 0.7, 5   # STRING 证据：直接分阈值与最多邻居数（string_knn.py 中最好的设置）
MASK_TARGETS = False     # v6：阶段 1 中 y 的靶基因列换成对照均值（--mask-targets）
NORM_BULK_SD_MIN = 0.01  # v6 --norm-bulk：bulk 标准差的下限（log1p 单位）
BULK_SUMS = DATA / "bulk_sums.npz"   # v10：打分空间群体伪 bulk（core1022/bulk_sums.py）
EXPR_MIN = 0.03          # v10：对照打分空间伪 bulk 不超过它的基因视为不表达（约 25% 的基因）


class H1Data:
    """全部数据放在 GPU 上；按 n 采样细胞并求平均。"""

    def __init__(self, dev):
        meta = json.loads((DATA / "meta.json").read_text())
        self.meta, self.dev = meta, dev
        self.X = torch.from_numpy(np.load(DATA / "X.npy")).to(dev)                  # [细胞, G]
        pert = np.load(DATA / "pert.npy")
        self.ctrl = torch.from_numpy(np.where(pert == -1)[0]).to(dev)
        self.cells = [torch.from_numpy(np.where(pert == i)[0]).to(dev) for i in range(len(meta["perts"]))]
        # 词表 = 模型基因 + 不在模型基因里的靶基因（只作为 pert 输入）
        genes = meta["genes"]
        self.vocab = genes + [p for p in meta["perts"] if p not in set(genes)]
        vidx = {g: i for i, g in enumerate(self.vocab)}
        self.gene_idx = torch.arange(len(genes), device=dev)
        self.pert_vocab = torch.tensor([vidx[p] for p in meta["perts"]], device=dev)
        self.esm = self._load_esm()
        lab = np.load(DATA / "labels.npz")                                         # label_h1.py：全部细胞的群体标签
        self.labels = {k: torch.from_numpy(lab[k]).to(dev) for k in ("de_mask", "tested", "lfc", "delta_p", "target")}
        self.use_evidence = False                                                  # set_evidence 之后由调用方打开
        self.use_nb = False                                                        # N1：batch 中给 nb 输入
        self.aug_sigma = 0.0                                                       # B1：批次增强（build_aug_tables 之后才可用）
        self.ev_mode = "delta"                                                     # B2：证据用 delta 或 fc
        self.depth_match = False                                                   # T1：x 的对照细胞按测序深度与 y 的细胞配对
        self.space = "dp"                                                          # v10：标签 / 证据空间（use_bulk 后为 bulk）
        self.need_counts = False                                                   # v10 zinb：batch 另给 y 细胞的计数与文库
        loglib = torch.from_numpy(np.log(np.load(DATA / "lib.npy"))).to(dev)
        self.loglib = loglib
        order = loglib[self.ctrl].argsort()
        self.ctrl_sorted, self.ctrl_lib_sorted = self.ctrl[order], loglib[self.ctrl][order].contiguous()

    def _load_esm(self):
        cache = torch.load(ESM, map_location="cpu", weights_only=False)
        pos = {str(g).upper(): i for i, g in enumerate(cache["genes"])}
        emb, found = cache["embeddings"].float(), cache["found_mask"].bool()
        out = torch.zeros(len(self.vocab), emb.shape[1])
        for i, g in enumerate(self.vocab):
            j = pos.get(g.upper())
            if j is not None and found[j]:
                out[i] = emb[j]
        n_ok = int((out.abs().sum(1) > 0).sum())
        n_pert_ok = int((out[self.pert_vocab.cpu()].abs().sum(1) > 0).sum())
        print(f"ESM：词表 {n_ok}/{len(self.vocab)} 个基因有表征；扰动靶基因 {n_pert_ok}/{len(self.pert_vocab)}", flush=True)
        return out

    def use_bulk(self):
        """v10：载入打分空间群体 δ（labels["dB"]）、对照伪 bulk bc [G]、表达掩码 expr [G]、原始文库 lib_raw [细胞]"""
        z = np.load(BULK_SUMS)
        self.labels["dB"] = torch.from_numpy(z["dB"].astype(np.float32)).to(self.dev)
        self.bc = torch.from_numpy(z["B"][-1].astype(np.float32)).to(self.dev)
        self.expr = self.bc > EXPR_MIN
        self.lib_raw = torch.from_numpy(np.load(DATA / "lib.npy").astype(np.float32)).to(self.dev)
        self.space = "bulk"

    def nb_theta(self, k=5000):
        """v10 zinb：每基因 NB 离散度的矩估计（k 个对照细胞）：1/θ = max(Σ[(c − Lf)² − Lf], 0) / Σ(Lf)²，f = Σc / ΣL"""
        idx = self.ctrl[torch.randperm(self.ctrl.numel(), device=self.dev)[:k]]
        L = self.lib_raw[idx][:, None]
        c = torch.round(torch.expm1(self.X[idx]) * L / 1e4)
        f = c.sum(0) / L.sum()
        mu = L * f
        inv = ((c - mu) ** 2 - mu).sum(0).clamp_min(0) / (mu ** 2).sum(0).clamp_min(1e-8)
        return (1.0 / inv.clamp_min(1e-4)).clamp(0.01, 1e4)

    def bulk(self, pool, k, n):
        """从 pool 中为 k 个 bulk 各随机抽 n 个细胞并平均 → [k, G]（同一 bulk 内不重复）"""
        idx = torch.rand(k, pool.numel(), device=self.dev).argsort(1)[:, :n] if pool.numel() < 4096 \
            else torch.randint(0, pool.numel(), (k, n), device=self.dev)
        return self.X[pool[idx]].mean(1)

    def relate(self, perts, n):
        """STRING 邻居（set_evidence 中的 str_idx / str_mask）：
        relate_genes [B, K] 邻居扰动的靶基因；relate_x [B, K, G] 每个邻居扰动 n 个真实细胞的平均；无效位置为 0 并屏蔽"""
        idx, mask = self.str_idx[perts], self.str_mask[perts]                     # [B, K]
        rx = torch.zeros(*idx.shape, self.X.shape[1], device=self.dev)
        for b, k in mask.nonzero().tolist():
            rx[b, k] = self.bulk(self.cells[int(idx[b, k])], 1, n)[0]
        return rx, mask, self.pert_vocab[idx]

    def matched(self, perts, n, jitter=20):
        """T1：每个扰动抽 n 个细胞作 y；每个 y 细胞配一个测序深度最接近的对照细胞（排序位置 ± jitter 内随机）作 x"""
        ys, xs = [], []
        for p in perts:
            pool = self.cells[int(p)]
            yi = pool[torch.rand(pool.numel(), device=self.dev).argsort()[:n]]
            pos = torch.searchsorted(self.ctrl_lib_sorted, self.loglib[yi])
            pos = (pos + torch.randint(-jitter, jitter + 1, pos.shape, device=self.dev)).clamp(0, self.ctrl_sorted.numel() - 1)
            ys.append(self.X[yi].mean(0))
            xs.append(self.X[self.ctrl_sorted[pos]].mean(0))
        return torch.stack(xs), torch.stack(ys)

    def nb_inputs(self, perts, with_c):
        """N1：perts 的候选邻居（池）的标签统计 [B, K, G]（与证据同样归一化）、mask、关联特征；with_c 时给 c_k"""
        idx, mask = self.str_idx[perts], self.str_mask[perts]
        d = {"dp": self.dp_n[idx], "lfc": self.lfc_n[idx], "up": self.up_f[idx], "down": self.down_f[idx],
             "tested": self.labels["tested"][idx], "mask": mask, "feat": self.nb_feat[perts]}
        if with_c:
            d["c"] = self.Cmat[perts[:, None], idx]
        return d

    def build_aug_tables(self):
        """B1：每组细胞（每个扰动、最后一组为对照）每个基因的群体均值 M[组, g, j] = mean_cells log1p(exp(grid_j) · expm1(X))，
        以及计数群体均值 mu_counts[组, g]（B2 用）"""
        grid = torch.linspace(-3.0, 3.0, 61, device=self.dev)
        groups = self.cells + [self.ctrl]
        T = torch.zeros(len(groups), self.X.shape[1], grid.numel(), device=self.dev)
        mu = torch.zeros(len(groups), self.X.shape[1], device=self.dev)
        sc = grid.exp()
        with torch.no_grad():
            for i, cells in enumerate(groups):
                for ch in cells.split(2048):
                    e = torch.expm1(self.X[ch])
                    T[i] += torch.log1p(e[:, :, None] * sc).sum(0)
                    mu[i] += e.sum(0)
                T[i] /= cells.numel()
                mu[i] /= cells.numel()
        self.aug_grid, self.aug_T, self.mu_counts = grid, T, mu

    def aug_mean(self, groups, ls):
        """B1：groups [B]（组编号；len(self.cells) = 对照），ls [B, G] → 该批次下的群体均值 [B, G]（在 grid 上线性插值）"""
        g = self.aug_grid
        u = ((ls - g[0]) / (g[1] - g[0])).clamp(0, g.numel() - 1 - 1e-4)
        lo = u.floor().long()
        fr = u - lo
        T = self.aug_T[groups]
        a = T.gather(2, lo[..., None]).squeeze(-1)
        b = T.gather(2, (lo + 1)[..., None]).squeeze(-1)
        return a + fr * (b - a)

    def bulk_aug(self, pool, n, ls):
        """B1：从 pool 抽 n 个细胞，按批次 ls [G] 变换后平均 → [G]"""
        idx = pool[torch.rand(pool.numel(), device=self.dev).argsort()[:n]] if pool.numel() < 4096 \
            else pool[torch.randint(0, pool.numel(), (n,), device=self.dev)]
        return torch.log1p(torch.expm1(self.X[idx]) * ls.exp()).mean(0)

    def batch(self, perts, n, train=False):
        """perts [B]（扰动编号）→ 模型输入 dict、y（n 个扰动细胞的平均，jepa 用）、群体标签（PIE loss 用）"""
        B = perts.numel()
        aug = train and self.aug_sigma > 0
        if aug:                                                                  # B1：每个样本一个随机批次
            G = self.X.shape[1]
            ls = (torch.randn(B, G, device=self.dev) * self.aug_sigma).clamp(-3.0, 3.0)
            x = torch.stack([self.bulk_aug(self.ctrl, n, ls[b]) for b in range(B)])
            y = torch.stack([self.bulk_aug(self.cells[int(p)], n, ls[b]) for b, p in enumerate(perts)])
        elif self.depth_match:
            x, y = self.matched(perts, n)
        else:
            x = None
            if self.need_counts:                                                 # v10 zinb：同一批 n 个细胞给 y（log 平均）、计数与文库
                idx = torch.stack([self.cells[int(p)][torch.randperm(self.cells[int(p)].numel(), device=self.dev)[:n]] for p in perts])
                Xi, Li = self.X[idx], self.lib_raw[idx]                          # [B, n, G]、[B, n]
                y = Xi.mean(1)
                counts = torch.round(torch.expm1(Xi) * Li[..., None] / 1e4)
            else:
                y = torch.stack([self.bulk(self.cells[int(p)], 1, n)[0] for p in perts])
        lab = {k: v[perts] for k, v in self.labels.items()}
        lab["pid"] = perts                                                       # D：区分度 loss 用（屏蔽同一扰动）
        if not hasattr(self, "nontarget"):
            self.nontarget = torch.ones(self.X.shape[1], dtype=torch.bool, device=self.dev)
            t = self.labels["target"]
            self.nontarget[t[t >= 0]] = False
        lab["nontarget"] = self.nontarget
        if self.space == "bulk":                                                 # v10：表达且不是本扰动靶基因的格子
            G = self.X.shape[1]
            t = self.labels["target"][perts]
            lab["gmask"] = self.expr[None, :] & (torch.arange(G, device=self.dev)[None, :] != t[:, None])
            lab["bc"] = self.bc
            if self.need_counts and not aug and not self.depth_match:
                lab["counts"], lab["lib"] = counts, Li
        if aug:
            ctrl_b = self.aug_mean(torch.full((B,), len(self.cells), device=self.dev), ls)
            dpa = self.aug_mean(perts, ls) - ctrl_b
            lab["delta_p"] = torch.where(torch.isfinite(lab["delta_p"]), dpa, lab["delta_p"])
        rx, rmask, rgenes = self.relate(perts, n)
        inp = dict(gene_idx=self.gene_idx, x=self.bulk(self.ctrl, B, n) if x is None else x,
                   relate_x=rx, relate_x_mask=rmask, pert_gene=self.pert_vocab[perts],
                   relate_genes=rgenes, relate_genes_mask=rmask)
        if self.use_evidence:
            inp["evidence"] = self.ev_batch(perts)
            if aug:
                inp["evidence"]["ctrl"] = ctrl_b                                 # 对照证据 = 该批次的对照群体均值
        if self.use_nb:
            inp["nb"] = self.nb_inputs(perts, with_c=train)
        return inp, y, lab

    def set_evidence(self, train, relation="string"):
        """PIE 式证据：每个扰动、每个基因，一组"供体"扰动的统计（供体只取训练扰动、不含自己）。

        两类供体（贡献矩阵 W [P, P]，W[p, j] = 1 表示 j 是 p 的供体）：
          ctx_*  本细胞系全部其他训练扰动（训练扰动即留一）
          str_*  STRING 直接分 ≥ STRING_MIN 的训练扰动，按分数取前 STRING_K 个（没有则 have = 0）
        每类三块，每块 [值, 离散度, have, log1p 计数]：
          *_dp   [mean/s_dp, std/s_dp, ...]        delta_p
          *_de   [up/t, down/t, ...]               DE 中上调 / 下调的比例，t = 已检验数
          *_lfc  [mean/s_lfc, std/s_lfc, ...]      已检验基因的 log2FC，截断在训练集 |lfc| 的 95 分位
        s_dp、s_lfc：训练扰动全部 (扰动, 基因) 上的总体标准差；std 在计数 < 2 时为 0。另存对照均值 ctrl [G]。
        """
        L = self.labels
        P = L["delta_p"].shape[0]
        intr = torch.zeros(P, dtype=torch.bool, device=self.dev)
        intr[train] = True
        eye = torch.eye(P, dtype=torch.bool, device=self.dev)
        W_ctx = (intr[None, :] & ~eye).double()
        sources = relation.split("+")
        scores, tops = [], []
        for src in sources:
            if src == "string":                                                  # 原 v3
                direct = torch.from_numpy(np.load(STRING_GRAPH)["direct"]).to(self.dev)
                sc = torch.where(intr[None, :] & ~eye & (direct >= STRING_MIN), direct, torch.full_like(direct, -np.inf))
            else:                                                                # 其他关联来源：同 train_h1.set_neighbors
                rel = np.load(RELATIONS / f"h1_{src}.npz")
                assert list(rel["perts"]) == self.meta["perts"], "关联矩阵的扰动顺序与 meta.json 不一致"
                score = torch.from_numpy(np.nan_to_num(rel["score"].astype(np.float64), nan=-np.inf)).to(self.dev)
                sc = torch.where(intr[None, :] & ~eye & (score >= float(rel["min"])), score, torch.full_like(score, -np.inf))
            scores.append(sc)
            tops.append(sc.topk(STRING_K, dim=1))
        if len(sources) == 1:                                                    # 单一来源：与 v3 相同
            top = tops[0]
            idx, ok = top.indices, torch.isfinite(top.values)
        else:                                                                    # 并集池：按来源轮流取各自的第 r 名，去重，最多 NB_K 个
            idx = torch.zeros(P, NB_K, dtype=torch.long, device=self.dev)
            ok = torch.zeros(P, NB_K, dtype=torch.bool, device=self.dev)
            ti, tv = [t.indices.tolist() for t in tops], [t.values.tolist() for t in tops]
            for p in range(P):
                seen = []
                for r in range(STRING_K):
                    for si in range(len(sources)):
                        j = ti[si][p][r]
                        if np.isfinite(tv[si][p][r]) and j not in seen and len(seen) < NB_K:
                            seen.append(j)
                idx[p, :len(seen)] = torch.tensor(seen, dtype=torch.long, device=self.dev) if seen else idx[p, :0]
                ok[p, :len(seen)] = True
        W_str = torch.zeros(P, P, dtype=torch.float64, device=self.dev)
        W_str.scatter_(1, idx, ok.double())
        # N1 的关联特征：每个来源 [该来源得分 / 该来源有效得分的最大值, 是否为该来源的前 5 邻居]
        feats = []
        for si, sc in enumerate(scores):
            fin = torch.isfinite(sc)
            mx = sc[fin].abs().max().clamp_min(1e-6) if fin.any() else torch.tensor(1.0, device=self.dev)
            v = torch.where(fin, sc / mx, torch.zeros_like(sc)).gather(1, idx)
            isn = torch.zeros(P, P, dtype=torch.bool, device=self.dev)
            isn.scatter_(1, tops[si].indices, torch.isfinite(tops[si].values))
            feats += [v.float() * ok, isn.gather(1, idx).float() * ok]
        self.nb_feat = torch.stack(feats, -1)                                    # [P, K, 2·来源数]

        def stats(v, valid, W):
            vv = torch.where(valid, v, torch.zeros_like(v)).double()
            w = valid.double()
            s1, s2, n = W @ (vv * w), W @ (vv * vv * w), W @ w                  # [P, G]
            mean = torch.where(n > 0, s1 / n.clamp(min=1), torch.zeros_like(s1))
            std = torch.where(n >= 2, (s2 / n.clamp(min=1) - mean ** 2).clamp(min=0).sqrt(), torch.zeros_like(s1))
            return mean, std, n

        def pop_std(v, valid):
            x = v[train][valid[train]].double()
            sd = x.std(unbiased=False).item() if x.numel() else 1.0
            return sd if sd > 0 else 1.0

        dp = L["dB"] if self.space == "bulk" else L["delta_p"]                  # v10：打分空间
        dp_ok = torch.isfinite(dp)
        dp = torch.nan_to_num(dp)
        dp_delta = dp
        if self.ev_mode == "fc":                                                 # B2：证据改用计数均值的 log2 fold change
            mu = self.mu_counts
            dp = torch.log2((mu[:-1] + 0.05) / (mu[-1:] + 0.05)) * dp_ok
        tested = L["tested"]
        lfc = torch.nan_to_num(L["lfc"])
        clip = torch.quantile(lfc[train][tested[train]].abs().float(), 0.95).item()
        lfc = lfc.clamp(-clip, clip)
        s_dp, s_lfc = pop_std(dp, dp_ok), pop_std(lfc, tested)
        up = (L["de_mask"] & (L["lfc"] > 0)).double()
        down = (L["de_mask"] & (L["lfc"] < 0)).double()
        ev = {}
        for fam, W in (("ctx", W_ctx), ("str", W_str)):
            m, sd, n = stats(dp, dp_ok, W)
            ev[f"{fam}_dp"] = torch.stack([m / s_dp, sd / s_dp, (n > 0).double(), torch.log1p(n)], -1)
            fu, _, t = stats(up, tested, W)
            fd, _, _ = stats(down, tested, W)
            ev[f"{fam}_de"] = torch.stack([fu, fd, (t > 0).double(), torch.log1p(t)], -1)
            m, sd, n = stats(lfc, tested, W)
            ev[f"{fam}_lfc"] = torch.stack([m / s_lfc, sd / s_lfc, (n > 0).double(), torch.log1p(n)], -1)
        self.evidence = {k: v.float() for k, v in ev.items()}                    # [P, G, 4]
        self.ctrl_mean = self.X[self.ctrl].mean(0)                               # [G]
        self.n_string_donors = W_str.sum(1)                                      # [P]
        self.str_idx, self.str_mask = idx, ok                                    # [P, K]：邻居（MSA 的相关行）
        # N1 用：归一化后的逐扰动标签、两两效果相关 Cmat（只在训练 batch 中使用训练扰动之间的值）
        self.s_dp, self.s_lfc = s_dp, s_lfc
        self.dp_n = (dp / s_dp).float()
        self.lfc_n = (lfc / s_lfc).float()
        self.up_f, self.down_f = up.float(), down.float()
        keepc = torch.ones(dp.shape[1], dtype=torch.bool, device=self.dev)
        keepc[L["target"][L["target"] >= 0]] = False
        Xc = dp_delta[:, keepc].double()                                         # c_k 按 delta_p 的相关（--space bulk 时为打分空间 δ）
        Xc = Xc - Xc.mean(1, keepdim=True)
        Xc = Xc / Xc.norm(dim=1, keepdim=True).clamp_min(1e-12)
        self.Cmat = (Xc @ Xc.T).float()

    def ev_batch(self, perts):
        e = {k: v[perts] for k, v in self.evidence.items()}
        e["ctrl"] = self.ctrl_mean.expand(perts.numel(), -1)
        return e

    def dp_grid(self, train):
        """PIE 的 delta_p 分箱：dp_max = 训练扰动 |delta_p| 的 99.9 分位，宽度 ln(1.1)，箱数取奇数"""
        dp_max = float(torch.quantile(self.labels["delta_p"][train].abs().flatten().float(), 0.999))
        k = math.ceil(2 * dp_max / math.log(DP_BIN_FC))
        return dict(dp_bins=k + 1 if k % 2 == 0 else k, dp_max=dp_max)


def lr_at(step, total, base):
    if step < WARMUP:
        return base * (step + 1) / WARMUP
    return base * 0.5 * (1 + math.cos(math.pi * (step - WARMUP) / max(1, total - WARMUP)))


def pearson_rows(d, t):
    d, t = d - d.mean(1, keepdim=True), t - t.mean(1, keepdim=True)
    return (d * t).sum(1) / (d.norm(dim=1) * t.norm(dim=1) + 1e-8)


def average_precision(score, label):
    """二分类 AUPRC（与 sklearn average_precision_score 相同：同分的样本作为一个阈值整体计入）"""
    order = score.argsort(descending=True)
    s, y = score[order], label[order].float()
    tp = y.cumsum(0)
    last = torch.ones_like(s, dtype=torch.bool)
    last[:-1] = s[1:] != s[:-1]                                                  # 每个同分块的最后一个位置
    idx = torch.nonzero(last, as_tuple=True)[0]
    tp_b = tp[idx]
    prec = tp_b / (idx + 1).float()
    gain = torch.diff(tp_b, prepend=tp_b.new_zeros(1))                           # 每个同分块新增的正例数
    return ((prec * gain).sum() / y.sum().clamp(min=1)).item()


@torch.no_grad()
def diag_eval(model, data, perts, n, draws=4):
    """评估扰动上的两个指标，各对 draws 次随机抽样平均：
    pearson_delta  delta 读出（argmax 箱中心，PIE）与真实 delta_p（全部细胞）的相关，按扰动平均
    pearson_exp    同上，但读出取各箱中心按 softmax 概率的期望（连续值，仅作对照）
    auprc          DE 头的 AUPRC（在已检验基因上汇总，PIE 的选模型指标）"""
    model.eval()
    r, re, ap, rc, rn, zc, zp = [], [], [], [], [], [], []
    bm, bp, bn = [], [], []
    centers = dp_centers(model.c, data.dev)
    lab = {k: v[perts] for k, v in data.labels.items()}
    for _ in range(draws):
        inp, _, _ = data.batch(perts, n)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP):
            o = model(**inp)
        r.append(pearson_rows(o.delta.float(), lab["delta_p"]).mean().item())
        d_exp = o.delta.float() if o.dp_logits is None else o.dp_logits.float().softmax(-1) @ centers   # v7 affine：连续 δ̂
        if o.dp_base is not None and o.dp_logits is not None:                    # v8 nb_resid：δ̂ = nbw + E[残差]
            d_exp = d_exp + o.dp_base.float()
        re.append(pearson_rows(d_exp, lab["delta_p"]).mean().item())
        p_de = o.de_logits.float().softmax(-1)[..., 1]
        ap.append(average_precision(p_de[lab["tested"]], lab["de_mask"][lab["tested"]]))
        if o.dp_cont is not None:
            rc.append(pearson_rows(o.dp_cont.float(), lab["delta_p"]).mean().item())
        if o.nbw_dp is not None:
            rn.append(pearson_rows(o.nbw_dp.float(), lab["delta_p"]).mean().item())
        if o.cell_patch0 is not None:
            zc.append(o.cell_patch0.float().flatten(1))
        zp.append(o.pert_emb_x.float().mean(1))
        if data.space == "bulk":                                                 # v10：打分空间 MSE / pearson（gmask 格子）
            m = data.expr[None, :] & (torch.arange(data.X.shape[1], device=data.dev)[None, :] != lab["target"][:, None])
            t, s_ = lab["dB"], float(model.s_dp)
            dh = o.dp_cont.float() * s_ if o.dp_cont is not None else o.delta.float()
            bm.append((((dh - t) / s_) ** 2)[m].mean().item())
            bp.append(masked_pearson(dh, t, m).mean().item())
            if o.nbw_dp is not None:
                bn.append(masked_pearson(o.nbw_dp.float() * s_, t, m)[o.nb_mask.any(1)].mean().item())
    model.train()
    mean = lambda v: sum(v) / len(v) if v else float("nan")  # noqa: E731
    rankme.last = (rank_me(torch.cat(zc)) if zc else float("nan"), rank_me(torch.cat(zp)))
    bulkdiag.last = {"val_mse": mean(bm), "val_pearson_bulk": mean(bp), "val_pearson_nbw_bulk": mean(bn)} if bm else {}
    return mean(r), mean(re), mean(ap), mean(rc), mean(rn)


def masked_pearson(a, b, m):
    """逐行 pearson，只用 m 为真的列。a、b [B, G]，m [B, G] bool → [B]"""
    w = m.float()
    n = w.sum(1, keepdim=True).clamp_min(1)
    a = (a - (a * w).sum(1, keepdim=True) / n) * w
    b = (b - (b * w).sum(1, keepdim=True) / n) * w
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-8)


class bulkdiag:                                                                  # v10：diag_eval 最近一次的打分空间指标
    last = {}


def rank_me(z):
    """RankMe（Garrido et al. 2023）：exp(奇异值分布的熵)，z [N, D]"""
    s = torch.linalg.svdvals(z.double())
    p = s / s.sum().clamp_min(1e-12) + 1e-12
    return float(torch.exp(-(p * p.log()).sum()))


class rankme:                                                                    # diag_eval 最近一次的 RankMe（CellPatch，扰动一路）
    last = (float("nan"), float("nan"))


def train_stage(model, data, train, ev, n, steps, lr, log, tag, save=None, eval_every=EVAL_EVERY, best=None, best_cont=None,
                select="pearson_exp"):
    """save：{步数: 路径}，在这些步数（训练完该步后）保存权重。best：路径，评估扰动上 pearson_exp 最高时保存；
    best_cont：同上，按 pearson_cont（N2 连续头）"""
    best_val = best_c = None
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=WD)
    centers = dp_centers(model.c, data.dev) if model.c.dp_bins > 0 else None
    model.train()
    t0 = time.time()
    for step in range(1, steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step - 1, steps, lr)
        perts = train[torch.randint(0, train.numel(), (BATCH,), device=data.dev)]
        inp, y, lab = data.batch(perts, n, train=True)
        if MASK_TARGETS and model.c.objective == "jepa":                        # v6：靶基因列 → 对照均值
            y = y.clone()
            y[:, data.target_cols] = data.ctrl_cell_mean[data.target_cols]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP):
            if model.c.objective == "mae":                                       # v6：MAE，不经过整个模型
                loss = model.mae_loss(data.gene_idx, torch.cat([inp["x"], y]))
            else:
                o = model(**inp)
                loss = model.loss(o, data.gene_idx, y=y, labels=lab)
            if SAMPLE_LOSS > 0 and model.c.objective == "recon":                # T1：逐样本 δ̂ 对 y − x
                d_exp = o.dp_logits.float().softmax(-1) @ centers
                if o.dp_base is not None:                                        # v8 nb_resid
                    d_exp = d_exp + o.dp_base.float()
                loss = loss + SAMPLE_LOSS * F.mse_loss(d_exp, (y - inp["x"]).float())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        model.update_target()
        if save and step in save:
            torch.save(model.state_dict(), save[step])
        if step % eval_every == 0 or step == steps:
            r, re, ap, rc, rn = diag_eval(model, data, ev, n)
            rec = {"stage": tag, "lr": lr, "step": step, "loss": loss.item(),
                   "eval_pearson_delta": r, "eval_pearson_exp": re, "eval_auprc": ap,
                   "eval_pearson_cont": rc, "eval_pearson_nbw": rn,
                   "eval_rankme_cell": rankme.last[0], "eval_rankme_pert": rankme.last[1], "sec": round(time.time() - t0)}
            rec.update(bulkdiag.last)
            crit = -bulkdiag.last["val_mse"] if select == "mse" else re           # v10：--select mse 时按 val MSE（越低越好）
            if best and (best_val is None or crit > best_val):
                best_val = crit
                torch.save(model.state_dict(), best)
                rec["best_" + ("val_mse" if select == "mse" else "pearson_exp")] = True
            if best_cont and rc == rc and (best_c is None or rc > best_c):
                best_c = rc
                torch.save(model.state_dict(), best_cont)
                rec["best_pearson_cont"] = True
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(rec, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int)
    ap.add_argument("fold", type=int, nargs="?", default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--s1", type=int, default=S1, help="阶段 1（jepa）步数；0 = 不做 jepa，直接 recon")
    ap.add_argument("--lr1", type=float, default=LR1)
    ap.add_argument("--s2", type=int, default=S2)
    ap.add_argument("--stage2", default=f"freeze:{LR2_FREEZE},ft:{LR2_FT}",
                    help="阶段 2 的版本列表，逗号分隔的 freeze|ft:学习率")
    ap.add_argument("--save-steps", default="", help="阶段 2 额外保存权重的步数，逗号分隔（默认只存最后一步）")
    ap.add_argument("--inner-val", type=int, default=0,
                    help="> 0：从本折训练扰动中留出这么多个做验证（调参用），评估在验证集上；本折测试扰动不使用")
    ap.add_argument("--no-evidence", action="store_true", help="不使用 PIE 的 context 证据")
    ap.add_argument("--split", default="", help="划分文件：train 训练，val 评估 / 选 best 权重，test 不使用；给出时忽略 fold 与 --inner-val")
    ap.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    ap.add_argument("--seed", type=int, default=0, help="随机种子偏移（0 = 原来的种子）")
    ap.add_argument("--relation", default="string", help="邻居来源：string（原 v3）或 data/relations/h1_<名字>.npz")
    ap.add_argument("--value-norm", action="store_true", help="A1：值先按基因相对对照标准化")
    ap.add_argument("--value-enc", default="autobin", choices=["autobin", "linear", "mul"], help="A2：值编码；v6 mul")
    ap.add_argument("--patch-groups", action="store_true", help="A3：每个 latent 只看一组基因")
    ap.add_argument("--freeze-genes", action="store_true", help="A5：gene_table 不训练")
    ap.add_argument("--token-ln", action="store_true", help="A6：基因 / 值 embedding 各自 LN")
    ap.add_argument("--var-weight", type=float, default=0.0, help="T2：jepa 的 VICReg 方差项权重")
    ap.add_argument("--relate-grad", action="store_true", help="T5：相关行也回传梯度")
    ap.add_argument("--fp32", action="store_true", help="T3：不用 bf16 autocast")
    ap.add_argument("--wd", type=float, default=0.01, help="T4：AdamW weight decay")
    ap.add_argument("--sample-loss", type=float, default=0.0, help="T1：逐样本 loss 权重（同时按测序深度配对 x 与 y）")
    ap.add_argument("--recon-weight", type=float, default=0.0, help="E1：encoder 重建对照的权重")
    ap.add_argument("--patch-var", type=float, default=0.0, help="E2：CellPatch 输出跨样本方差项的权重")
    ap.add_argument("--patch-var-gamma", type=float, default=1.0, help="E2：目标标准差")
    ap.add_argument("--nb-learn", action="store_true", help="N1：邻居学习权重")
    ap.add_argument("--nb-tau", type=float, default=0.1)
    ap.add_argument("--nb-weight", type=float, default=1.0)
    ap.add_argument("--cont-head", action="store_true", help="N2：连续 δ 头")
    ap.add_argument("--cont-weight", type=float, default=1.0)
    ap.add_argument("--nb-direct", action="store_true", help="N3：连续头 + α · 加权邻居 δ")
    ap.add_argument("--pert-var", type=float, default=0.0, help="P1：扰动一路跨样本 VICReg 权重")
    ap.add_argument("--slot-attn", action="store_true", help="S：CellPatch 用 Slot Attention 式 latent 竞争")
    ap.add_argument("--slot-iters", type=int, default=3)
    ap.add_argument("--sigreg", type=float, default=0.0, help="J：SIGReg 权重")
    ap.add_argument("--disc", type=float, default=0.0, help="D：区分度 InfoNCE 权重")
    ap.add_argument("--disc-tau", type=float, default=0.1)
    ap.add_argument("--aug-sigma", type=float, default=0.0, help="B1：批次增强，log 计数缩放的标准差")
    ap.add_argument("--ev-mode", default="delta", choices=["delta", "fc"], help="B2：证据用 delta 或 fold change")
    ap.add_argument("--norm-bulk", action="store_true", help="v6：A1 / MAE 的标准化统计量用 bulk")
    ap.add_argument("--jepa-mode", default="ema", choices=["ema", "sg"], help="v6：阶段 1 jepa 的目标")
    ap.add_argument("--idm", type=float, default=0.0, help="v6：EP-IDM 权重")
    ap.add_argument("--jepa-sigreg", type=float, default=0.0, help="v6：阶段 1 SIGReg 权重（池化在线 latent）")
    ap.add_argument("--mask-targets", action="store_true", help="v6：阶段 1 中 y 的靶基因列换成对照均值")
    ap.add_argument("--mae", action="store_true", help="v6：阶段 1 改为 MAE")
    ap.add_argument("--mae-ratio", type=float, default=0.3)
    ap.add_argument("--init-cellpatch", default="", help="v6：Tahoe 预训练的 CellPatch 权重")
    ap.add_argument("--no-encoder", action="store_true", help="消融：cell token = 可学习常量（需 --s1 0）")
    ap.add_argument("--no-cell-msa", action="store_true", help="消融：跳过 cell_MSA")
    ap.add_argument("--no-pert-msa", action="store_true", help="消融：跳过 pert_MSA")
    ap.add_argument("--ev-drop", default="", help="消融：decoder 证据去掉的类别，逗号分隔 ctx / str / nbw，或 all")
    ap.add_argument("--const-z", action="store_true", help="消融：decoder 读常量而不是 backbone 输出")
    ap.add_argument("--dec-type", default="xattn", choices=["xattn", "xattn_sa", "affine"], help="v7：decoder 类型")
    ap.add_argument("--ctrl-jepa", type=float, default=0.0, help="v7：对照 JEPA 的权重（阶段 1）")
    ap.add_argument("--ctrl-jepa-ratio", type=float, default=0.3)
    ap.add_argument("--input-mask", type=float, default=0.0, help="v7：训练时对照输入遮住的基因比例（阶段 2；未给 --input-mask-s1 时两个阶段都用它）")
    ap.add_argument("--nb-resid", action="store_true", help="v8：分箱头学残差 delta_p − nbw，读出 nbw + E[残差]（需 --nb-learn）")
    ap.add_argument("--affine-id", action="store_true", help="10-21 修正：affine decoder 初始化为不变（ŷ = x）")
    ap.add_argument("--ev-family-dropout", type=float, default=0.25, help="10-21 修正：证据整类丢弃的概率（只留 nbw 时用 0.0156 = 0.25³）")
    ap.add_argument("--input-mask-s1", type=float, default=-1.0, help="v7（用户 10-21）：阶段 1 的输入 mask 比例（如 0.3），阶段 2 用 --input-mask")
    ap.add_argument("--w-de", type=float, default=1.0, help="v9：pie_loss 各项权重")
    ap.add_argument("--w-lfc", type=float, default=1.0)
    ap.add_argument("--w-dir", type=float, default=1.0)
    ap.add_argument("--w-dp", type=float, default=1.0)
    ap.add_argument("--mse-dhat", type=float, default=0.0, help="v9：期望读出 δ̂ 对 delta_p 的 MSE 权重")
    ap.add_argument("--space", default="dp", choices=["dp", "bulk"], help="v10：标签 / 证据 / 输出的空间")
    ap.add_argument("--recon-loss", default="pie", choices=["pie", "mse", "zinb"], help="v10：阶段 2 的 loss")
    ap.add_argument("--select", default="pearson_exp", choices=["pearson_exp", "mse"], help="v10：阶段 2 选 best 的指标")
    ap.add_argument("--from-stage1", default="", help="v9：跳过阶段 1，载入已有 stage1.pt（路径中的 {fold} 换成折号）；需 --s1 0")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=str(RUNS))
    args = ap.parse_args()
    global AMP, WD, SAMPLE_LOSS, MASK_TARGETS
    AMP, WD, SAMPLE_LOSS, MASK_TARGETS = not args.fp32, args.wd, args.sample_loss, args.mask_targets
    if args.mae and (args.idm > 0 or args.jepa_sigreg > 0 or args.jepa_mode != "ema" or args.mask_targets):
        raise SystemExit("--mae 与 jepa 的选项互斥")
    if args.norm_bulk and not args.value_norm:
        raise SystemExit("--norm-bulk 需要 --value-norm")
    dev = torch.device(f"cuda:{args.gpu}")
    torch.manual_seed(1000 * args.n + args.fold + 100_000 * args.seed)
    data = H1Data(dev)
    folds = data.meta["folds"]
    train = sorted(set(range(len(data.meta["perts"]))) - set(folds[args.fold]))
    if args.split:
        sp = json.loads(Path(args.split).read_text())
        train, ev, ev_name = sp["train"], sp["val"], "val"
    elif args.inner_val > 0:
        g = np.random.default_rng(args.fold)
        val = sorted(g.choice(train, args.inner_val, replace=False).tolist())
        train, ev, ev_name = sorted(set(train) - set(val)), val, "val"
    else:
        ev, ev_name = folds[args.fold], "test"
    train, ev = torch.tensor(train, device=dev), torch.tensor(ev, device=dev)
    data.ev_mode = args.ev_mode
    if args.aug_sigma > 0 or args.ev_mode == "fc":
        data.build_aug_tables()
    data.aug_sigma = args.aug_sigma
    if args.space == "bulk":                                                    # v10：打分空间标签（先于 set_evidence，证据也用它）
        data.use_bulk()
    elif args.recon_loss != "pie":
        raise SystemExit("--recon-loss mse / zinb 需要 --space bulk")
    data.need_counts = args.recon_loss == "zinb"
    data.set_evidence(train, args.relation)
    n_nb = data.str_mask[ev].sum(1).float()
    print(f"关联来源 {args.relation}：评估扰动有邻居 {(n_nb > 0).sum().item()}/{ev.numel()}，"
          f"训练扰动有邻居 {(data.str_mask[train].sum(1) > 0).sum().item()}/{train.numel()}", flush=True)
    data.use_evidence = not args.no_evidence
    data.depth_match = args.sample_loss > 0
    data.use_nb = args.nb_learn
    dp = data.labels["delta_p"]
    base = pearson_rows(dp[train].mean(0, keepdim=True).expand(ev.numel(), -1), dp[ev]).mean().item()
    # DE 参照：每个基因在训练扰动中为 DE 的比例作为分数
    L = data.labels
    frac = (L["de_mask"][train].float().sum(0) / L["tested"][train].float().sum(0).clamp(min=1)).expand(ev.numel(), -1)
    base_ap = average_precision(frac[L["tested"][ev]], L["de_mask"][ev][L["tested"][ev]])
    prev = L["de_mask"][ev][L["tested"][ev]].float().mean().item()
    grid = data.dp_grid(train)
    fixes = dict(value_norm=args.value_norm, value_enc=args.value_enc, patch_groups=args.patch_groups,
                 freeze_genes=args.freeze_genes, token_ln=args.token_ln, var_weight=args.var_weight, relate_grad=args.relate_grad,
                 recon_weight=args.recon_weight, patch_var_weight=args.patch_var, patch_var_gamma=args.patch_var_gamma,
                 nb_learn=args.nb_learn, nb_feat_dim=int(data.nb_feat.shape[-1]) if args.nb_learn else 0, nb_tau=args.nb_tau,
                 nb_weight=args.nb_weight, cont_head=args.cont_head or args.recon_loss != "pie", cont_weight=args.cont_weight, nb_direct=args.nb_direct,
                 pert_var_weight=args.pert_var, slot_attn=args.slot_attn, slot_iters=args.slot_iters,
                 sigreg_weight=args.sigreg, disc_weight=args.disc, disc_tau=args.disc_tau,
                 jepa_mode=args.jepa_mode, idm_weight=args.idm, n_actions=len(data.meta["perts"]) if args.idm > 0 else 0,
                 jepa_sigreg=args.jepa_sigreg, mae=args.mae, mae_ratio=args.mae_ratio,
                 no_encoder=args.no_encoder, no_cell_msa=args.no_cell_msa, no_pert_msa=args.no_pert_msa,
                 ev_drop=args.ev_drop, const_z=args.const_z,
                 dec_type=args.dec_type, ctrl_jepa=args.ctrl_jepa, ctrl_jepa_ratio=args.ctrl_jepa_ratio, input_mask=args.input_mask,
                 affine_id=args.affine_id, nb_resid=args.nb_resid, ev_family_dropout=args.ev_family_dropout,
                 w_de=args.w_de, w_lfc=args.w_lfc, w_dir=args.w_dir, w_dp=args.w_dp, mse_weight=args.mse_dhat,
                 recon_loss=args.recon_loss)
    if args.no_encoder and args.s1 > 0:
        raise SystemExit("--no-encoder 需要 --s1 0（没有可预训练的 encoder）")
    if args.nb_direct and not (args.nb_learn and args.cont_head):
        raise SystemExit("--nb-direct 需要同时 --nb-learn --cont-head")
    cdef = {f.name: f.default for f in dataclasses.fields(Config3)}
    if any(v != cdef[k] for k, v in fixes.items()):
        grid.update(fixes, n_genes=data.gene_idx.numel())                      # 只有开了修复时才写，原 v3 的 grid.json 不变
    ctrl_x = data.X[data.ctrl]
    ctrl_mu, ctrl_sd = ctrl_x.mean(0), ctrl_x.std(0).clamp_min(CTRL_SD_MIN)
    data.ctrl_cell_mean = ctrl_mu                                               # v6 --mask-targets
    tg = data.labels["target"]
    data.target_cols = tg[tg >= 0].unique()
    if args.norm_bulk or args.mae:                                              # v6：bulk 的每基因均值 / 标准差（只在开时抽样）
        with torch.no_grad():
            pb = train[torch.randint(0, train.numel(), (512,), device=dev)]
            bb = torch.cat([data.bulk(data.ctrl, 512, args.n), torch.stack([data.bulk(data.cells[int(p)], 1, args.n)[0] for p in pb])])
            bulk_mu, bulk_sd = bb.mean(0), bb.std(0).clamp_min(NORM_BULK_SD_MIN)
        if args.norm_bulk:
            ctrl_mu, ctrl_sd = bulk_mu, bulk_sd
    theta0 = data.nb_theta() if args.recon_loss == "zinb" else None            # v10 zinb：θ 初值（对照细胞矩估计）
    rec_mu = rec_sd = None
    if args.recon_weight > 0:                                                   # E1：重建目标按对照 bulk（n 个细胞）的均值 / 标准差标准化
        with torch.no_grad():                                                   # （只在开 E1 时抽样，不改变其他设置的随机数序列）
            cb = data.bulk(data.ctrl, 512, args.n)
            rec_mu, rec_sd = cb.mean(0), cb.std(0).clamp_min(1e-3)

    def init_norms(model):
        if args.value_norm:
            model.set_value_norm(ctrl_mu, ctrl_sd)
        if args.recon_weight > 0:
            model.set_rec_norm(rec_mu, rec_sd)
        if args.nb_learn or args.cont_head or args.recon_loss != "pie":
            model.set_scale(data.s_dp)
        if theta0 is not None:
            model.set_theta(theta0)
        if args.mae:
            model.set_mae_norm(bulk_mu, bulk_sd)
        if args.init_cellpatch:                                                 # v6：Tahoe 预训练的 CellPatch（不含标准化缓冲区）
            sd = torch.load(args.init_cellpatch, map_location=dev)
            sd = {k: v for k, v in sd.items() if k not in ("ctrl_mu", "ctrl_sd")}
            missing, unexpected = model.cell_patch.load_state_dict(sd, strict=False)
            assert not unexpected and set(missing) <= {"ctrl_mu", "ctrl_sd"}, (missing, unexpected)
            print(f"载入 CellPatch：{args.init_cellpatch}（{len(sd)} 个张量）", flush=True)
    out = Path(args.out) / f"n{args.n}" / (Path(args.split).stem if args.split else f"fold{args.fold}") / args.tag
    out.mkdir(parents=True, exist_ok=True)
    print(f"n={args.n} fold={args.fold} {args.tag}：训练扰动 {train.numel()}，评估（{ev_name}）{ev.numel()}，"
          f"G={data.gene_idx.numel()}；参照（训练扰动平均 delta）pearson_delta={base:.3f}，"
          f"（训练扰动 DE 比例）auprc={base_ap:.3f}，DE 占比 {prev:.3f}；delta_p 分箱 {grid}", flush=True)
    (out / "grid.json").write_text(json.dumps(grid))
    (out / "data.json").write_text(json.dumps({"model": "v3", "relation": args.relation, "use_evidence": data.use_evidence,
                                              "fp32": args.fp32, "sample_loss": args.sample_loss, "wd": args.wd,
                                              "aug_sigma": args.aug_sigma, "ev_mode": args.ev_mode,
                                              "norm_bulk": args.norm_bulk, "mask_targets": args.mask_targets,
                                              "init_cellpatch": args.init_cellpatch, "space": args.space,
                                              "recon_loss": args.recon_loss}))
    save_steps = [int(x) for x in args.save_steps.split(",") if x]

    with open(out / "log.jsonl", "w") as log:
        log.write(json.dumps({"eval_set": ev_name, "n_train": train.numel(), "n_eval": ev.numel(),
                              "baseline_train_mean_delta": base, "baseline_de_frac_auprc": base_ap,
                              "de_prevalence": prev, "grid": grid, "args": vars(args)}) + "\n")
        if args.s1 > 0:
            obj1 = "mae" if args.mae else "jepa"
            g1 = dict(grid, input_mask=args.input_mask_s1) if args.input_mask_s1 >= 0 else grid   # v7：阶段 1 单独的输入 mask
            model = PertFold3(Config3(objective=obj1, **g1), data.esm).to(dev)
            init_norms(model)
            train_stage(model, data, train, ev, args.n, args.s1, args.lr1, log, f"stage1_{obj1}", eval_every=args.eval_every)
            torch.save(model.state_dict(), out / "stage1.pt")
        s1_path = out / "stage1.pt"
        if args.from_stage1:                                                    # v9：复用已有的阶段 1 权重
            if args.s1 > 0:
                raise SystemExit("--from-stage1 需要 --s1 0")
            obj1 = "jepa"
            s1_path = Path(args.from_stage1.format(fold=args.fold))
            model = PertFold3(Config3(objective=obj1, **grid), data.esm).to(dev)
            init_norms(model)
            log.write(json.dumps({"from_stage1": str(s1_path)}) + "\n")
        for spec in args.stage2.split(","):
            variant, lr = spec.split(":")[0], float(spec.split(":")[1])
            name = f"{variant}_lr{lr:g}"
            if args.s1 > 0 or args.from_stage1:
                missing, unexpected = model.load_state_dict(torch.load(s1_path, map_location=dev), strict=False)
                new = ("decoder.cont.", "decoder.out_adapters.cont.", "decoder.zi.", "log_theta")   # v10：阶段 1 不训练 decoder
                assert not unexpected and all(k.startswith(new) for k in missing), (missing, unexpected)
                if args.nb_learn or args.cont_head or args.recon_loss != "pie":   # v10：s_dp 按本次的空间重设（dp 空间时与阶段 1 相同）
                    model.set_scale(data.s_dp)
                model.set_objective(obj1)                                   # 恢复全部可训练，再切换
                model.c = dataclasses.replace(model.c, input_mask=args.input_mask)   # v7：阶段 2 的输入 mask
                model.set_objective("recon", freeze_backbone=(variant == "freeze"))
            else:
                if variant == "freeze":
                    continue                                                # 没有预训练，冻结无意义
                model = PertFold3(Config3(objective="recon", **grid), data.esm).to(dev)
                init_norms(model)
            save = {k: out / f"stage2_{name}_step{k}.pt" for k in save_steps}
            save[args.s2] = out / f"stage2_{name}.pt"
            sel = "val_mse" if args.select == "mse" else "pearson_exp"
            train_stage(model, data, train, ev, args.n, args.s2, lr, log, f"stage2_{name}", save,
                        eval_every=args.eval_every, best=out / f"stage2_{name}_best_{sel}.pt",
                        best_cont=out / f"stage2_{name}_best_pearson_cont.pt" if args.cont_head and args.recon_loss == "pie" else None,
                        select=args.select)


if __name__ == "__main__":
    main()

