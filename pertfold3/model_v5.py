"""v11（2026-10-22 晚，用户要求：证据直通 decoder，与 r4 对比；不改原有路径）：
  PertFold3.encode_z  只算 backbone（①②③），返回 decoder 读的 z（同 forward 的 cell_patch_y_hat；不支持 no_encoder / input_mask / const_z）
  PassDecoder         δ̂_g = (1 + a_g)·base_g + s·c_g；[a_g, c_g] = MLP([h_g, f_g])，最后一层零初始化（初始 δ̂ = base，即证据本身）；
                      h_g = LN(CrossAttn(LN(Linear(esm_g)), z))，esm 为固定的标准化 ESM（输出基因可以多于 encoder 的输入基因）
v10 开关（2026-10-22 晚，用户要求：标签 / 证据 / 输出统一到打分空间；loss 先用 MSE + nb loss，再试 ZINB；默认 = v9）：
  recon_loss   "pie"（原四项）| "mse" | "zinb"。后两者需要 cont_head：连续头输出打分空间的 δ̂ / s_dp
               （打分空间 = log1p(5e4 · 伪 bulk 表达比例)，δ = 扰动 − 对照；训练脚本 --space bulk 提供标签 labels["dB"]）。
               mse   MSE(δ̂ / s_dp, δ / s_dp)，只算 labels["gmask"]（对照有表达、且不是本扰动靶基因）的格子
               zinb  该扰动 n 个细胞计数（labels["counts"] [B, n, G]，文库 labels["lib"] [B, n]）的 ZINB 负对数似然：
                     群体比例 q̂ = expm1(softplus₅₀(b_c + δ̂)) / 5e4（b_c = 全部对照的打分空间伪 bulk，labels["bc"]），
                     NB 均值 = 文库 · q̂ / (1 − π)（使 ZINB 均值 = 文库 · q̂）；θ 每基因一个参数（log_theta），π 由 decoder 的 zi 头给出
               两者都不再用 DE / LFC / 分箱三个头（仍然建，不进 loss）；delta 读出 = 连续头 × s_dp
v9 开关（2026-10-22，用户要求：按 loss 情况加权约束过大的 loss；试 PDS loss 与 MSE loss；默认 = v8）：
  w_de / w_lfc / w_dir / w_dp  pie_loss 四项的权重（默认 1）。诊断（diag_decoder_loss.py）：前 500 步 lfc_dir 对共享层的梯度是 dp 的 20–50 倍
  mse_weight   期望读出 δ̂（+ nb_resid 时的 nbw）对群体 delta_p 的 MSE（都除以 s_dp，去掉靶基因列），只在 recon 阶段
  PDS loss 用已有的 disc_weight（D，批内 InfoNCE）；nb_resid 时 δ̂ 加上 nbw
v8 开关（2026-10-22，用户要求测试"残差输出"；默认关）：
  nb_resid     delta_p 分箱头改学残差 r = delta_p − nbw（nbw = a_k 加权的邻居 δ，停梯度）；读出 δ̂ = nbw + E[r]（argmax 时 nbw + 箱中心）；
               分箱头最后一层零初始化 → 初始 δ̂ = nbw。没有邻居的行 nbw = 0，与原来相同。需要 nb_learn。
v7 开关（2026-10-21，用户的设计，见 pertfold/docs/plans/PLAN_v7_1021.md；默认全关 = v6）：
  dec_type     "xattn"（原：基因查询 cross-attention + 三个分箱头）| "xattn_sa"（另加 2 层基因 self-attention）|
               "affine"（CellPatch2_perturb_newdecoder：基因 token cross-attention → 2 层基因 self-attention →
               AffineExprDecoder ŷ = softplus(a)·x_ctrl + b；MSE(ŷ, x + delta_p) + DE 两类交叉熵；读出 δ̂ = ŷ − x）
  ev_drop 增加 "ctrl"：decoder 证据不含对照均值特征
  ctrl_jepa    对照 JEPA（只在阶段 1）：对照 x 遮住 ctrl_jepa_ratio 的基因作上下文 → ctrl_pred → 预测在线 CellPatch(x)（停梯度），MSE
  input_mask   训练时对输入的对照 x 随机遮住该比例的基因（两个阶段）
  affine_id    （10-21 修正）affine decoder 初始化为"不变"：系数分支最后一层权重 0、偏置 log(e−1)（softplus = 1），偏移分支最后一层全 0 → 初始 ŷ = x
v6 消融开关（2026-10-20，用户要求：以 R4 配方重训，去掉 encoder / MSA / decoder 各部分；默认全关）：
  no_encoder   对照 x 与邻居细胞行都不经 CellPatch：cell token = 可学习常量 [p, d]（cell_MSA 一并跳过）；不能做阶段 1
  no_cell_msa  跳过 cell_MSA（不编码邻居细胞行）
  no_pert_msa  跳过 pert_MSA（不编码邻居靶基因）
  ev_drop      decoder 证据按类去掉，逗号分隔：ctx / str / nbw；"all" = decoder 不接收任何证据（含对照特征）
  const_z      decoder 读可学习常量 [p, d] 而不是 backbone 输出 z（decoder 只靠基因查询与证据）
v6 开关（2026-10-20，用户要求：以 N1u + mul_std 为底座，比较 阶段 1 的预训练目标；默认全关 = v5）：
  value_enc="mul"    值调制基因：token = gene_emb ⊙ LN(AutoBin(x))（encoder 单独测试中的 mul；配 value_norm 即 mul_std）
  jepa_mode="sg"     arXiv 2610.07540 官方代码（JEPA_control，detach_targets=True、无 EMA）：目标 = 在线 CellPatch(y) 停梯度，MSE
  idm_weight         EP-IDM（同文）：D([池化 CellPatch(x), 池化 CellPatch(y)]) → 扰动类别，CE；两端都回传梯度；
                     D = 两层隐藏 256 的 ReLU MLP（同文 A.3.1）；只在阶段 1（objective="jepa"）
  jepa_sigreg        阶段 1 的 SIGReg：池化后的在线 latent（x 与 y 两端）→ sigreg()（同文 MSP+EP-IDM+SIG 的形式）
  mae                objective="mae"：每个 bulk 随机遮住 mae_ratio 的基因（CellPatch 不看这些 token），
                     MAEHead（基因查询交叉注意力 → LN → Linear）预测被遮基因的标准化表达（mae_mu / mae_sd），MSE 只算被遮基因
以下为 v5 说明。
PertFold3 v5（2026-10-18）：model_v3x.py 的副本，加了针对已诊断问题的开关（默认全关 = v3x）：
  N1 nb_learn        邻居学习权重（用户 SUGGESTIONS §7 的设计）：s_k = MLP([e_q, e_k, e_q⊙e_k, 关联特征])，
                     辅助 loss MSE(s_k, c_k)（c_k = 邻居与查询扰动真实 delta_p 的相关，仅训练扰动），a_k = softmax(s_k / τ)；
                     a_k 加权的邻居 δ / DE / LFC 统计作为新的基因级证据块 nbw_*（进 decoder）
  N2 cont_head       连续 δ 头：decoder h（+ 证据 adapter）→ Linear → δ / s_dp，MSE 对群体 delta_p；读出用它
  N3 nb_direct       连续头再加 α · (a_k 加权的邻居 δ)，α 为标量参数，初值 0
  P1 pert_var_weight 扰动一路（pert_MSA 输出对 T 个 token 取平均）的跨样本 VICReg：方差项 + pert_cov_weight · 协方差项
  （2026-10-19，塌缩修复，用户要求三项一起 + 逐个剔除）
  S  slot_attn       CellPatch 的交叉注意力改为 Slot Attention 式：softmax 在 latent 维上（latent 之间竞争基因），
                     再按基因归一化做加权平均；slot_iters 次迭代（每次 latent 作查询、残差 + FFN）
  J  sigreg_weight   LeJEPA 的 SIGReg：CellPatch(x) 展平后投影到 sigreg_slices 个随机方向，Epps–Pulley 统计量逼近 N(0, 1)；两个阶段都加
  D  disc_weight     区分度 loss：batch 内 InfoNCE，logits_ij = cos(δ̂_i, δ_j) / disc_tau（δ̂ 为期望值读出，δ 为群体标签，去掉靶基因列，
                     同一扰动的其他样本屏蔽）；只在 recon 阶段
以下为 v3x 说明。
PertFold3 v3x（2026-10-16）：model_v3.py 的副本，加了 encoder / 训练修复的开关（Config3 中 A* / T* 字段，默认全关 = 原 v3）。
  A1 value_norm      值先按基因相对对照标准化：x' = (x − 对照均值) / 对照标准差（缓冲区 ctrl_mu / ctrl_sd，存在 state_dict 中）
  A2 value_enc       "autobin"（原）| "linear"：连续线性值投影 Linear(1, d)
  A3 patch_groups    每个 latent 只看一组基因（基因序号 mod n_patches），不再对全部基因平均
  A5 freeze_genes    基因 embedding 表 gene_table 不训练（保持 ESM 标准化值）
  A6 token_ln        基因 embedding 与值 embedding 各自 LN 后再相加
  T2 var_weight      jepa 加 VICReg 方差项：λ · mean(relu(1 − std_batch(cell_patch_x)))
  T5 relate_grad     相关细胞 / 相关基因的编码也回传梯度
  E1 recon_weight    encoder 重建对照（2026-10-18）：基因查询对 CellPatch(x) 做交叉注意力，重建标准化后的对照 bulk x，λ · MSE
  E2 patch_var_weight  CellPatch 输出（MSA 之前）的跨样本防塌缩：λ · mean(relu(γ − std_batch))，每个 (token, 维) 一项
以下为原说明。
PertFold3 v3 冻结版（2026-10-15，decoder 含 PIE 证据与适配器；runs3/h1_v2、h1_v3 的权重用它加载）。PertFold3：预测单基因扰动（CRISPRi）后的单细胞表达。各节编号与 DATAFLOW_pertfold3.md 一致。

输入
    x [B, G]：pert 前的细胞（ctrl）；relate_x [B, Rc, G]：相关细胞
    pert_gene [B]：被扰动的基因；relate_genes [B, Rp]：相关基因
    gene_idx [G]：本批基因，取基因 embedding 表（ESM2 逐维标准化后初始化，可训练）

① 细胞模块
    cell_emb_x          = gene_emb(ESM2) + AutoBin(x)                   [B, G, d]
    cell_patch_x        = CellPatch(cell_emb_x)                         [B, p, d]
    cell_patch_relate_x = 相关细胞走同一个 CellPatch（不算梯度）         [B, Rc, p, d]
    cell_patch_x        ← cell_MSA([cell_patch_x ; cell_patch_relate_x]) 取第 0 行

② 扰动模块
    pert_emb_x          = Linear(ESM2(pert_gene)) 拆成 T 个 token        [B, T, d]
    pert_embs_relate    = 相关基因走同一个 Linear（不算梯度）            [B, Rp, T, d]
    pert_emb_x          ← pert_MSA([pert_emb_x ; pert_embs_relate]) 取第 0 行

③ 扰动预测
    cell_patch_x'       = CrossAttn(cell_patch_x ← pert_emb_x)          [B, p, d]
    cell_patch_y_hat    = Linear(LN(TransformerBlocks(cell_patch_x')))  [B, p, d]

④ Decoder（三个头与证据注入，同 PIE）
    证据（可选）：对照均值特征 + 本细胞系其他训练扰动、STRING 高置信邻居的 delta / DE / LFC 统计 → 融合 → 零初始化适配器
    de_logits, lfc, dp_logits = Decoder(cell_patch_y_hat, 证据)      [B, G, 2] / [B, G] / [B, G, K]
    delta               = dp_logits 的 argmax 箱中心值                   [B, G]
    y_hat               = x + delta

⑤ 训练目标（Config3.objective）
    recon   PIE 的四项 loss：DE 交叉熵 + LFC Huber + 方向 hinge + delta_p 两箱软标签交叉熵（群体标签）
    jepa    L1(cell_patch_y_hat, cell_patch_y)，cell_patch_y = LN(target_patch(y))，停梯度；
            target_patch 为 CellPatch 的 EMA 副本
    两阶段：先 jepa 预训练 ①②③，再 set_objective("recon", freeze_backbone=...) 训练 decoder
"""
from __future__ import annotations

import copy
import dataclasses
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class Config3:
    esm_dim: int = 1280
    d: int = 512
    n_patches: int = 128        # p
    n_pert_tokens: int = 8      # T：一个 pert 基因展开成 T 个 token，供 cell patch 交叉注意力选择
    # 表达值的软分箱（scFoundation AutoDiscretization）：n_bins 个可学习的箱 embedding
    n_bins: int = 100
    bin_alpha: float = 1.0
    n_heads: int = 8
    ffn_mult: int = 4
    n_cell_msa: int = 2         # cell_MSA block 数
    n_pert_msa: int = 2         # pert_MSA block 数
    n_blocks: int = 4           # 扰动预测的 TransformerBlocks 层数
    # 训练目标："recon" 用 PIE 的四项 loss（群体标签）；"jepa" 优化 L1(cell_patch_y_hat, cell_patch_y)
    objective: str = "recon"
    ema_momentum: float = 0.996 # 仅 "jepa"：目标编码器的 EMA 动量（I-JEPA 的初值）
    # delta_p 分箱（PIE）：K 个箱均分 [-dp_max, dp_max]，K 为奇数；dp_max 取训练扰动 |delta_p| 的 99.9 分位
    dp_bins: int = 0
    dp_max: float = 1.0
    # PIE loss 超参数（pie/configs/train.yaml）
    class_weight_cap: float = 10.0
    class_weight_cap_dp: float = 5.0
    lfc_huber_delta: float = 1.0
    lfc_target_alpha: float = 1e-3
    lfc_dir_tau: float = 0.25
    # 证据（PIE）：每个基因的对照特征 + 本细胞系其他训练扰动的统计，经零初始化适配器注入 decoder
    ev_encoder_dim: int = 64
    ev_dim: int = 128
    ev_dropout: float = 0.1
    ev_family_dropout: float = 0.25      # 训练时每行、每类证据（ctx / str）独立整类丢弃的概率
    # 证据块：ctx_* = 本细胞系其他训练扰动的统计；str_* = STRING 高置信邻居（训练扰动）的统计
    ev_blocks: tuple = ("ctx_dp", "ctx_de", "ctx_lfc", "str_dp", "str_de", "str_lfc")
    # v3x 开关（默认 = 原 v3）
    n_genes: int = 0                     # A1 需要：模型基因数（缓冲区大小）
    value_norm: bool = False             # A1
    value_enc: str = "autobin"           # A2
    patch_groups: bool = False           # A3
    freeze_genes: bool = False           # A5
    token_ln: bool = False               # A6
    var_weight: float = 0.0              # T2
    relate_grad: bool = False            # T5
    recon_weight: float = 0.0            # E1
    patch_var_weight: float = 0.0        # E2
    patch_var_gamma: float = 1.0         # E2 的目标标准差
    # v5 开关（默认 = v3x）
    nb_learn: bool = False               # N1
    nb_feat_dim: int = 0                 # N1：每个候选邻居的关联特征维数
    nb_tau: float = 0.1                  # N1：a_k = softmax(s_k / τ)
    nb_weight: float = 1.0               # N1：MSE(s_k, c_k) 的权重
    cont_head: bool = False              # N2
    cont_weight: float = 1.0             # N2：MSE(δ̂ / s_dp, delta_p / s_dp) 的权重
    nb_direct: bool = False              # N3
    pert_var_weight: float = 0.0         # P1
    pert_cov_weight: float = 0.04        # P1：协方差项相对方差项的系数（VICReg 1 : 25）
    slot_attn: bool = False              # S
    slot_iters: int = 3
    sigreg_weight: float = 0.0           # J
    sigreg_slices: int = 256
    disc_weight: float = 0.0             # D
    disc_tau: float = 0.1
    # v6 开关（默认 = v5）
    jepa_mode: str = "ema"               # "ema"：原 jepa；"sg"：在线编码器停梯度目标 + MSE（2610.07540）
    idm_weight: float = 0.0              # EP-IDM 的权重（> 0 时建 idm 头）
    n_actions: int = 0                   # EP-IDM 的类别数（扰动数）
    idm_hidden: int = 256
    jepa_sigreg: float = 0.0             # 阶段 1 SIGReg（池化在线 latent）
    mae: bool = False                    # 建 MAE 头（objective="mae" 用）
    mae_ratio: float = 0.3
    # v6 消融开关（默认 = 不消融）
    no_encoder: bool = False
    no_cell_msa: bool = False
    no_pert_msa: bool = False
    ev_drop: str = ""
    const_z: bool = False
    # v7 开关（默认 = v6）
    dec_type: str = "xattn"
    ctrl_jepa: float = 0.0
    ctrl_jepa_ratio: float = 0.3
    input_mask: float = 0.0
    affine_id: bool = False
    nb_resid: bool = False               # v8：分箱头学 delta_p − nbw
    # v9 开关（默认 = v8）
    w_de: float = 1.0
    w_lfc: float = 1.0
    w_dir: float = 1.0
    w_dp: float = 1.0
    mse_weight: float = 0.0
    # v10 开关（默认 = v9）
    recon_loss: str = "pie"              # "pie" | "mse" | "zinb"


class FFN(nn.Module):
    def __init__(self, d: int, mult: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d * mult), nn.GELU(), nn.Linear(d * mult, d))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class CrossAttn(nn.Module):
    """Pre-LN 交叉注意力 + FFN：x ← x + Attn(LN x, LN ctx)，x ← x + FFN(x)。

    x [n, Nq, d]，ctx [n, Nk, d]，pad [n, Nk] bool（True = 无效键）。
    """

    def __init__(self, d: int, n_heads: int, ffn_mult: int) -> None:
        super().__init__()
        self.ln_q = nn.LayerNorm(d)
        self.ln_kv = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.ffn = FFN(d, ffn_mult)

    def forward(self, x: Tensor, ctx: Tensor, pad: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None) -> Tensor:
        kv = self.ln_kv(ctx)
        x = x + self.attn(self.ln_q(x), kv, kv, key_padding_mask=pad, attn_mask=attn_mask, need_weights=False)[0]
        return x + self.ffn(x)


class SelfAttn(nn.Module):
    """Pre-LN 自注意力 + FFN。x [n, N, d]，pad [n, N] bool（True = 无效键）"""

    def __init__(self, d: int, n_heads: int, ffn_mult: int) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.ffn = FFN(d, ffn_mult)

    def forward(self, x: Tensor, pad: Optional[Tensor] = None) -> Tensor:
        h = self.ln(x)
        x = x + self.attn(h, h, h, key_padding_mask=pad, need_weights=False)[0]
        return x + self.ffn(x)


# ============================================================================
# ① 细胞模块
# ============================================================================

class AutoBinEmbedding(nn.Module):
    """scFoundation 的软分箱（AutoDiscretizationEmbedding）：x [...] → [..., d]

    h = LeakyReLU(Linear(1, K)(x))；w = softmax(α·h + Linear(K, K)(h))；输出 = w @ E，E 为 K 个可学习的箱 embedding。
    """

    def __init__(self, d: int, n_bins: int, alpha: float) -> None:
        super().__init__()
        self.alpha = alpha
        self.mlp = nn.Linear(1, n_bins)
        self.mlp2 = nn.Linear(n_bins, n_bins)
        self.act = nn.LeakyReLU(0.1)
        self.emb = nn.Embedding(n_bins, d)

    def forward(self, x: Tensor) -> Tensor:
        h = self.act(self.mlp(x.unsqueeze(-1)))
        w = torch.softmax(self.alpha * h + self.mlp2(h), dim=-1)
        return w @ self.emb.weight


class SlotCross(nn.Module):
    """S：Slot Attention 式交叉注意力。lat [N, p, d] 作查询，tok [N, G, d] 作键 / 值（多头）；
    A = softmax_latent(q·k / √dh)（latent 之间竞争），W = A / Σ_基因 A（对基因加权平均），lat ← lat + O(W v)，lat ← lat + FFN(lat)，迭代 iters 次"""

    def __init__(self, c: Config3) -> None:
        super().__init__()
        d = c.d
        self.h, self.iters = c.n_heads, c.slot_iters
        self.ln_q, self.ln_kv = nn.LayerNorm(d), nn.LayerNorm(d)
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d)
        self.ffn = FFN(d, c.ffn_mult)

    def forward(self, lat: Tensor, tok: Tensor) -> Tensor:
        N, G, d = tok.shape
        h, dh = self.h, d // self.h
        kv = self.ln_kv(tok)
        k = self.k(kv).view(N, G, h, dh).transpose(1, 2)                      # [N, h, G, dh]
        v = self.v(kv).view(N, G, h, dh).transpose(1, 2)
        for _ in range(self.iters):
            q = self.q(self.ln_q(lat)).view(N, -1, h, dh).transpose(1, 2)      # [N, h, p, dh]
            A = (q @ k.transpose(-1, -2) / dh ** 0.5).softmax(dim=-2) + 1e-8   # 在 latent 维上归一化：竞争
            A = A / A.sum(-1, keepdim=True)                                     # 对基因加权平均
            lat = lat + self.o((A @ v).transpose(1, 2).reshape(N, -1, d))
            lat = lat + self.ffn(lat)
        return lat


def sigreg(z: Tensor, n_slices: int) -> Tensor:
    """J：SIGReg（LeJEPA）。z [N, D] 投影到 n_slices 个随机单位方向，Epps–Pulley 统计量（经验特征函数与 N(0, 1) 的加权 L2 差）的平均"""
    A = torch.randn(z.shape[1], n_slices, device=z.device, dtype=z.dtype)
    proj = z @ (A / A.norm(dim=0, keepdim=True))                              # [N, M]
    t = torch.linspace(-3, 3, 17, device=z.device, dtype=z.dtype)
    xt = proj[..., None] * t
    phi = torch.exp(-0.5 * t ** 2)
    err = ((torch.cos(xt).mean(0) - phi) ** 2 + torch.sin(xt).mean(0) ** 2) * phi
    return torch.trapz(err, t, dim=-1).mean()


class CellPatch(nn.Module):
    """表达谱 → patch。x [N, G]，gene_esm [G, esm_dim] → [N, p, d]"""

    def __init__(self, c: Config3) -> None:
        super().__init__()
        self.c = c
        self.gene_emb = nn.Linear(c.esm_dim, c.d)
        self.value_emb = AutoBinEmbedding(c.d, c.n_bins, c.bin_alpha) if c.value_enc in ("autobin", "mul") else nn.Linear(1, c.d)
        if c.value_enc == "mul":                                                 # v6：token = 基因 ⊙ LN(AutoBin(x))
            self.value_ln = nn.LayerNorm(c.d)
        self.latents = nn.Parameter(torch.randn(c.n_patches, c.d) * 0.02)
        self.cross = CrossAttn(c.d, c.n_heads, c.ffn_mult)
        self.slot = SlotCross(c) if c.slot_attn else None                       # S
        self.ln_out = nn.LayerNorm(c.d)
        if c.value_norm:                                                         # A1：由 set_value_norm 写入，随 state_dict 保存
            self.register_buffer("ctrl_mu", torch.zeros(c.n_genes))
            self.register_buffer("ctrl_sd", torch.ones(c.n_genes))
        if c.token_ln:                                                           # A6
            self.ln_gene = nn.LayerNorm(c.d)
            self.ln_value = nn.LayerNorm(c.d)

    def forward(self, x: Tensor, gene_esm: Tensor, pad: Optional[Tensor] = None) -> Tensor:
        """pad [N, G] bool（True = 该基因 token 被遮住，latent 不看；MAE 用）"""
        c = self.c
        if c.value_norm:
            x = (x - self.ctrl_mu) / self.ctrl_sd
        v = self.value_emb(x) if c.value_enc in ("autobin", "mul") else self.value_emb(x.unsqueeze(-1))
        g = self.gene_emb(gene_esm)
        if c.token_ln:
            g, v = self.ln_gene(g), self.ln_value(v)
        cell_emb_x = g * self.value_ln(v) if c.value_enc == "mul" else g + v   # [N, G, d]
        lat = self.latents.expand(x.shape[0], -1, -1)                           # [N, p, d]
        if pad is not None:
            if self.slot is not None or c.patch_groups:
                raise ValueError("pad 不支持 slot_attn / patch_groups")
            return self.ln_out(self.cross(lat, cell_emb_x, pad))
        mask = None
        if c.patch_groups:                                                       # A3：latent i 只看 基因序号 mod p == i 的基因
            G = x.shape[-1]
            grp = torch.arange(G, device=x.device) % c.n_patches
            mask = grp[None, :] != torch.arange(c.n_patches, device=x.device)[:, None]   # [p, G]，True = 不可见
        if self.slot is not None:
            return self.ln_out(self.slot(lat, cell_emb_x))
        return self.ln_out(self.cross(lat, cell_emb_x, attn_mask=mask))


class MSABlock(nn.Module):
    """行注意力 → 列注意力 → FFN（均为 Pre-LN 残差）。M [B, S, N, d]，pad [B, S]（True = 无效行）→ M

    行注意力：一行内 N 个 token 之间。列注意力：同一 token 位置上 S 行之间，无效行不作为键。
    """

    def __init__(self, d: int, n_heads: int, ffn_mult: int) -> None:
        super().__init__()
        self.ln_row = nn.LayerNorm(d)
        self.row = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.ln_col = nn.LayerNorm(d)
        self.col = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.ffn = FFN(d, ffn_mult)

    def forward(self, M: Tensor, pad: Optional[Tensor]) -> Tensor:
        B, S, N, d = M.shape
        h = self.ln_row(M.reshape(B * S, N, d))
        M = M + self.row(h, h, h, need_weights=False)[0].reshape(B, S, N, d)
        Mt = M.transpose(1, 2).reshape(B * N, S, d)
        col_pad = None if pad is None else pad.repeat_interleave(N, dim=0)      # [B·N, S]
        h = self.ln_col(Mt)
        Mt = Mt + self.col(h, h, h, key_padding_mask=col_pad, need_weights=False)[0]
        M = Mt.reshape(B, N, S, d).transpose(1, 2)
        return M + self.ffn(M)


class MSA(nn.Module):
    """[query ; relate] → blocks → 第 0 行。query [B, N, d]，relate [B, R, N, d]，relate_mask [B, R]（True = 有效）"""

    def __init__(self, d: int, n_heads: int, ffn_mult: int, n_blocks: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([MSABlock(d, n_heads, ffn_mult) for _ in range(n_blocks)])

    def forward(self, query: Tensor, relate: Tensor, relate_mask: Optional[Tensor] = None) -> Tensor:
        M = torch.cat([query.unsqueeze(1), relate], dim=1)                      # [B, 1+R, N, d]
        pad = None
        if relate_mask is not None:
            pad = torch.cat([relate_mask.new_zeros(relate_mask.shape[0], 1), ~relate_mask], dim=1)   # 第 0 行恒有效
        for blk in self.blocks:
            M = blk(M, pad)
        return M[:, 0]


# ============================================================================
# ④ Decoder
# ============================================================================

N_CTRL_FEATURES = 3                     # [对照均值 z, log1p(100 z), z == 0]
BLOCK_WIDTH = 4                         # [值, 离散度, have, log1p(计数)]
HAVE_COL = 2


def ev_encoder(n_in: int, width: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(n_in, width), nn.SiLU(), nn.Linear(width, width), nn.RMSNorm(width))


def zero_last(seq: nn.Sequential) -> nn.Sequential:
    nn.init.zeros_(seq[-1].weight)
    return seq


class Evidence(nn.Module):
    """每个基因的证据 → 融合向量 [B, G, ev_dim]（PIE 的 _fuse_evidence）。

    ctrl  [B, G] 对照均值 → [z, log1p(100 z), z == 0] → 编码器
    其余每块 [B, G, 4]（c.ev_blocks）→ 编码器，乘以 have 列（无数据时为 0）
      ctx_*  本细胞系其他训练扰动的统计；str_*  STRING 高置信邻居的统计
    训练时每行对每一类（按前缀 ctx / str）独立以 ev_family_dropout 的概率整类丢弃。
    """

    def __init__(self, c: Config3) -> None:
        super().__init__()
        w, dim = c.ev_encoder_dim, c.ev_dim
        self.p_drop = c.ev_family_dropout
        self.use_ctrl = "ctrl" not in set(filter(None, c.ev_drop.split(",")))      # v7：可去掉对照特征
        self.ctrl = ev_encoder(N_CTRL_FEATURES, w)
        self.blocks = nn.ModuleDict({k: ev_encoder(BLOCK_WIDTH, w) for k in c.ev_blocks})
        n = w * (int(self.use_ctrl) + len(c.ev_blocks))
        self.fuse = nn.Sequential(nn.RMSNorm(n), nn.Linear(n, 2 * dim), nn.SiLU(), nn.Dropout(c.ev_dropout),
                                  nn.Linear(2 * dim, dim), nn.RMSNorm(dim))

    def forward(self, ev: dict) -> Tensor:
        z = ev["ctrl"].float()
        parts = [self.ctrl(torch.stack([z, torch.log1p(100.0 * z), (z == 0).float()], dim=-1))] if self.use_ctrl else []
        keep = {}
        if self.training and self.p_drop > 0:
            for fam in {k.split("_")[0] for k in self.blocks}:
                keep[fam] = (torch.rand(z.shape[0], device=z.device) >= self.p_drop).float()[:, None, None]
        for k, enc in self.blocks.items():
            x = ev[k].float()
            if keep:
                x = x * keep[k.split("_")[0]]
            parts.append(enc(x) * x[..., HAVE_COL:HAVE_COL + 1])
        return self.fuse(torch.cat(parts, dim=-1))


class NbScorer(nn.Module):
    """N1：s_k = MLP([e_q, e_k, e_q ⊙ e_k, 关联特征])，e = Linear(基因 embedding 表中靶基因的行)"""

    def __init__(self, c: Config3) -> None:
        super().__init__()
        w = 128
        self.proj = nn.Linear(c.esm_dim, w)
        self.mlp = nn.Sequential(nn.Linear(3 * w + c.nb_feat_dim, w), nn.SiLU(), nn.Linear(w, 1))

    def forward(self, eq: Tensor, ek: Tensor, feat: Tensor) -> Tensor:
        q, k = self.proj(eq)[:, None].expand(-1, ek.shape[1], -1), self.proj(ek)
        return self.mlp(torch.cat([q, k, q * k, feat], dim=-1)).squeeze(-1)


class ReconHead(nn.Module):
    """E1：CellPatch(x) [B, p, d] → 每个基因的标准化表达 [B, G]（基因查询 → 交叉注意力 → LN → Linear）"""

    def __init__(self, c: Config3) -> None:
        super().__init__()
        self.gene_query = nn.Linear(c.esm_dim, c.d)
        self.cross = CrossAttn(c.d, c.n_heads, c.ffn_mult)
        self.ln = nn.LayerNorm(c.d)
        self.out = nn.Linear(c.d, 1)

    def forward(self, patches: Tensor, gene_esm: Tensor) -> Tensor:
        q = self.gene_query(gene_esm).expand(patches.shape[0], -1, -1)
        return self.out(self.ln(self.cross(q, patches))).squeeze(-1)


class GeneSA(nn.Module):
    """v7：基因间 self-attention（Pre-LN，SDPA 省显存）+ FFN。x [B, G, d]"""

    def __init__(self, d: int, n_heads: int, ffn_mult: int) -> None:
        super().__init__()
        self.h = n_heads
        self.ln = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.ffn = FFN(d, ffn_mult)

    def forward(self, x: Tensor) -> Tensor:
        B, G, d = x.shape
        q, k, v = self.qkv(self.ln(x)).view(B, G, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, G, d)
        x = x + self.o(a)
        return x + self.ffn(x)


class Decoder(nn.Module):
    """cell_patch_y_hat [B, p, d] → 每个基因三个头（PIE）：DE logits [B, G, 2]、LFC [B, G]、delta_p 分箱 logits [B, G, K]

    每个基因一个查询（ESM2 → Linear）；有证据时查询加上 query_adapter(证据)（零初始化）后再 LN；
    交叉注意力读 p 个 patch，再 LN；每个头的输入加上 out_adapter([证据, 特征])（零初始化）。
    """

    def __init__(self, c: Config3) -> None:
        super().__init__()
        if c.objective == "recon" or c.dp_bins > 0:
            if c.dp_bins <= 0 or c.dp_bins % 2 == 0:
                raise ValueError("dp_bins 必须为正奇数（由训练数据拟合）")
        d, dim = c.d, c.ev_dim
        self.gene_query = nn.Linear(c.esm_dim, d)
        self.query_norm = nn.LayerNorm(d)
        self.cross = CrossAttn(d, c.n_heads, c.ffn_mult)
        self.ln = nn.LayerNorm(d)
        self.de = nn.Linear(d, 2)
        self.lfc = nn.Linear(d, 1)
        self.dp = nn.Linear(d, max(c.dp_bins, 1))
        if c.nb_resid:                                                           # v8：初始残差 E[r] = 0（各箱等概率，箱中心对称）
            nn.init.zeros_(self.dp.weight)
            nn.init.zeros_(self.dp.bias)
        self.evidence = Evidence(c)
        self.query_adapter = zero_last(nn.Sequential(nn.Linear(dim, 2 * dim), nn.SiLU(), nn.Linear(2 * dim, d, bias=False)))
        heads = ("de", "lfc", "dp", "cont") if c.cont_head else ("de", "lfc", "dp")
        self.out_adapters = nn.ModuleDict({h: zero_last(nn.Sequential(
            nn.RMSNorm(dim + d), nn.Linear(dim + d, dim), nn.SiLU(), nn.Dropout(c.ev_dropout),
            nn.Linear(dim, d, bias=False))) for h in heads})
        self.cont = nn.Linear(d, 1) if c.cont_head else None                    # N2
        if c.recon_loss != "pie" and not c.cont_head:
            raise ValueError("recon_loss mse / zinb 需要 cont_head")
        self.zi = None
        if c.recon_loss == "zinb":                                               # v10：零膨胀概率的 logit（初始 π ≈ 0.018）
            self.zi = nn.Linear(d, 1)
            nn.init.zeros_(self.zi.weight)
            nn.init.constant_(self.zi.bias, -4.0)
        self.dec_type = c.dec_type
        if c.dec_type in ("xattn_sa", "affine"):                                 # v7：2 层基因 self-attention
            self.gene_sa = nn.ModuleList([GeneSA(d, c.n_heads, c.ffn_mult) for _ in range(2)])
        if c.dec_type == "affine":                                               # v7：AffineExprDecoder（CellPatch2 newdecoder）
            self.coeff = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1), nn.Softplus())
            self.bias = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))
            if c.nb_resid:
                raise ValueError("nb_resid 只用于分箱头（dec_type xattn / xattn_sa）")
            if c.affine_id:                                                      # 10-21 修正：初始 ŷ = x（δ = 0）
                with torch.no_grad():
                    self.coeff[2].weight.zero_()
                    self.coeff[2].bias.fill_(math.log(math.e - 1))
                    self.bias[2].weight.zero_()
                    self.bias[2].bias.zero_()

    def forward(self, z: Tensor, gene_esm: Tensor, evidence: Optional[dict] = None, x: Optional[Tensor] = None):
        """→ (de_logits, lfc, dp_logits, cont, y_aff, zi)；affine 时 lfc / dp_logits / cont 为 None，y_aff = ŷ [B, G]；
        zi：v10 zinb 的零膨胀 logit [B, G]（与连续头共用输入），否则 None"""
        q = self.gene_query(gene_esm).expand(z.shape[0], -1, -1)              # [B, G, d]
        fused = None
        if evidence is not None:
            fused = self.evidence(evidence)                                     # [B, G, ev_dim]
            q = q + self.query_adapter(fused)
        h = self.ln(self.cross(self.query_norm(q), z))
        if self.dec_type in ("xattn_sa", "affine"):
            for blk in self.gene_sa:
                h = blk(h)
        if self.dec_type == "affine":
            y_aff = self.coeff(h).squeeze(-1) * x + self.bias(h).squeeze(-1)
            return self.de(h), None, None, None, y_aff, None
        if fused is None:
            cont = self.cont(h).squeeze(-1) if self.cont is not None else None
            zi = self.zi(h).squeeze(-1) if self.zi is not None else None
            return self.de(h), self.lfc(h).squeeze(-1), self.dp(h), cont, None, zi
        cat = torch.cat([fused, h], dim=-1)
        a = self.out_adapters
        hc = h + a["cont"](cat) if self.cont is not None else None
        cont = self.cont(hc).squeeze(-1) if self.cont is not None else None
        zi = self.zi(hc).squeeze(-1) if self.zi is not None else None
        return (self.de(h + a["de"](cat)), self.lfc(h + a["lfc"](cat)).squeeze(-1), self.dp(h + a["dp"](cat)), cont, None, zi)


class PassDecoder(nn.Module):
    """v11 证据直通 decoder。z [B, p, d]，esm [G, esm_dim]（固定），feat [B, G, F]，base [B, G]（证据，打分空间 δ），s 标量 → δ̂ [B, G]"""

    def __init__(self, c: Config3, n_feat: int, hidden: int = 256) -> None:
        super().__init__()
        d = c.d
        self.gene_query = nn.Linear(c.esm_dim, d)
        self.query_norm = nn.LayerNorm(d)
        self.cross = CrossAttn(d, c.n_heads, c.ffn_mult)
        self.ln = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d + n_feat, hidden), nn.SiLU(), nn.Linear(hidden, 2))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, z: Tensor, esm: Tensor, feat: Tensor, base: Tensor, s: float) -> Tensor:
        q = self.query_norm(self.gene_query(esm)).expand(z.shape[0], -1, -1)   # [B, G, d]
        h = self.ln(self.cross(q, z))
        ac = self.mlp(torch.cat([h, feat.to(h.dtype)], dim=-1)).float()        # [B, G, 2]
        return (1.0 + ac[..., 0]) * base.float() + s * ac[..., 1]


def dp_centers(c: Config3, device) -> Tensor:
    """K 个箱的中心值（PIE dequantize）"""
    k = torch.arange(c.dp_bins, device=device, dtype=torch.float32)
    return ((k + 0.5) / c.dp_bins * 2 - 1) * c.dp_max


def two_hot(delta_p: Tensor, c: Config3):
    """连续 delta_p → (软标签 [..., K], valid [...])：每个值按线性插值分给相邻两个箱中心（PIE two_hot）"""
    K = c.dp_bins
    valid = torch.isfinite(delta_p)
    d = torch.nan_to_num(delta_p, nan=0.0).clamp(-c.dp_max, c.dp_max)
    u = (d / c.dp_max + 1.0) * 0.5 * K - 0.5
    lo = u.floor().clamp(0, K - 2).long()
    frac = (u - lo.to(u.dtype)).clamp(0.0, 1.0)
    t = torch.zeros(*delta_p.shape, K, device=delta_p.device, dtype=u.dtype)
    t.scatter_(-1, lo.unsqueeze(-1), (1.0 - frac).unsqueeze(-1))
    t.scatter_add_(-1, (lo + 1).unsqueeze(-1), frac.unsqueeze(-1))
    return t * valid.unsqueeze(-1), valid


def pie_loss(out, labels: dict, c: Config3) -> dict:
    """PIE 的四项 loss，权重各为 1（pie/model/losses.py）。

    labels：de_mask [B, G] bool、tested [B, G] bool、lfc [B, G]（log2，未检验为 NaN）、
            delta_p [B, G]、target [B]（靶基因在本批基因中的位置，-1 = 无）
    """
    de_mask, tested = labels["de_mask"], labels["tested"]
    # de：两类交叉熵，每行按类别频率倒数加权（上限 cap）
    logits = out.de_logits.float()
    tgt = de_mask.long()
    ce = F.cross_entropy(logits.reshape(-1, 2), tgt.reshape(-1), reduction="none").view_as(tgt).float()
    keep = tested.float()
    counts = torch.zeros(tgt.shape[0], 2, device=tgt.device).scatter_add_(1, tgt, keep)
    w = (keep.sum(1, keepdim=True).clamp(min=1) / (2 * counts.clamp(min=1))).clamp(max=c.class_weight_cap)
    de = (ce * w.gather(1, tgt) * keep).sum() / keep.sum().clamp(min=1)
    # lfc：DE 基因上的 Huber；靶基因单独算，乘 alpha
    lfc_t = labels["lfc"]
    m = de_mask & torch.isfinite(lfc_t)
    pred = out.lfc.float()
    tcell = torch.zeros_like(m)
    rows = torch.nonzero(labels["target"] >= 0, as_tuple=True)[0]
    tcell[rows, labels["target"][rows]] = True
    def huber(cells):
        return F.huber_loss(pred[cells], lfc_t[cells], delta=c.lfc_huber_delta) if cells.any() else pred.sum() * 0
    lfc = huber(m & ~tcell) + c.lfc_target_alpha * huber(m & tcell)
    # 方向：relu(−sign(y) · ŷ / tau)，同一组基因中 y ≠ 0 的
    md = m & (lfc_t != 0)
    sgn = torch.sign(torch.where(md, lfc_t, torch.zeros_like(lfc_t)))
    dirl = torch.where(md, F.relu(-sgn * pred / c.lfc_dir_tau), torch.zeros_like(pred)).sum() / md.sum().clamp(min=1)
    # dp：两箱软标签交叉熵，每行按软计数的频率倒数加权（上限 cap_dp）
    dp_t = labels["delta_p"] if out.dp_base is None else labels["delta_p"] - out.dp_base.float()   # v8：残差目标
    probs, valid = two_hot(dp_t, c)
    ce = -(probs * F.log_softmax(out.dp_logits.float(), dim=-1)).sum(-1)
    keep = valid.float()
    soft = torch.einsum("bgk,bg->bk", probs, keep)
    w = (keep.sum(1, keepdim=True).clamp(min=1) / (c.dp_bins * soft.clamp(min=1e-6))).clamp(max=c.class_weight_cap_dp)
    dp = (ce * torch.einsum("bgk,bk->bg", probs, w) * keep).sum() / keep.sum().clamp(min=1)
    return {"loss": de + lfc + dirl + dp, "de_loss": de, "lfc_loss": lfc, "lfc_dir_loss": dirl, "dp_loss": dp}


def affine_loss(out, labels: dict, c: Config3) -> Tensor:
    """v7 affine decoder：MSE(ŷ, x + delta_p)（群体效应加在本样本对照上，只算 delta_p 有限的基因）+ DE 两类交叉熵（同 pie_loss 的 de 项）"""
    tgt_y = out.x_in.float() + torch.nan_to_num(labels["delta_p"].float())
    ok = torch.isfinite(labels["delta_p"])
    mse = F.mse_loss(out.y_aff.float()[ok], tgt_y[ok])
    de_mask, tested = labels["de_mask"], labels["tested"]
    logits = out.de_logits.float()
    tgt = de_mask.long()
    ce = F.cross_entropy(logits.reshape(-1, 2), tgt.reshape(-1), reduction="none").view_as(tgt).float()
    keep = tested.float()
    counts = torch.zeros(tgt.shape[0], 2, device=tgt.device).scatter_add_(1, tgt, keep)
    w = (keep.sum(1, keepdim=True).clamp(min=1) / (2 * counts.clamp(min=1))).clamp(max=c.class_weight_cap)
    de = (ce * w.gather(1, tgt) * keep).sum() / keep.sum().clamp(min=1)
    return mse + de


def bulk_mse(out, labels: dict, s_dp: Tensor) -> Tensor:
    """v10 mse：MSE(δ̂ / s_dp, δ / s_dp)，δ = 打分空间群体 δ（labels["dB"]）；连续头本身就是 δ̂ / s_dp。只算 labels["gmask"] 的格子。"""
    m = labels["gmask"]
    return ((out.dp_cont.float() - labels["dB"].float() / s_dp) ** 2)[m].mean()


def zinb_nll(out, labels: dict, s_dp: Tensor, log_theta: Tensor) -> Tensor:
    """v10 zinb：labels["counts"] [B, n, G] 计数、labels["lib"] [B, n] 文库、labels["bc"] [G] 对照打分空间伪 bulk、labels["gmask"] [B, G]。
    q̂ = expm1(softplus₅₀(b_c + δ̂)) / 5e4；NB 均值 μ = 文库 · q̂ / (1 − π)；ZINB 负对数似然在 gmask 的格子上平均。"""
    b = F.softplus(labels["bc"].float() + out.dp_cont.float() * s_dp, beta=50.0)   # 预测的打分空间伪 bulk（> 0）
    q = torch.expm1(b) / 5e4                                                      # [B, G]
    zl = out.zi_logit.float()
    log_pi, log_1mpi = F.logsigmoid(zl), F.logsigmoid(-zl)
    x = labels["counts"].float()                                                  # [B, n, G]
    mu = labels["lib"].float()[..., None] * (q * torch.exp(-log_1mpi))[:, None]  # [B, n, G]
    th = log_theta.float().exp()[None, None]
    log_th_mu = torch.log(th + mu + 1e-8)
    nb0 = th * (log_theta.float()[None, None] - log_th_mu)                        # log NB(0)
    log_nb = (torch.lgamma(x + th) - torch.lgamma(th) - torch.lgamma(x + 1.0) + nb0
              + x * (torch.log(mu + 1e-8) - log_th_mu))
    ll = torch.where(x < 0.5, torch.logaddexp(log_pi[:, None], log_1mpi[:, None] + nb0), log_1mpi[:, None] + log_nb)
    m = labels["gmask"][:, None].expand_as(x)
    return -ll[m].mean()


# ============================================================================
# 组装
# ============================================================================

@dataclass
class Output3:
    cell_patch_x: Tensor        # [B, p, d]   cell_MSA 更新后
    pert_emb_x: Tensor          # [B, T, d]   pert_MSA 更新后
    cell_patch_x_prime: Tensor  # [B, p, d]   注入 pert 后
    cell_patch_y_hat: Tensor    # [B, p, d]
    de_logits: Tensor           # [B, G, 2]   DE 头
    lfc: Tensor                 # [B, G]      log2 倍数变化头
    dp_logits: Tensor           # [B, G, K]   delta_p 分箱头
    delta: Tensor               # [B, G]      delta_p 读出：argmax 箱的中心值（PIE readout），不可导
    y_hat: Tensor               # [B, G]      x + delta
    cell_patch0: Optional[Tensor] = None   # [B, p, d]  CellPatch(x)，MSA 之前（E1 / E2 用）
    x_rec: Optional[Tensor] = None         # [B, G]     E1：重建的标准化对照
    x_in: Optional[Tensor] = None          # [B, G]     输入 x
    dp_cont: Optional[Tensor] = None       # [B, G]     N2：连续头输出（δ / s_dp）
    nbw_dp: Optional[Tensor] = None        # [B, G]     N1：a_k 加权的邻居 δ（/ s_dp）
    nb_s: Optional[Tensor] = None          # [B, K]     N1：邻居得分 s_k
    nb_c: Optional[Tensor] = None          # [B, K]     N1：目标 c_k（训练时）
    nb_mask: Optional[Tensor] = None       # [B, K]
    y_aff: Optional[Tensor] = None         # [B, G]     v7 affine decoder 的 ŷ
    dp_base: Optional[Tensor] = None       # [B, G]     v8 nb_resid：残差的基准 nbw（delta_p 单位，停梯度）；δ̂ = dp_base + 分箱读出
    zi_logit: Optional[Tensor] = None      # [B, G]     v10 zinb：零膨胀概率 π 的 logit


def standardize_esm(e: Tensor) -> Tensor:
    """ESM 逐维标准化：只用有表征（非全零）的行计算均值 / 标准差，全零行保持 0。

    原始 ESM 平均表征的大部分范数是所有基因共有的分量（H1 的 147 个靶基因两两余弦均值 0.877，标准化后 0.033），
    不标准化时模型几乎无法区分不同基因 / 扰动。
    """
    e = e.float().clone()
    ok = e.abs().sum(1) > 0
    if ok.sum() >= 2:
        e[ok] = (e[ok] - e[ok].mean(0)) / e[ok].std(0).clamp_min(1e-6)
    return e


class PertFold3(nn.Module):
    """protein_emb [n_gene_vocab, esm_dim]：ESM2 表征，逐维标准化后作为基因 embedding 表 gene_table 的初始值；
    gene_table 可训练（写入 state_dict），供基因位置、扰动 token、decoder 基因查询共用。

    训练目标（Config3.objective，可用 set_objective 切换）：
      "recon"  pie_loss(out, labels)                群体标签：DE、LFC、delta_p（见 pie_loss）
      "jepa"   L1(cell_patch_y_hat, cell_patch_y)   cell_patch_y = LayerNorm(target_patch(y))（无可学习参数），不回传梯度
               target_patch 是 cell_patch 的 EMA 副本：初始权重相同，每个优化步后调用 update_target()
    预测侧与目标侧的归一化、L1 距离均照 I-JEPA / V-JEPA。

    两阶段训练：
      阶段 1  PertFold3(Config3(objective="jepa"), ...)，训练 ①②③（decoder 无梯度）
      阶段 2  model.set_objective("recon", freeze_backbone=True)   只训练 decoder
              model.set_objective("recon", freeze_backbone=False)  全部微调
              切换后需按 requires_grad 重新建优化器
    """

    def __init__(self, c: Config3, protein_emb: Tensor) -> None:
        super().__init__()
        if c.objective not in ("recon", "jepa", "mae"):
            raise ValueError("objective 只能为 'recon'、'jepa' 或 'mae'")
        if c.objective == "mae" and not c.mae:
            raise ValueError("objective='mae' 需要 mae=True")
        drop = set(filter(None, c.ev_drop.split(",")))
        if c.nb_learn and "nbw_dp" not in c.ev_blocks and not drop & {"nbw", "all"}:   # N1：加三个证据块（消融 nbw 时不加）
            c = dataclasses.replace(c, ev_blocks=tuple(c.ev_blocks) + ("nbw_dp", "nbw_de", "nbw_lfc"))
        if drop - {"all"}:                                                       # 消融：按类去掉证据块
            c = dataclasses.replace(c, ev_blocks=tuple(k for k in c.ev_blocks if k.split("_")[0] not in drop))
        if c.no_encoder and c.objective != "recon":
            raise ValueError("no_encoder 只能直接训练 recon（没有可预训练的 encoder）")
        self.c = c
        self.gene_table = nn.Embedding.from_pretrained(standardize_esm(protein_emb), freeze=c.freeze_genes)
        self.cell_patch = CellPatch(c)
        self.cell_msa = MSA(c.d, c.n_heads, c.ffn_mult, c.n_cell_msa)
        self.pert_emb = nn.Linear(c.esm_dim, c.n_pert_tokens * c.d)
        self.pert_msa = MSA(c.d, c.n_heads, c.ffn_mult, c.n_pert_msa)
        self.inject = CrossAttn(c.d, c.n_heads, c.ffn_mult)
        self.blocks = nn.ModuleList([SelfAttn(c.d, c.n_heads, c.ffn_mult) for _ in range(c.n_blocks)])
        self.pred_norm = nn.LayerNorm(c.d)                                      # I-JEPA predictor_norm
        self.pred_proj = nn.Linear(c.d, c.d)                                    # I-JEPA predictor_proj
        self.decoder = Decoder(c)
        if c.nb_learn:                                                           # N1
            self.nb_scorer = NbScorer(c)
        if c.nb_direct:                                                          # N3
            self.nb_alpha = nn.Parameter(torch.zeros(()))
        if c.nb_learn or c.cont_head:
            self.register_buffer("s_dp", torch.ones(()))                         # delta_p 的总体标准差（set_scale）
        if c.recon_loss == "zinb":                                               # v10：每基因 NB 离散度 θ = exp(log_theta)（set_theta 初始化）
            self.log_theta = nn.Parameter(torch.zeros(c.n_genes))
        if c.recon_weight > 0:                                                   # E1
            self.recon_head = ReconHead(c)
            self.register_buffer("rec_mu", torch.zeros(c.n_genes))
            self.register_buffer("rec_sd", torch.ones(c.n_genes))
        if c.idm_weight > 0:                                                     # v6：EP-IDM 头
            self.idm = nn.Sequential(nn.Linear(2 * c.d, c.idm_hidden), nn.ReLU(), nn.Linear(c.idm_hidden, c.idm_hidden),
                                     nn.ReLU(), nn.Linear(c.idm_hidden, c.n_actions))
        if c.mae:                                                                # v6：MAE 头与重建目标的标准化
            self.mae_head = ReconHead(c)
            self.register_buffer("mae_mu", torch.zeros(c.n_genes))
            self.register_buffer("mae_sd", torch.ones(c.n_genes))
        if c.no_encoder:                                                         # 消融：cell token = 可学习常量
            self.cell_const = nn.Parameter(torch.randn(c.n_patches, c.d))
        if c.ctrl_jepa > 0:                                                      # v7：对照 JEPA 的 predictor（2 层 SelfAttn + LN + Linear）
            self.ctrl_pred = nn.Sequential(*[SelfAttn(c.d, c.n_heads, c.ffn_mult) for _ in range(2)], nn.LayerNorm(c.d), nn.Linear(c.d, c.d))
        if c.const_z:                                                            # 消融：decoder 读常量
            self.z_const = nn.Parameter(torch.randn(c.n_patches, c.d))
        self.target_patch = None
        if c.objective == "jepa" and c.jepa_mode == "ema":
            self.target_patch = copy.deepcopy(self.cell_patch).requires_grad_(False)

    def set_objective(self, objective: str, freeze_backbone: bool = False) -> None:
        """切换训练目标。freeze_backbone=True：除 decoder 外全部冻结（target_patch 始终冻结）。"""
        if objective not in ("recon", "jepa", "mae"):
            raise ValueError("objective 只能为 'recon'、'jepa' 或 'mae'")
        self.c = dataclasses.replace(self.c, objective=objective)
        if objective == "jepa" and self.c.jepa_mode == "ema" and self.target_patch is None:
            self.target_patch = copy.deepcopy(self.cell_patch).requires_grad_(False)
        for name, child in self.named_children():
            if name != "target_patch":
                child.requires_grad_(name == "decoder" or not freeze_backbone)
        if self.c.freeze_genes:                                                  # A5
            self.gene_table.requires_grad_(False)

    @torch.no_grad()
    def set_theta(self, theta: Tensor) -> None:
        """v10 zinb：每基因 NB 离散度的初值（训练脚本按对照细胞矩估计）"""
        self.log_theta.copy_(theta.float().log())

    @torch.no_grad()
    def set_scale(self, s_dp: float) -> None:
        """N1 / N2：delta_p 的总体标准差（与证据的归一化相同）"""
        self.s_dp.fill_(float(s_dp))

    def nb_evidence(self, pert_gene: Tensor, relate_genes: Tensor, nb: dict):
        """N1：→ (证据块 dict, 加权 δ [B, G], s [B, K])。nb：dp / lfc [B, K, G]（已归一化）、up / down / tested [B, K, G]、mask [B, K]、feat [B, K, F]"""
        mask = nb["mask"]
        s = self.nb_scorer(self.gene_table(pert_gene), self.gene_table(relate_genes), nb["feat"])
        a = torch.softmax((s / self.c.nb_tau).masked_fill(~mask, float("-inf")), dim=-1)
        a = torch.nan_to_num(a, nan=0.0) * mask                                   # 没有邻居的行全为 0
        A = a[..., None]
        have = mask.any(1).float()[:, None].expand(-1, nb["dp"].shape[-1])
        logn = torch.log1p(mask.sum(1).float())[:, None].expand_as(have)
        dpw = (A * nb["dp"]).sum(1)
        disp = ((A * (nb["dp"] - dpw[:, None]) ** 2).sum(1)).clamp_min(0).sqrt()
        t = nb["tested"].float()
        at = (A * t).sum(1).clamp_min(1e-6)
        upw, downw = (A * nb["up"] * t).sum(1) / at, (A * nb["down"] * t).sum(1) / at
        lfcw = (A * nb["lfc"] * t).sum(1) / at
        smax = s.masked_fill(~mask, -1.0).max(1).values[:, None].expand_as(have)
        blocks = {"nbw_dp": torch.stack([dpw, disp, have, logn], -1),
                  "nbw_de": torch.stack([upw, downw, have, logn], -1),
                  "nbw_lfc": torch.stack([lfcw, smax, have, logn], -1)}
        return blocks, dpw, s

    @torch.no_grad()
    def set_mae_norm(self, mu: Tensor, sd: Tensor) -> None:
        """v6 MAE：重建目标的标准化"""
        self.mae_mu.copy_(mu)
        self.mae_sd.copy_(sd)

    def mae_loss(self, gene_idx: Tensor, bulks: Tensor) -> Tensor:
        """v6 MAE：bulks [N, G]；每行随机遮住 round(mae_ratio · G) 个基因，只在被遮基因上算 MSE（标准化表达）"""
        N, G = bulks.shape
        k = int(round(self.c.mae_ratio * G))
        pad = torch.zeros(N, G, dtype=torch.bool, device=bulks.device)
        pad.scatter_(1, torch.rand(N, G, device=bulks.device).argsort(1)[:, :k], True)
        ge = self.gene_table(gene_idx)
        rec = self.mae_head(self.cell_patch(bulks, ge, pad=pad), ge)
        tgt = (bulks - self.mae_mu) / self.mae_sd
        return F.mse_loss(rec.float()[pad], tgt.float()[pad])

    @torch.no_grad()
    def set_rec_norm(self, mu: Tensor, sd: Tensor) -> None:
        """E1：重建目标的标准化（对照 bulk 的均值 / 标准差）"""
        self.rec_mu.copy_(mu)
        self.rec_sd.copy_(sd)

    @torch.no_grad()
    def set_value_norm(self, mu: Tensor, sd: Tensor) -> None:
        """A1：写入对照均值 / 标准差（在线与 EMA 目标编码器都写）"""
        for enc in (self.cell_patch, self.target_patch):
            if enc is not None:
                enc.ctrl_mu.copy_(mu)
                enc.ctrl_sd.copy_(sd)

    def encode_cell(self, gene_idx: Tensor, x: Tensor, target: bool = False) -> Tensor:
        """x [..., G] → [..., p, d]。target=True 时用 EMA 目标编码器（JEPA 的 cell_patch_y）。"""
        enc = self.target_patch if target else self.cell_patch
        lead = x.shape[:-1]
        z = enc(x.reshape(-1, x.shape[-1]), self.gene_table(gene_idx))
        return z.reshape(*lead, *z.shape[1:])

    def encode_z(self, gene_idx: Tensor, x: Tensor, relate_x: Tensor, pert_gene: Tensor, relate_genes: Tensor,
                 relate_x_mask: Optional[Tensor] = None, relate_genes_mask: Optional[Tensor] = None, **_) -> Tensor:
        """v11：只算 backbone，返回 z [B, p, d]（与 forward 里 decoder 读的 z 相同）"""
        c = self.c
        if c.no_encoder or c.input_mask > 0 or c.const_z:
            raise ValueError("encode_z 不支持 no_encoder / input_mask / const_z")
        cell_patch_x = self.encode_cell(gene_idx, x)
        if not c.no_cell_msa:
            with torch.set_grad_enabled(c.relate_grad and torch.is_grad_enabled()):
                rel = self.encode_cell(gene_idx, relate_x)
            cell_patch_x = self.cell_msa(cell_patch_x, rel, relate_x_mask)
        p = self.encode_pert(pert_gene)
        if not c.no_pert_msa:
            with torch.set_grad_enabled(c.relate_grad and torch.is_grad_enabled()):
                pr = self.encode_pert(relate_genes)
            p = self.pert_msa(p, pr, relate_genes_mask)
        z = self.inject(cell_patch_x, p)
        for blk in self.blocks:
            z = blk(z)
        return self.pred_proj(self.pred_norm(z))

    def encode_pert(self, pert_gene: Tensor) -> Tensor:
        """pert_gene [...] long → [..., T, d]"""
        return self.pert_emb(self.gene_table(pert_gene)).unflatten(-1, (self.c.n_pert_tokens, self.c.d))

    def forward(
        self,
        gene_idx: Tensor,               # [G] long
        x: Tensor,                      # [B, G]       pert 前的细胞
        relate_x: Tensor,               # [B, Rc, G]   相关细胞
        pert_gene: Tensor,              # [B] long
        relate_genes: Tensor,           # [B, Rp] long 相关 pert 基因
        relate_x_mask: Optional[Tensor] = None,       # [B, Rc] bool，True = 有效
        relate_genes_mask: Optional[Tensor] = None,   # [B, Rp] bool
        evidence: Optional[dict] = None,              # 每个基因的证据：ctrl [B, G]、c.ev_blocks 中各块 [B, G, 4]
        nb: Optional[dict] = None,                    # N1：候选邻居的标签统计（见 nb_evidence）；c [B, K] 训练时给
    ) -> Output3:
        c = self.c
        if c.no_encoder:                                                         # 消融：不读 x 与邻居细胞
            cell_patch_x = self.cell_const.expand(x.shape[0], -1, -1)
            cell_patch0 = x_rec = None
        else:
            if c.input_mask > 0 and self.training:                               # v7：对照输入随机遮住部分基因
                G = x.shape[-1]
                pad = torch.zeros(x.shape[0], G, dtype=torch.bool, device=x.device)
                pad.scatter_(1, torch.rand(x.shape[0], G, device=x.device).argsort(1)[:, :int(round(c.input_mask * G))], True)
                cell_patch_x = self.cell_patch(x, self.gene_table(gene_idx), pad=pad)
            else:
                cell_patch_x = self.encode_cell(gene_idx, x)
            cell_patch0 = cell_patch_x
            x_rec = self.recon_head(cell_patch0, self.gene_table(gene_idx)) if c.recon_weight > 0 else None
            if not c.no_cell_msa:
                with torch.set_grad_enabled(c.relate_grad and torch.is_grad_enabled()):   # 原 v3：相关细胞只提供特征，不回传梯度（T5 打开）
                    cell_patch_relate_x = self.encode_cell(gene_idx, relate_x)
                cell_patch_x = self.cell_msa(cell_patch_x, cell_patch_relate_x, relate_x_mask)

        pert_emb_x = self.encode_pert(pert_gene)
        if not c.no_pert_msa:
            with torch.set_grad_enabled(c.relate_grad and torch.is_grad_enabled()):   # 原 v3：相关基因不回传梯度（T5 打开）
                pert_embs_relate = self.encode_pert(relate_genes)
            pert_emb_x = self.pert_msa(pert_emb_x, pert_embs_relate, relate_genes_mask)

        h = self.inject(cell_patch_x, pert_emb_x)
        z = h
        for blk in self.blocks:
            z = blk(z)
        z = self.pred_proj(self.pred_norm(z))                                   # cell_patch_y_hat
        dpw = s_nb = None
        drop = set(filter(None, c.ev_drop.split(",")))
        if "all" in drop:                                                        # 消融：decoder 不接收证据
            evidence = None
        if c.nb_learn and nb is not None:
            blocks, dpw, s_nb = self.nb_evidence(pert_gene, relate_genes, nb)
            if evidence is not None and "nbw" not in drop:
                evidence = {**evidence, **blocks}
        if evidence is not None:                                                 # 消融：只传模型用到的证据块
            evidence = {k: v for k, v in evidence.items() if k == "ctrl" or k in c.ev_blocks}
        zd = self.z_const.expand(z.shape[0], -1, -1) if c.const_z else z         # 消融：decoder 读常量
        de_logits, lfc, dp_logits, cont, y_aff, zi = self.decoder(zd, self.gene_table(gene_idx), evidence, x)
        if cont is not None and self.c.nb_direct and dpw is not None:            # N3
            cont = cont + self.nb_alpha * dpw
        if y_aff is not None:                                                    # v7 affine：δ̂ = ŷ − x
            delta = y_aff - x
        elif c.recon_loss != "pie":                                              # v10：连续头 × s_dp = 打分空间 δ̂
            delta = cont.float() * self.s_dp
        else:
            delta = dp_centers(self.c, x.device)[dp_logits.argmax(-1)] if self.c.dp_bins > 0 else torch.zeros_like(x)
        dp_base = None
        if c.nb_resid:                                                           # v8：δ̂ = nbw + 残差
            if dpw is None:
                raise ValueError("nb_resid 需要 nb_learn 且传入 nb")
            dp_base = (dpw * self.s_dp).detach().to(delta.dtype)
            delta = delta + dp_base
        return Output3(cell_patch_x, pert_emb_x, h, z, de_logits, lfc, dp_logits, delta, x + delta, cell_patch0, x_rec, x,
                       cont, dpw, s_nb, None if nb is None else nb.get("c"), None if nb is None else nb["mask"], y_aff, dp_base, zi)

    def loss(self, out: Output3, gene_idx: Tensor, y: Optional[Tensor] = None,
             labels: Optional[dict] = None) -> Tensor:
        """recon：labels（群体标签，见 pie_loss）；jepa：y [B, G]，pert 后的细胞。"""
        loss = self._main_loss(out, gene_idx, y, labels)
        c = self.c
        if c.recon_weight > 0 and out.x_rec is not None:                        # E1：重建对照
            tgt = (out.x_in.float() - self.rec_mu) / self.rec_sd
            loss = loss + c.recon_weight * F.mse_loss(out.x_rec.float(), tgt)
        if c.patch_var_weight > 0 and out.cell_patch0 is not None:              # E2：跨样本防塌缩
            f = out.cell_patch0.float().flatten(1)
            loss = loss + c.patch_var_weight * F.relu(c.patch_var_gamma - torch.sqrt(f.var(0) + 1e-4)).mean()
        if c.nb_learn and out.nb_s is not None and out.nb_c is not None and out.nb_mask.any():   # N1：s_k 回归 c_k
            m = out.nb_mask
            loss = loss + c.nb_weight * F.mse_loss(out.nb_s.float()[m], out.nb_c.float()[m])
        if c.cont_head and c.recon_loss == "pie" and c.objective == "recon" and labels is not None and out.dp_cont is not None:   # N2
            t = labels["delta_p"].float()
            ok = torch.isfinite(t)
            loss = loss + c.cont_weight * F.mse_loss(out.dp_cont.float()[ok], (t / self.s_dp)[ok])
        if c.sigreg_weight > 0 and out.cell_patch0 is not None:                  # J：SIGReg
            loss = loss + c.sigreg_weight * sigreg(out.cell_patch0.float().flatten(1), c.sigreg_slices)
        if c.mse_weight > 0 and c.objective == "recon" and labels is not None and out.dp_logits is not None:   # v9：δ̂ 的 MSE
            dh = out.dp_logits.float().softmax(-1) @ dp_centers(c, out.dp_logits.device)
            if out.dp_base is not None:
                dh = dh + out.dp_base.float()
            t = labels["delta_p"].float()
            ok = torch.isfinite(t)
            if "nontarget" in labels:
                ok = ok & labels["nontarget"][None, :]
            sc = self.s_dp if hasattr(self, "s_dp") else 1.0
            loss = loss + c.mse_weight * F.mse_loss(dh[ok] / sc, t[ok] / sc)
        if c.disc_weight > 0 and c.objective == "recon" and labels is not None:  # D：区分度 InfoNCE
            dh = (out.dp_logits.float().softmax(-1) @ dp_centers(c, out.dp_logits.device))
            if out.dp_base is not None:                                          # v9：nb_resid 时 δ̂ = nbw + E[残差]
                dh = dh + out.dp_base.float()
            tt = torch.nan_to_num(labels["delta_p"].float())
            if "nontarget" in labels:
                dh, tt = dh[:, labels["nontarget"]], tt[:, labels["nontarget"]]
            logits = F.normalize(dh, dim=1) @ F.normalize(tt, dim=1).T / c.disc_tau
            pid = labels["pid"]
            B = pid.numel()
            eye = torch.eye(B, dtype=torch.bool, device=pid.device)
            logits = logits.masked_fill((pid[:, None] == pid[None, :]) & ~eye, float("-inf"))
            loss = loss + c.disc_weight * F.cross_entropy(logits, torch.arange(B, device=pid.device))
        if c.pert_var_weight > 0:                                                 # P1：扰动一路跨样本 VICReg
            f = out.pert_emb_x.float().mean(1)
            fc = f - f.mean(0)
            var = F.relu(1 - torch.sqrt(f.var(0) + 1e-4)).mean()
            cov = fc.T @ fc / max(f.shape[0] - 1, 1)
            off = cov - torch.diag(torch.diag(cov))
            loss = loss + c.pert_var_weight * (var + c.pert_cov_weight * (off ** 2).sum() / f.shape[1])
        return loss

    def _main_loss(self, out: Output3, gene_idx: Tensor, y: Optional[Tensor] = None,
                   labels: Optional[dict] = None) -> Tensor:
        c = self.c
        if c.objective == "recon":
            if c.dec_type == "affine":
                return affine_loss(out, labels, c)
            if c.recon_loss == "mse":                                            # v10
                return bulk_mse(out, labels, self.s_dp)
            if c.recon_loss == "zinb":
                return zinb_nll(out, labels, self.s_dp, self.log_theta)
            d = pie_loss(out, labels, c)                                         # v9：四项可加权（默认全 1 = 原 loss）
            return c.w_de * d["de_loss"] + c.w_lfc * d["lfc_loss"] + c.w_dir * d["lfc_dir_loss"] + c.w_dp * d["dp_loss"]
        if c.objective != "jepa":
            raise ValueError("objective='mae' 用 mae_loss()")
        zy = None
        if c.jepa_mode == "ema":
            with torch.no_grad():
                cell_patch_y = self.encode_cell(gene_idx, y, target=True)
                cell_patch_y = F.layer_norm(cell_patch_y, (cell_patch_y.shape[-1],))   # I-JEPA：目标再做无参数 LN
            loss = F.l1_loss(out.cell_patch_y_hat, cell_patch_y)
        else:                                                                    # v6 "sg"：在线编码器的停梯度目标，MSE
            zy = self.encode_cell(gene_idx, y)
            loss = F.mse_loss(out.cell_patch_y_hat.float(), zy.detach().float())
        if c.idm_weight > 0 or c.jepa_sigreg > 0:                                # v6：两端都用在线编码器、回传梯度
            zy = self.encode_cell(gene_idx, y) if zy is None else zy
            ends = (out.cell_patch0.mean(1), zy.mean(1))                         # 池化 latent [B, d]
            if c.idm_weight > 0:                                                 # EP-IDM：(z_x, z_y) → 扰动类别
                logits = self.idm(torch.cat(ends, -1))
                loss = loss + c.idm_weight * F.cross_entropy(logits.float(), labels["pid"])
            if c.jepa_sigreg > 0:
                loss = loss + c.jepa_sigreg * sigreg(torch.cat(ends).float(), c.sigreg_slices)
        if c.ctrl_jepa > 0 and out.cell_patch0 is not None:                      # v7：对照 JEPA
            xin = out.x_in
            G = xin.shape[-1]
            pad = torch.zeros(xin.shape[0], G, dtype=torch.bool, device=xin.device)
            pad.scatter_(1, torch.rand(xin.shape[0], G, device=xin.device).argsort(1)[:, :int(round(c.ctrl_jepa_ratio * G))], True)
            pred = self.ctrl_pred(self.cell_patch(xin, self.gene_table(gene_idx), pad=pad))
            with torch.no_grad():                                                # 目标 = 完整对照的在线编码（输入 mask 打开时重新编码）
                tgt = self.encode_cell(gene_idx, xin) if c.input_mask > 0 else out.cell_patch0
            loss = loss + c.ctrl_jepa * F.mse_loss(pred.float(), tgt.detach().float())
        if self.c.var_weight > 0:                                                # T2：VICReg 方差项（批内每个特征的标准差 ≥ 1）
            f = out.cell_patch_x.float().flatten(1)
            loss = loss + self.c.var_weight * F.relu(1 - torch.sqrt(f.var(0) + 1e-4)).mean()
        return loss

    @torch.no_grad()
    def update_target(self) -> None:
        """JEPA：target_patch ← m · target_patch + (1 − m) · cell_patch。每个优化步之后调用。"""
        if self.c.objective != "jepa" or self.target_patch is None:
            return
        m = self.c.ema_momentum
        for pt, p in zip(self.target_patch.parameters(), self.cell_patch.parameters()):
            pt.mul_(m).add_(p.detach(), alpha=1 - m)
