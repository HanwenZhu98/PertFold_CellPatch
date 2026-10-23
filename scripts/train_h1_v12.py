"""v12（2026-10-23，用户要求）：r4-v2 主体 + 修好的内容 + 全基因 ZINB + 弱 DE + nb loss；可选 对照重构 / 能量距离 / 判别器。
split_811（train 118 / val 15 / test 14），从 runs3/h1_v7/n32/split_811/V7_S1c修正_s<种子>/stage1.pt 出发，只跑阶段 2（全部微调）。

结构（r4-v2 不变的部分）：CellPatch → cell_MSA → pert_MSA → 扰动注入 → 4 层 Transformer → z；NbScorer 学邻居权重，nbw 证据进 decoder
（Evidence 融合 + query_adapter + out_adapter，ev_drop ctx,str,ctrl）。
修好的内容：
  输出基因 = 对照 CPM ≥ 1 ∪ 原 2,125 个基因（prep_v12.py，约 12,800 个）；decoder 的基因查询来自固定的标准化 ESM，与 encoder 输入（2,125 个基因）解耦；
  打分空间：δ̂ = 连续头 × s_dp 是 log1p(5e4·伪 bulk 比例) 之差；邻居证据也在打分空间（输出基因）；
  供体：训练时 = train；选模（val）时 = train；测试推理时 = train + val，尺度冻结为训练时的值。
loss：
  ZINB（主）：该扰动 32 个真实细胞在输出基因上的计数，均值 = 文库 · expm1(softplus₅₀(b_c + δ̂)) / 5e4 / (1 − π)，θ 每基因一个；去掉本扰动靶基因列
  弱 DE（--w-de 0.1）：decoder 的 DE 头，类别平衡交叉熵（同 PIE），标签 = 全部细胞 Wilcoxon p_adj < 0.05（de_full.parquet）
  nb：MSE(s_k, c_k)，c_k = 邻居与查询扰动打分空间 δ 的相关（只训练扰动）
  --ctrl-recon λ：每步另加 8 个"无扰动"样本（扰动 token 换成可学习的空 token、没有邻居），目标 = 另 32 个对照细胞的计数，ZINB
  --energy λ：生成细胞 = 32 个对照模板细胞的计数 × 预测倍数 r = q̂ / q_c（连续、可导），与 32 个真实扰动细胞在
              log1p(1e4·计数/文库) 的对照 PCA 50 维空间里算能量距离（去掉本扰动靶基因列）
  --adv λ：判别器（谱归一化 MLP，输入 = 细胞的 PCA 50 维 + 扰动条件），hinge loss；生成器项 = −λ · D(生成细胞)
选模：每 25 步在 val 上算 PDS 代理（预测与 train + val 真值 δ 的余弦距离排名，去掉面板内全部靶基因列），取最高。
预测：val（供体 train，第 0 次重复）与 test（供体 train + val，5 次重复），抽样与 predict_h1_v5.py 相同（同一随机数序列）。

5 折模式（--fold k，10-23 夜加）：test = 第 k 折；val = 本折训练扰动中按 train_h1_v5 --inner-val 的规则抽 23 个；train = 其余；
  stage1 = runs3/h1_v7k/n32/fold<k>/K_S1c修正/stage1.pt；只预测 test（5 次重复），文件名 pred_<tag>_s<seed>_fold<k>_test_rep<r>.npz，
  merge_folds.py 合并 5 折后用 summarize_bench.py 交叉拟合选 s。
用法：python train_h1_v12.py --seed 0 --tag Z0 [--fold k] [--ctrl-recon 1] [--energy λ] [--adv λ] [--s2 1000] [--gpu 0]
输出：runs3/h1_v12/n32/split_811/<tag>_s<seed>/{best_pds.pt, log.jsonl, manifest.json}、runs3/h1_v12/n32/pred_<tag>_s<seed>_{val,test}_rep<r>.npz
"""
import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "/cluster2/home/hanwen/vcc_26/pertfold_r4")
from pertfold3.model_v5 import Config3, PertFold3, zinb_nll  # noqa: E402
from train_h1_v5 import DATA, H1Data, lr_at  # noqa: E402
from train_h1_v11 import out_esm  # noqa: E402

V12_DIR = DATA / "v12"
RUNS = Path("/cluster2/home/hanwen/vcc_26/pertfold/runs3")
import os
OUT = Path(os.environ.get("V12_OUT", "/cluster2/home/hanwen/vcc_26/pertfold_r4/runs/h1_v12"))
N, BATCH, N_NULL, AMP = 32, 32, 8, True


def de_loss(logits, de, tested, cap=10.0):
    """类别平衡的两类交叉熵（同 pie_loss 的 de 项）"""
    tgt = de.long()
    ce = F.cross_entropy(logits.float().reshape(-1, 2), tgt.reshape(-1), reduction="none").view_as(tgt)
    keep = tested.float()
    counts = torch.zeros(tgt.shape[0], 2, device=tgt.device).scatter_add_(1, tgt, keep)
    w = (keep.sum(1, keepdim=True).clamp(min=1) / (2 * counts.clamp(min=1))).clamp(max=cap)
    return (ce * w.gather(1, tgt) * keep).sum() / keep.sum().clamp(min=1)


def energy_distance(a, b):
    """a [B, n, k]、b [B, m, k] → 每个样本的能量距离的平均（V 统计量）"""
    def md(x, y):
        return torch.cdist(x, y).mean((1, 2))
    return (2 * md(a, b) - md(a, a) - md(b, b)).mean()


class Disc(nn.Module):
    def __init__(self, k, d_cond):
        super().__init__()
        sn = nn.utils.parametrizations.spectral_norm
        self.cond = nn.Linear(d_cond, 64)
        self.net = nn.Sequential(sn(nn.Linear(k + 64, 256)), nn.LeakyReLU(0.2), sn(nn.Linear(256, 256)), nn.LeakyReLU(0.2),
                                 sn(nn.Linear(256, 1)))

    def forward(self, f, cond):                                                  # f [B, n, k]，cond [B, d]
        c = self.cond(cond)[:, None].expand(-1, f.shape[1], -1)
        return self.net(torch.cat([f, c], -1)).squeeze(-1)


class V12(nn.Module):
    def __init__(self, model, esm, theta0, pca_mu, pca_w, use_null):
        super().__init__()
        self.m = model
        self.register_buffer("esm", esm)
        self.log_theta = nn.Parameter(theta0.log())
        self.register_buffer("pc_mu", pca_mu)
        self.register_buffer("pc_w", pca_w)
        self.null = nn.Parameter(torch.zeros(model.c.n_pert_tokens, model.c.d)) if use_null else None

    def encode(self, inp, null=None):
        m, c = self.m, self.m.c
        cp = m.encode_cell(inp["gene_idx"], inp["x"])
        with torch.no_grad():
            rel = m.encode_cell(inp["gene_idx"], inp["relate_x"])
        cp = m.cell_msa(cp, rel, inp["relate_x_mask"])
        p = m.encode_pert(inp["pert_gene"])
        with torch.no_grad():
            pr = m.encode_pert(inp["relate_genes"])
        p = m.pert_msa(p, pr, inp["relate_genes_mask"])
        if null is not None:
            p = torch.where(null[:, None, None], self.null.to(p.dtype).expand_as(p), p)
        z = m.inject(cp, p)
        for blk in m.blocks:
            z = blk(z)
        return m.pred_proj(m.pred_norm(z)), p

    def forward(self, inp, nb, null=None):
        z, p = self.encode(inp, null)
        blocks, _, s_nb = self.m.nb_evidence(inp["pert_gene"], inp["relate_genes"], nb)
        ev = {k: v for k, v in blocks.items() if k in self.m.c.ev_blocks}
        ev["ctrl"] = torch.zeros(z.shape[0], self.esm.shape[0], device=z.device)   # Evidence 用它取 batch 大小（ev_drop 含 ctrl，不进模型）
        de_logits, _, _, cont, _, zi = self.m.decoder(z, self.esm, ev, None)
        return de_logits, cont, zi, s_nb, p


class Data12:
    """输出基因上的标签、计数缓存、证据（只依赖供体集合与冻结的尺度）"""

    def __init__(self, data, dev):
        self.d, self.dev = data, dev
        self.genes = np.load(V12_DIR / "genes.npy")
        lab = np.load(V12_DIR / "labels.npz")
        T = lambda a: torch.from_numpy(np.asarray(a)).to(dev)  # noqa: E731
        self.dB, self.bc = T(lab["dB"]), T(lab["bc"])
        self.de, self.tested = T(lab["de"]), T(lab["tested"])
        self.lfc = T(np.nan_to_num(lab["lfc"]))
        cells = np.load(V12_DIR / "cells.npz")
        self.counts = T(np.load(V12_DIR / "counts.npy"))                             # int16 [Nc, G]
        self.lib = T(cells["lib"])
        grp = cells["group"]
        P = self.dB.shape[0]
        self.cidx = [T(np.where(grp == p)[0]) for p in range(P)]
        self.ctrl_idx = T(np.where(grp == -1)[0])
        pca = np.load(V12_DIR / "pca.npz")
        self.pc_mu, self.pc_w = T(pca["mean"]), T(pca["comp"])
        var = json.loads((DATA / "var_names.json").read_text())
        gpos = {var[g]: j for j, g in enumerate(self.genes)}
        self.tcol = T(np.array([gpos.get(p, -1) for p in data.meta["perts"]]))
        self.names = [var[g] for g in self.genes]
        G = len(self.genes)
        self.panel = torch.ones(G, dtype=torch.bool, device=dev)                 # 类 PDS：去掉面板内全部靶基因列
        self.panel[self.tcol[self.tcol >= 0]] = False

    def set_donors(self, donors, scales=None):
        """邻居池（H1Data.set_evidence）+ 输出基因上的 nb 输入；scales=None 时按供体计算尺度（训练），否则用给定的（推理）"""
        d = self.d
        d.set_evidence(torch.tensor(donors, device=self.dev), self.relation)
        D = torch.tensor(donors, device=self.dev)
        if scales is None:
            s_dp = float(self.dB[D].std(unbiased=False))
            t = self.tested[D]
            clip = float(torch.quantile(self.lfc[D][t].abs().float()[:2_000_000], 0.95))
            s_lfc = float(self.lfc[D][t].clamp(-clip, clip).std(unbiased=False))
            scales = {"s_dp": s_dp, "s_lfc": s_lfc, "clip": clip}
        self.scales = scales
        self.dp_n = self.dB / scales["s_dp"]
        self.lfc_n = self.lfc.clamp(-scales["clip"], scales["clip"]) / scales["s_lfc"]
        self.up = (self.de & (self.lfc > 0)).float()
        self.down = (self.de & (self.lfc < 0)).float()
        X = self.dB.clone()
        X[:, ~self.panel] = 0
        X = X - X.mean(1, keepdim=True)
        X = X / X.norm(dim=1, keepdim=True).clamp_min(1e-12)
        self.C = X @ X.T
        return scales

    def nb(self, perts, with_c):
        idx, mask = self.d.str_idx[perts], self.d.str_mask[perts]
        out = {"dp": self.dp_n[idx], "lfc": self.lfc_n[idx], "up": self.up[idx], "down": self.down[idx],
               "tested": self.tested[idx], "mask": mask, "feat": self.d.nb_feat[perts]}
        if with_c:
            out["c"] = self.C[perts[:, None], idx]
        return out

    def null_nb(self, B):
        K = self.d.str_idx.shape[1]
        G = len(self.genes)
        z = torch.zeros(B, K, G, device=self.dev)
        return {"dp": z, "lfc": z, "up": z, "down": z, "tested": z.bool(), "mask": torch.zeros(B, K, dtype=torch.bool, device=self.dev),
                "feat": torch.zeros(B, K, self.d.nb_feat.shape[-1], device=self.dev)}

    def cells_of(self, pools):
        """pools：每个样本的缓存行号池 → 计数 [B, N, G]（float）与文库 [B, N]"""
        idx = torch.stack([pl[torch.randint(0, pl.numel(), (N,), device=self.dev)] for pl in pools])
        return self.counts[idx].float(), self.lib[idx]


def gen_logexpr(counts, lib, q_hat, q_c):
    """模板细胞计数 × 倍数 r（连续）→ log1p(1e4 · 计数 / 文库)。counts [B, n, G]、q_hat [B, G]、q_c [G]"""
    r = q_hat / q_c.clamp_min(1e-12)
    return torch.log1p(1e4 * counts * r[:, None] / lib[..., None])


def pca_feat(L, mu, W, keep):
    """L [B, n, G]（log1p 表达）、keep [B, G] → [B, n, 50]，keep 为假的列（本扰动靶基因）按均值处理"""
    L = torch.where(keep[:, None], L, mu)
    return (L - mu) @ W


def build(args, data, d12, dev):
    if args.fold >= 0:
        s1 = RUNS / f"h1_v7k/n32/fold{args.fold}/K_S1c修正/stage1.pt"
    else:
        s1 = Path(str(RUNS / "h1_v7/n32/split_811/V7_S1c修正_s{seed}/stage1.pt").format(seed=args.seed))
    grid = json.loads((s1.parent / "grid.json").read_text())
    d12.relation = json.loads((s1.parent / "data.json").read_text()).get("relation", "string")
    model = PertFold3(Config3(objective="jepa", **dict(grid, recon_loss="zinb", cont_head=True)), data.esm).to(dev)
    missing, unexpected = model.load_state_dict(torch.load(s1, map_location=dev), strict=False)
    assert not unexpected and all(k.startswith(("decoder.cont.", "decoder.out_adapters.cont.", "decoder.zi.", "log_theta")) for k in missing)
    esm = out_esm(d12.names, dev)
    ci = d12.ctrl_idx
    c = d12.counts[ci].float()
    L = d12.lib[ci][:, None]
    f = c.sum(0) / L.sum()
    mu = L * f
    inv = ((c - mu) ** 2 - mu).sum(0).clamp_min(0) / (mu ** 2).sum(0).clamp_min(1e-8)
    theta0 = (1.0 / inv.clamp_min(1e-4)).clamp(0.01, 1e4)
    net = V12(model, esm, theta0, d12.pc_mu, d12.pc_w, use_null=args.ctrl_recon > 0).to(dev)
    disc = Disc(d12.pc_w.shape[1], model.c.d).to(dev) if args.adv > 0 else None
    return net, disc, s1


def inputs(data, perts, n, rng_cells=True):
    inp, _, _ = data.batch(perts, n, train=rng_cells)
    return inp


@torch.no_grad()
def val_pds(net, data, d12, val, ref, s_dp, draws=2):
    st = (torch.get_rng_state(), torch.cuda.get_rng_state(data.dev))
    torch.manual_seed(12345)
    net.eval()
    T = d12.dB[ref][:, d12.panel]
    Tn = T / T.norm(dim=1, keepdim=True).clamp_min(1e-12)
    pos = {int(p): i for i, p in enumerate(ref.tolist())}
    pds = []
    for _ in range(draws):
        inp = inputs(data, val, N, rng_cells=False)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP):
            _, cont, _, _, _ = net(inp, d12.nb(val, False))
        dh = cont.float() * s_dp
        v = dh[:, d12.panel]
        v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-12)
        dist = 1 - v @ Tn.T                                                     # [V, R]
        for i, p in enumerate(val.tolist()):
            dv = dist[i, pos[p]]
            pds.append(1 - float((dist[i] < dv).sum()) / (len(pos) - 1))
    net.train()
    torch.set_rng_state(st[0]); torch.cuda.set_rng_state(st[1], data.dev)
    return float(np.mean(pds))


def train(args, dev):
    torch.manual_seed(1000 * N + 100_000 * args.seed + 7)
    data = H1Data(dev)
    if args.fold >= 0:                                                          # 5 折：同 train_h1_v5 --inner-val 23
        P = len(data.meta["perts"])
        te = sorted(data.meta["folds"][args.fold])
        trall = sorted(set(range(P)) - set(te))
        va = sorted(np.random.default_rng(args.fold).choice(trall, 23, replace=False).tolist())
        tr = sorted(set(trall) - set(va))
        if args.no_val:                                                         # 10-23：不留内部 val，全部非测试扰动训练，固定步数保存
            tr, va = trall, []
    else:
        sp = json.loads((DATA / "split_811.json").read_text())
        tr, va, te = sp["train"], sp["val"], sp["test"]
    d12 = Data12(data, dev)
    net, disc, s1 = build(args, data, d12, dev)
    scales = d12.set_donors(tr)
    net.m.set_scale(scales["s_dp"])
    s_dp = scales["s_dp"]
    trt, vat = torch.tensor(tr, device=dev), torch.tensor(va, device=dev)
    ref = torch.tensor(sorted(tr + va), device=dev)
    out = OUT / "n32" / (f"fold{args.fold}" if args.fold >= 0 else "split_811") / f"{args.tag}_s{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps({"train": tr, "val": va, "test": te, "donors_train": tr, "donors_test": sorted(tr + va),
                                                   "scales": scales, "stage1": str(s1), "n_out": len(d12.genes), "relation": d12.relation,
                                                   "args": vars(args)}))
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    dopt = torch.optim.Adam(disc.parameters(), lr=2e-4, betas=(0.5, 0.9)) if disc is not None else None
    q_c = torch.expm1(d12.bc) / 5e4
    ema = {k: p.detach().float().clone() for k, p in net.named_parameters() if p.requires_grad} if args.ema > 0 else None   # 10-23：权重滑动平均
    best, t0 = None, time.time()
    vpds = (lambda *a, **k: float("nan")) if args.no_val else val_pds            # 没有 val 时不评估，最后一步的权重留下
    with open(out / "log.jsonl", "w") as log:
        v0 = vpds(net, data, d12, vat, ref, s_dp)
        log.write(json.dumps({"step": 0, "val_pds": v0}) + "\n")
        print(f"{args.tag} s{args.seed}：输出基因 {len(d12.genes)}，s_dp {s_dp:.4f}，初始 val PDS 代理 {v0:.3f}", flush=True)
        for step in range(1, args.s2 + 1):
            for g in opt.param_groups:
                g["lr"] = lr_at(step - 1, args.s2, args.lr)
            perts = trt[torch.randint(0, trt.numel(), (BATCH,), device=dev)]
            inp = inputs(data, perts, N)
            nb = d12.nb(perts, True)
            cnt, lib = d12.cells_of([d12.cidx[int(p)] for p in perts])
            keep = torch.ones(BATCH, len(d12.genes), dtype=torch.bool, device=dev)
            tc = d12.tcol[perts]
            keep[torch.arange(BATCH, device=dev)[tc >= 0], tc[tc >= 0]] = False
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP):
                de_logits, cont, zi, s_nb, ptok = net(inp, nb)
            o = SimpleNamespace(dp_cont=cont.float(), zi_logit=zi.float())
            lab = {"counts": cnt, "lib": lib, "bc": d12.bc, "gmask": keep}
            terms = {"zinb": zinb_nll(o, lab, s_dp, net.log_theta)}
            terms["de"] = args.w_de * de_loss(de_logits, d12.de[perts], d12.tested[perts])
            m = nb["mask"]
            terms["nb"] = F.mse_loss(s_nb.float()[m], nb["c"].float()[m]) if m.any() else cont.sum() * 0
            if args.ctrl_recon > 0:                                              # 无扰动样本：对照重构
                cperts = trt[:N_NULL]
                cinp = inputs(data, cperts, N)
                cinp["relate_x_mask"] = torch.zeros_like(cinp["relate_x_mask"])
                cinp["relate_genes_mask"] = torch.zeros_like(cinp["relate_genes_mask"])
                cinp["x"] = data.bulk(data.ctrl, N_NULL, N)
                ccnt, clib = d12.cells_of([d12.ctrl_idx] * N_NULL)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP):
                    _, ccont, czi, _, _ = net(cinp, d12.null_nb(N_NULL), null=torch.ones(N_NULL, dtype=torch.bool, device=dev))
                co = SimpleNamespace(dp_cont=ccont.float(), zi_logit=czi.float())
                terms["ctrl"] = args.ctrl_recon * zinb_nll(co, {"counts": ccnt, "lib": clib, "bc": d12.bc,
                                                                "gmask": torch.ones_like(keep[:N_NULL])}, s_dp, net.log_theta)
            if args.energy > 0 or args.adv > 0:                                  # 生成细胞（可导）与真实细胞
                tcnt, tlib = d12.cells_of([d12.ctrl_idx] * BATCH)
                q_hat = torch.expm1(F.softplus(d12.bc + cont.float() * s_dp, beta=50.0)) / 5e4
                fg = pca_feat(gen_logexpr(tcnt, tlib, q_hat, q_c), d12.pc_mu, d12.pc_w, keep)
                fr = pca_feat(torch.log1p(1e4 * cnt / lib[..., None]), d12.pc_mu, d12.pc_w, keep)
                if args.energy > 0:
                    terms["energy"] = args.energy * energy_distance(fg, fr)
                if args.adv > 0:
                    cond = ptok.float().mean(1).detach()
                    dl = F.relu(1 - disc(fr, cond)).mean() + F.relu(1 + disc(fg.detach(), cond)).mean()
                    dopt.zero_grad(set_to_none=True); dl.backward(); dopt.step()
                    terms["adv"] = -args.adv * disc(fg, cond).mean()
                    terms["d_loss"] = dl.detach()
            loss = sum(v for k, v in terms.items() if k != "d_loss")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            if ema is not None:
                with torch.no_grad():
                    for k, p in net.named_parameters():
                        if p.requires_grad:
                            ema[k].mul_(args.ema).add_(p.detach().float(), alpha=1 - args.ema)
            if step % args.eval_every == 0 or step == args.s2:
                if ema is not None:                                              # 评估与保存都用 EMA 权重，之后换回
                    cur = {k: p.detach().clone() for k, p in net.named_parameters() if p.requires_grad}
                    with torch.no_grad():
                        for k, p in net.named_parameters():
                            if p.requires_grad:
                                p.copy_(ema[k].to(p.dtype))
                vp = vpds(net, data, d12, vat, ref, s_dp)
                rec = {"step": step, "loss": float(loss), **{k: float(v) for k, v in terms.items()}, "grad_norm": float(gn),
                       "val_pds": vp, "sec": round(time.time() - t0)}
                if best is None or vp > best or (args.no_val and step == args.s2):
                    best = vp
                    torch.save(net.state_dict(), out / "best_pds.pt")
                    rec["best"] = True
                if ema is not None:
                    with torch.no_grad():
                        for k, p in net.named_parameters():
                            if p.requires_grad:
                                p.copy_(cur[k])
                log.write(json.dumps(rec) + "\n"); log.flush()
                print(rec, flush=True)
    return data, d12, net, out


def sampling(data, r, n=N):
    """与 predict_h1_v5.py 相同的随机数序列：对照对半、每个扰动的模板细胞与真值细胞"""
    g = np.random.default_rng(10_000 * n + r)
    ctrl = data.ctrl.cpu().numpy()
    cells = [c.cpu().numpy() for c in data.cells]
    P = len(cells)
    perm = g.permutation(ctrl)
    half = len(perm) // 2
    ctrl_pred, ctrl_real = np.sort(perm[:half]), np.sort(perm[half: 2 * half])
    x_idx = np.stack([g.choice(ctrl_pred, n, replace=False) for _ in range(P)])
    real_idx = np.stack([g.choice(cells[p], n, replace=False) for p in range(P)])
    return x_idx, real_idx, ctrl_pred, ctrl_real


@torch.no_grad()
def predict(args, data, d12, net, out, dev):
    net.load_state_dict(torch.load(out / "best_pds.pt", map_location=dev))
    net.eval()
    man = json.loads((out / "manifest.json").read_text())
    var_n = len(json.loads((DATA / "var_names.json").read_text()))
    cells = [c.cpu().numpy() for c in data.cells]
    jobs = (("val", man["donors_train"], [0]), ("test", man["donors_test"], range(5))) if args.fold < 0 else (("test", man["donors_test"], range(5)),)
    for which, donors, reps in jobs:
        d12.set_donors(donors, man["scales"])                                    # 尺度冻结为训练时的值
        pts = sorted(man[which])
        t = torch.tensor(pts, device=dev)
        for r in reps:
            x_idx, real_idx, ctrl_pred, ctrl_real = sampling(data, r)
            g = np.random.default_rng(777_000 + r)
            inp = {"gene_idx": data.gene_idx, "x": data.X[torch.from_numpy(x_idx[pts]).to(dev)].mean(1),
                   "pert_gene": data.pert_vocab[t], "relate_genes": data.pert_vocab[data.str_idx[t]],
                   "relate_x_mask": data.str_mask[t], "relate_genes_mask": data.str_mask[t]}
            rel = torch.zeros(t.numel(), data.str_idx.shape[1], data.X.shape[1], device=dev)
            for b, j in data.str_mask[t].nonzero().tolist():
                rel[b, j] = data.X[torch.from_numpy(g.choice(cells[int(data.str_idx[t[b], j])], N, replace=False)).to(dev)].mean(0)
            inp["relate_x"] = rel
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP):
                _, cont, _, _, _ = net(inp, d12.nb(t, False))
            full = np.zeros((len(pts), var_n), np.float32)
            full[:, d12.genes] = (cont.float() * man["scales"]["s_dp"]).cpu().numpy()
            fk = f"_fold{args.fold}" if args.fold >= 0 else ""
            np.savez(OUT / "n32" / f"pred_{args.tag}_s{args.seed}{fk}_{which}_rep{r}.npz", perts=np.array(pts), x_idx=x_idx[pts],
                     real_idx=real_idx[pts], ctrl_pred=ctrl_pred, ctrl_real=ctrl_real, dhat_all=full, space="bulk")
    print(f"预测完成：{OUT}/n32/pred_{args.tag}_s{args.seed}_{{val,test}}_rep*.npz", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="Z0")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--s2", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--w-de", type=float, default=0.1)
    ap.add_argument("--ctrl-recon", type=float, default=0.0)
    ap.add_argument("--energy", type=float, default=0.0)
    ap.add_argument("--adv", type=float, default=0.0)
    ap.add_argument("--no-val", action="store_true", help="10-23：5 折模式下不留内部 val，用全部非测试扰动训练，保存最后一步")
    ap.add_argument("--ema", type=float, default=0.0, help="10-23：权重滑动平均的衰减（如 0.99）；评估与保存用 EMA 权重；0 = 不用")
    ap.add_argument("--fold", type=int, default=-1, help="≥ 0：5 折模式的第几折；默认 split_811")
    args = ap.parse_args()
    dev = torch.device(f"cuda:{args.gpu}")
    data, d12, net, out = train(args, dev)
    predict(args, data, d12, net, out, dev)


if __name__ == "__main__":
    main()
