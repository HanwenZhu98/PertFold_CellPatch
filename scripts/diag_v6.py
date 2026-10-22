"""v6 防塌缩检查（2026-10-20）：阶段 1 结束（stage1.pt）与阶段 2 最好权重各算一遍，fp32，种子固定。
encoder（CellPatch，MSA 之前）对"扰动"是否可分 —— 用 val + test 的 29 个扰动（从未作为 IDM 的类别出现），每个扰动 8 次 bulk 抽样（n = 32）：
  扰动检索 top-1        每次抽样的池化 embedding 与各扰动"其余 7 次抽样的质心"比余弦，最近的是否为自己（随机 1/29）
  η²（扰动）            池化 embedding 中扰动之间方差占总方差的比例
  样本间余弦 / RankMe   CellPatch 输出（展平）
  另给同一检索在输入层（标准化后的 bulk 表达）上的值作参照：encoder 最多能保留多少
  "屏蔽靶基因"版本：y 的 147 个靶基因列换成对照均值后再算（--mask-targets 训练的 encoder 只见过这种 y）
对照 x（32 个 bulk）：样本间余弦、RankMe
阶段 1 目标自己的指标（只对 stage1.pt）：
  jepa  预测 / 猜平均：loss(批内平均目标) / loss(预测)，> 1 表示预测好于猜平均；目标与训练相同（ema：LN(EMA 编码)，L1；sg：在线编码，MSE）
  IDM   训练扰动上的分类准确率（新抽样，随机 1/118）
  MAE   被遮基因的 R²（val + test 扰动 bulk 与对照 bulk；相对 mae_mu）
用法：python diag_v6.py <run 目录> ... → 每个目录下 diag_v6.json，并打印汇总表
"""
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "/cluster2/home/hanwen/vcc_26/pertfold_r4")
from pertfold3.model_v5 import Config3, PertFold3, sigreg  # noqa: E402,F401
from train_h1_v5 import H1Data, rank_me  # noqa: E402

CK = "stage2_ft_lr0.0001_best_pearson_exp.pt"
N, DRAWS = 32, 8
dev = torch.device("cuda:0")
data = H1Data(dev)
sp = json.loads(Path("/cluster2/home/hanwen/vcc_26/pertfold/data/h1/split_811.json").read_text())
train = torch.tensor(sp["train"], device=dev)
held = sorted(sp["val"] + sp["test"])
tg = data.labels["target"]
tcols = tg[tg >= 0].unique()
ctrl_mu = data.X[data.ctrl].mean(0)


def pair_cos(M):
    M = F.normalize(M, dim=1)
    S = M @ M.T
    return float(S[~torch.eye(len(M), dtype=torch.bool, device=M.device)].mean())


def retrieval(E, lab):
    """E [P·D, f]（按扰动分组，每组 D 行），lab [P·D] → (top-1, η²)"""
    P = int(lab.max()) + 1
    E = E.double()
    acc = 0
    for i in range(len(E)):
        keep = torch.ones(len(E), dtype=torch.bool, device=E.device)
        keep[i] = False
        cent = torch.stack([E[keep & (lab == p)].mean(0) for p in range(P)])
        acc += int(F.cosine_similarity(E[i][None], cent).argmax() == lab[i])
    mu = E.mean(0)
    cm = torch.stack([E[lab == p].mean(0) for p in range(P)])
    between = ((cm - mu) ** 2).sum() * (len(E) / P)
    return acc / len(E), float(between / ((E - mu) ** 2).sum())


def load(run, ck):
    grid = json.loads((run / "grid.json").read_text())
    sd = torch.load(run / ck, map_location=dev)
    ema = any(k.startswith("target_patch.") for k in sd)
    obj = "mae" if ck == "stage1.pt" and grid.get("mae") else ("jepa" if (ema or ck == "stage1.pt") else "recon")
    m = PertFold3(Config3(objective=obj, **grid), data.esm).to(dev)
    m.load_state_dict(sd)
    return m.eval()


@torch.no_grad()
def encoder_stats(m, ys, yms, lab, xs):
    enc = lambda X: m.encode_cell(data.gene_idx, X)  # noqa: E731
    r = {}
    for name, Y in (("y", ys), ("y 屏蔽靶基因", yms)):
        Z = enc(Y)
        top1, eta = retrieval(Z.mean(1), lab)
        r[f"扰动检索 top-1（{name}）"], r[f"η² 扰动（{name}）"] = top1, eta
        if name == "y":
            r["样本间余弦（y）"], r["RankMe（y）"] = pair_cos(Z.flatten(1)), rank_me(Z.flatten(1))
    Zx = enc(xs).flatten(1)
    r["样本间余弦（x）"], r["RankMe（x）"] = pair_cos(Zx), rank_me(Zx)
    return r


@torch.no_grad()
def stage1_stats(m, run):
    c, r = m.c, {}
    torch.manual_seed(1)
    dset = json.loads((run / "data.json").read_text())
    if c.objective == "mae":
        pert_b = torch.cat([data.bulk(data.cells[p], 4, N) for p in held])
        ctrl_b = data.bulk(data.ctrl, 64, N)
        for name, B in (("扰动 bulk", pert_b), ("对照 bulk", ctrl_b)):
            Gn = B.shape[1]
            k = int(round(c.mae_ratio * Gn))
            pad = torch.zeros_like(B, dtype=torch.bool)
            pad.scatter_(1, torch.rand(B.shape, device=dev).argsort(1)[:, :k], True)
            ge = m.gene_table(data.gene_idx)
            rec = m.mae_head(m.cell_patch(B, ge, pad=pad), ge)
            tgt = (B - m.mae_mu) / m.mae_sd
            r[f"MAE R²（{name}）"] = float(1 - ((rec - tgt)[pad] ** 2).sum() / (tgt[pad] ** 2).sum())
        return r
    data.set_evidence(train, dset.get("relation", "string"))
    data.use_evidence, data.use_nb = True, m.c.nb_learn
    num = den = acc = 0.0
    for _ in range(8):
        perts = train[torch.randint(0, train.numel(), (32,), device=dev)]
        inp, y, lab = data.batch(perts, N, train=True)
        if dset.get("mask_targets"):
            y = y.clone()
            y[:, tcols] = ctrl_mu[tcols]
        o = m(**inp)
        if c.jepa_mode == "ema":
            t = F.layer_norm(m.encode_cell(data.gene_idx, y, target=True), (c.d,))
            dist = lambda a, b: F.l1_loss(a, b)  # noqa: E731
        else:
            t = m.encode_cell(data.gene_idx, y)
            dist = lambda a, b: F.mse_loss(a, b)  # noqa: E731
        num += float(dist(t.mean(0, keepdim=True).expand_as(t), t))
        den += float(dist(o.cell_patch_y_hat, t))
        if c.idm_weight > 0:
            logits = m.idm(torch.cat([o.cell_patch0.mean(1), m.encode_cell(data.gene_idx, y).mean(1)], -1))
            acc += float((logits.argmax(-1) == perts).float().mean())
    r["jepa 猜平均 / 预测"] = num / den
    if c.idm_weight > 0:
        r["IDM 准确率（训练扰动）"] = acc / 8
    return r


torch.manual_seed(0)
ys = torch.cat([data.bulk(data.cells[p], DRAWS, N) for p in held])
lab = torch.arange(len(held), device=dev).repeat_interleave(DRAWS)
yms = ys.clone()
yms[:, tcols] = ctrl_mu[tcols]
xs = data.bulk(data.ctrl, 32, N)
sdx = ys.std(0).clamp_min(1e-2)
ref = {"输入层 扰动检索 top-1（y）": retrieval((ys - ys.mean(0)) / sdx, lab)[0],
       "输入层 扰动检索 top-1（y 屏蔽靶基因）": retrieval((yms - yms.mean(0)) / sdx, lab)[0]}
print(json.dumps(ref, ensure_ascii=False), flush=True)
rows = []
for run in map(Path, sys.argv[1:]):
    res = {"参照": ref}
    for ck, name in (("stage1.pt", "阶段 1 结束"), (CK, "阶段 2 最好")):
        if not (run / ck).exists():
            continue
        m = load(run, ck)
        r = encoder_stats(m, ys, yms, lab, xs)
        if ck == "stage1.pt":
            r.update(stage1_stats(m, run))
        res[name] = r
        rows.append((f"{run.name}｜{name}", r))
        del m
        torch.cuda.empty_cache()
    (run / "diag_v6.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
cols = ["扰动检索 top-1（y）", "扰动检索 top-1（y 屏蔽靶基因）", "η² 扰动（y）", "样本间余弦（y）", "RankMe（y）", "样本间余弦（x）",
        "RankMe（x）", "jepa 猜平均 / 预测", "IDM 准确率（训练扰动）", "MAE R²（扰动 bulk）", "MAE R²（对照 bulk）"]
print("run".ljust(40) + "".join(f"{c[:9]:>11s}" for c in cols))
for name, r in rows:
    print(name[:40].ljust(40) + "".join(f"{r.get(c, float('nan')):11.4g}" for c in cols))
