"""v11（2026-10-22 晚，用户要求：证据直通 decoder，与 r4 对比）。

输出：全部表达基因的打分空间 δ̂（log1p(5e4 · 伪 bulk 比例) 之差）。
- 表达基因只用对照细胞定：对照 CPM ≥ --cpm-min，约 12,700 个；
- encoder 输入不变（2,125 个基因），backbone 用 K_S1c修正 的 stage1.pt。

证据（base）：并集邻居打分空间群体 δ 的均匀平均 kNN_g。
- 邻居池同 r4：STRING ≥ 0.7 + K562 Perturb-seq + DepMap，各前 5，最多 10 个；
- 供体只用本次训练扰动（不含内部验证）；预测时从 manifest 恢复同一批供体与尺度。

decoder（model_v5.PassDecoder）：δ̂_g = (1 + a_g)·kNN_g + s·c_g；最后一层零初始化，所以初始 δ̂ = kNN。
- 特征 f_g = [kNN/s, 邻居离散度/s, log1p(邻居数), 有无邻居, 训练平均/s, 对照伪 bulk]。

loss：MSE((δ̂ − δ)/s)。
- 只算输出基因中不是本扰动靶基因的格子；
- s = 训练扰动、输出基因上 δ 的总体标准差。

选 best：每 25 步在内部验证扰动上算同一 MSE。验证抽样用固定种子，并保存 / 恢复训练随机数状态，不影响训练轨迹。

用法：
  训练  python train_h1_v11.py 32 <fold> --from-stage1 '<路径，{fold} 换折号>' --tag 名字 [--s2 1000 --lr 1e-4]
  预测  python train_h1_v11.py 32 --predict --tag 名字 [--reps 5]
        → n32/pred_<tag>_full_rep<r>.npz、pred_<tag>_base_rep<r>.npz（dhat_all，供 eval_bulk_h1.py）
        抽样（x_idx / real_idx / 对照对半）与 K_S1c修正 的预测文件第 r 次重复相同。
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "/cluster2/home/hanwen/vcc_26/pertfold_r4")
from pertfold3.model_v5 import Config3, PassDecoder, PertFold3, standardize_esm  # noqa: E402
from train_h1_v5 import DATA, ESM, H1Data, lr_at  # noqa: E402

OUT = Path("/cluster2/home/hanwen/vcc_26/pertfold_r4/runs/h1_v11")
SAMPLING = "/cluster2/home/hanwen/vcc_26/pertfold/runs3/h1_v7k/n32/pred_K_S1c修正_stage2_ft_lr0.0001_best_pearson_exp_exp_rep{r}.npz"
BATCH, AMP = 32, True


def output_genes(cpm_min):
    """对照 CPM ≥ cpm_min 的基因（全部 18,080 个基因中的列号）、名字、全部基因打分空间 δ 与对照伪 bulk"""
    z = np.load(DATA / "bulk_sums_all.npz")
    cpm = z["S"][-1] / z["tot"][-1] * 1e6
    var = json.loads((DATA / "var_names.json").read_text())
    idx = np.where(cpm >= cpm_min)[0]
    return idx, [var[i] for i in idx], var, z["dB"][:, idx].astype(np.float32), z["B"][-1, idx].astype(np.float32)


def out_esm(names, dev):
    cache = torch.load(ESM, map_location="cpu", weights_only=False)
    pos = {str(g).upper(): i for i, g in enumerate(cache["genes"])}
    emb, found = cache["embeddings"].float(), cache["found_mask"].bool()
    e = torch.zeros(len(names), emb.shape[1])
    for i, g in enumerate(names):
        j = pos.get(g.upper())
        if j is not None and found[j]:
            e[i] = emb[j]
    return standardize_esm(e).to(dev)


def evidence(data, Y, donors, dev):
    """data.set_evidence(donors) 之后：并集邻居的均匀平均与离散度、邻居数、训练平均、尺度 s"""
    Yt = torch.from_numpy(Y).to(dev)
    idx, ok = data.str_idx, data.str_mask.float()                                # [P, K]
    n = ok.sum(1)
    w = ok / n.clamp_min(1)[:, None]
    nb = Yt[idx]                                                                 # [P, K, G]
    mean = (w[..., None] * nb).sum(1)
    var = (w[..., None] * (nb - mean[:, None]) ** 2).sum(1)
    sd = torch.where(n[:, None] >= 2, var.clamp_min(0).sqrt(), torch.zeros_like(mean))
    d = torch.tensor(donors, device=dev)
    m = Yt[d].mean(0)
    s = float(Yt[d].std())
    return Yt, mean, sd, n, m, s


def features(mean, sd, n, m, bc, s):
    P, G = mean.shape
    one = torch.ones(P, G, device=mean.device)
    return torch.stack([mean / s, sd / s, torch.log1p(n)[:, None] * one, (n > 0).float()[:, None] * one,
                        (m / s)[None] * one, bc[None] * one], -1)


def target_mask(perts_names, names, P, dev):
    pos = {g: i for i, g in enumerate(names)}
    mask = torch.ones(P, len(names), dtype=torch.bool, device=dev)
    for p, g in enumerate(perts_names):
        if g in pos:
            mask[p, pos[g]] = False
    return mask


def build(grid, data, dev, n_feat):
    model = PertFold3(Config3(objective="jepa", **grid), data.esm).to(dev)
    dec = PassDecoder(model.c, n_feat).to(dev)
    return model, dec


@torch.no_grad()
def val_mse(model, dec, data, ev, n, esm, feat, base, Y, mask, s):
    """固定种子抽样，算完恢复训练随机数状态"""
    st = (torch.get_rng_state(), torch.cuda.get_rng_state(data.dev))
    torch.manual_seed(12345)
    model.eval(); dec.eval()
    mse, r = [], []
    for _ in range(2):
        inp, _, _ = data.batch(ev, n)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP):
            dh = dec(model.encode_z(**inp), esm, feat[ev], base[ev], s)
        e = ((dh - Y[ev]) / s) ** 2
        mse.append(e[mask[ev]].mean().item())
        w = mask[ev].float()
        a = (dh - (dh * w).sum(1, keepdim=True) / w.sum(1, keepdim=True)) * w
        b = (Y[ev] - (Y[ev] * w).sum(1, keepdim=True) / w.sum(1, keepdim=True)) * w
        r.append(((a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-8)).mean().item())
    model.train(); dec.train()
    torch.set_rng_state(st[0]); torch.cuda.set_rng_state(st[1], data.dev)
    return sum(mse) / len(mse), sum(r) / len(r)


def train(args, dev):
    torch.manual_seed(1000 * args.n + args.fold + 100_000 * args.seed)
    data = H1Data(dev)
    P = len(data.meta["perts"])
    folds = data.meta["folds"]
    train_ids = sorted(set(range(P)) - set(folds[args.fold]))
    g = np.random.default_rng(args.fold)                                         # 同 train_h1_v5 --inner-val
    val = sorted(g.choice(train_ids, args.inner_val, replace=False).tolist())
    donors = sorted(set(train_ids) - set(val))
    s1 = Path(args.from_stage1.format(fold=args.fold))
    grid = json.loads((s1.parent / "grid.json").read_text())
    relation = json.loads((s1.parent / "data.json").read_text()).get("relation", "string")
    tr, ev = torch.tensor(donors, device=dev), torch.tensor(val, device=dev)
    data.set_evidence(tr, relation)                                              # 邻居池只含供体
    idx, names, var, Y, bc = output_genes(args.cpm_min)
    Yt, mean, sd, n, m, s = evidence(data, Y, donors, dev)
    bct = torch.from_numpy(bc).to(dev)
    feat = features(mean, sd, n, m, bct, s)
    mask = target_mask(data.meta["perts"], names, P, dev)
    esm = out_esm(names, dev)
    model, dec = build(grid, data, dev, feat.shape[-1])
    model.load_state_dict(torch.load(s1, map_location=dev))
    model.requires_grad_(True)
    if model.target_patch is not None:
        model.target_patch.requires_grad_(False)
    out = Path(args.out) / f"n{args.n}" / f"fold{args.fold}" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps({
        "fold": args.fold, "train": train_ids, "val": val, "donors": donors, "test": folds[args.fold], "s": s,
        "cpm_min": args.cpm_min, "n_out": len(idx), "relation": relation, "stage1": str(s1), "args": vars(args)}))
    params = [p for p in list(model.parameters()) + list(dec.parameters()) if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    best = None
    t0 = time.time()
    with open(out / "log.jsonl", "w") as log:
        v0 = val_mse(model, dec, data, ev, args.n, esm, feat, mean, Yt, mask, s)
        log.write(json.dumps({"step": 0, "val_mse": v0[0], "val_pearson": v0[1], "n_out": len(idx), "s": s}) + "\n")
        print(f"fold {args.fold}：输出基因 {len(idx)}，供体 {len(donors)}，val {len(val)}，s={s:.4f}；初始（= kNN）val_mse {v0[0]:.4f} r {v0[1]:.3f}", flush=True)
        for step in range(1, args.s2 + 1):
            for gr in opt.param_groups:
                gr["lr"] = lr_at(step - 1, args.s2, args.lr)
            perts = tr[torch.randint(0, tr.numel(), (BATCH,), device=dev)]
            inp, _, _ = data.batch(perts, args.n, train=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP):
                dh = dec(model.encode_z(**inp), esm, feat[perts], mean[perts], s)
            loss = (((dh - Yt[perts]) / s) ** 2)[mask[perts]].mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            if step % args.eval_every == 0 or step == args.s2:
                vm, vr = val_mse(model, dec, data, ev, args.n, esm, feat, mean, Yt, mask, s)
                rec = {"step": step, "loss": loss.item(), "val_mse": vm, "val_pearson": vr, "sec": round(time.time() - t0)}
                if best is None or vm < best:
                    best = vm
                    torch.save({"model": model.state_dict(), "dec": dec.state_dict()}, out / "best_val_mse.pt")
                    rec["best"] = True
                log.write(json.dumps(rec) + "\n"); log.flush()
                print(rec, flush=True)


@torch.no_grad()
def predict(args, dev):
    data = H1Data(dev)
    P, n = len(data.meta["perts"]), args.n
    cells = [c.cpu().numpy() for c in data.cells]
    folds = data.meta["folds"]
    models = []
    for k in range(len(folds)):
        d = Path(args.out) / f"n{n}" / f"fold{k}" / args.tag
        man = json.loads((d / "manifest.json").read_text())
        s1 = Path(man["stage1"])
        grid = json.loads((s1.parent / "grid.json").read_text())
        data.set_evidence(torch.tensor(man["donors"], device=dev), man["relation"])   # 训练时的供体
        idx, names, var, Y, bc = output_genes(man["cpm_min"])
        Yt, mean, sd, nn_, m, s = evidence(data, Y, man["donors"], dev)
        assert abs(s - man["s"]) < 1e-6 * max(1.0, abs(s)), "尺度与训练不一致"
        feat = features(mean, sd, nn_, m, torch.from_numpy(bc).to(dev), s)
        model, dec = build(grid, data, dev, feat.shape[-1])
        ck = torch.load(d / "best_val_mse.pt", map_location=dev)
        model.load_state_dict(ck["model"]); dec.load_state_dict(ck["dec"])
        model.eval(); dec.eval()
        models.append((model, dec, data.str_idx.clone(), data.str_mask.clone(), feat, mean, s, man["test"]))
    esm = out_esm(names, dev)
    OUTN = Path(args.out) / f"n{n}"
    for r in range(args.reps):
        z = np.load(SAMPLING.format(r=r))
        perts = z["perts"]
        row = {int(p): i for i, p in enumerate(perts)}
        full = np.zeros((len(perts), len(var)), np.float32)
        base = np.zeros_like(full)
        g = np.random.default_rng(777_000 + 10_000 * n + r)                       # 邻居细胞行的抽样
        for model, dec, sidx, smask, feat, mean, s, te in models:
            t = torch.tensor(te, device=dev)
            x = data.X[torch.from_numpy(z["x_idx"][[row[p] for p in te]]).to(dev)].mean(1)
            rel = torch.zeros(t.numel(), sidx.shape[1], data.X.shape[1], device=dev)
            for b, j in smask[t].nonzero().tolist():
                rel[b, j] = data.X[torch.from_numpy(g.choice(cells[int(sidx[t[b], j])], n, replace=False)).to(dev)].mean(0)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP):
                zz = model.encode_z(gene_idx=data.gene_idx, x=x, relate_x=rel, pert_gene=data.pert_vocab[t],
                                    relate_genes=data.pert_vocab[sidx[t]], relate_x_mask=smask[t], relate_genes_mask=smask[t])
                dh = dec(zz, esm, feat[t], mean[t], s)
            ri = [row[p] for p in te]
            full[np.ix_(ri, idx)] = dh.float().cpu().numpy()
            base[np.ix_(ri, idx)] = mean[t].cpu().numpy()
        keep = {k_: z[k_] for k_ in ("perts", "x_idx", "real_idx", "ctrl_pred", "ctrl_real")}
        for nm, arr in (("full", full), ("base", base)):
            np.savez(OUTN / f"pred_{args.tag}_{nm}_rep{r}.npz", **keep, dhat_all=arr, space="bulk")
        print(f"rep {r} → {OUTN}/pred_{args.tag}_{{full,base}}_rep{r}.npz", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int)
    ap.add_argument("fold", type=int, nargs="?", default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--from-stage1", default="/cluster2/home/hanwen/vcc_26/pertfold/runs3/h1_v7k/n32/fold{fold}/K_S1c修正/stage1.pt")
    ap.add_argument("--s2", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--inner-val", type=int, default=23)
    ap.add_argument("--cpm-min", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="pass")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--predict", action="store_true")
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    dev = torch.device(f"cuda:{args.gpu}")
    predict(args, dev) if args.predict else train(args, dev)


if __name__ == "__main__":
    main()
