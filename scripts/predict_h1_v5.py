"""v10（2026-10-22 晚）：训练时 --space bulk 的模型（data.json 的 space）按打分空间建证据；npz 另存 dhat（读出的 δ̂，打分空间）
与 space 字段，供 eval_bulk_h1.py 打分（yhat = xbar + δ̂ 在打分空间没有意义，只为兼容保留）。打分空间模型用 --readout cont（连续头）或 nbw。
v5 模型（pertfold3/model_v5.py）的预测：同 predict_h1_v3.py，另给 N1 的邻居输入，并多两种读出：
  --readout cont   N2 连续头（δ = dp_cont · s_dp）
  --readout nbw    只用 N1 学到的权重 a_k 对邻居真实 delta_p 加权平均（不经过 decoder，"学习权重的 kNN"）
以下为 predict_h1_v3.py 说明。
v3 模型（pertfold3/model_v3.py，decoder 含 PIE 证据）的预测：与 predict_h1.py 相同的抽样，只是模型用 model_v3、
数据用 train_h1_v3.H1Data，并按该模型的训练扰动与关联来源（data.json 的 relation，没有则 string）重建证据传给模型。

以下为 predict_h1.py 原说明。
用 5 折模型为全部 147 个扰动生成预测（vcc26 环境）。每个扰动由"没见过它"的那一折模型预测。

每次重复 r：
  对照细胞随机对半：ctrl_pred（预测侧）与 ctrl_real（真值侧），与 benchmark 的对半方式一致
  每个扰动 p：
    x_idx     从 ctrl_pred 抽 n 个细胞，其平均作为模型输入 x；这 n 个细胞也是预测细胞的模板
    relate_x  该折 STRING 邻居扰动各 n 个真实细胞的平均（没有邻居则屏蔽）；该折 data.json 中 relate_diff 为真时减去全部对照均值；
              data.json 中有 relation 时改用该关联来源的邻居（train_h1.py --relation）
    real_idx  从 p 的细胞中抽 n 个，作为真值
    yhat      模型输出 y_hat = x + delta（log1p CP10k，模型基因）
    de_prob   DE 头的 P(DE)
    dp_prob   delta 分箱的 softmax 概率 [扰动, 基因, K]（float16；5 折且各折 K 不同时不保存）

--split <划分文件>：只用一个模型（<runs>/n<N>/<划分文件名>/<tag>），只预测该划分的 test 扰动；npz 只含这些扰动（perts 字段）。
抽样仍对全部扰动进行、再取子集，所以与 5 折版本同一扰动的抽样完全相同。

用法：python predict_h1.py <n> <ckpt 文件名> [--tag 名字] [--reps 5] [--gpu 0] [--runs runs3/h1] [--split 划分文件]
输出：<runs>/n<N>/pred_<tag>_<ckpt>_rep<r>.npz
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "/cluster2/home/hanwen/vcc_26/pertfold_r4")
from pertfold3.model_v5 import Config3, PertFold3, dp_centers  # noqa: E402
from train_h1_v5 import RUNS, H1Data  # noqa: E402


def sample(pool, k, n, g):
    """从 pool（numpy 索引）为 k 组各不放回抽 n 个 → [k, n]"""
    return np.stack([g.choice(pool, n, replace=False) for _ in range(k)])


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int)
    ap.add_argument("ckpt", help="每折目录下的权重文件名，如 stage2_ft_lr0.0001_step500.pt")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--runs", default=str(RUNS))
    ap.add_argument("--tag", default="", help="与 train_h1.py 的 --tag 相同")
    ap.add_argument("--split", default="", help="划分文件：只预测 test 扰动（单个模型）")
    ap.add_argument("--eval-set", default="test", choices=["test", "val"], help="--split 时预测哪一组扰动（val 的输出名加 _val）")
    ap.add_argument("--readout", choices=["argmax", "exp", "cont", "nbw"], default="argmax",
                    help="delta 读出：argmax 箱中心（PIE）或各箱中心按 softmax 概率的期望")
    args = ap.parse_args()
    runs = Path(args.runs)
    dev = torch.device(f"cuda:{args.gpu}")
    data = H1Data(dev)
    folds, P, n = data.meta["folds"], len(data.meta["perts"]), args.n
    ctrl = data.ctrl.cpu().numpy()
    cells = [c.cpu().numpy() for c in data.cells]
    if args.split:
        sp = json.loads(Path(args.split).read_text())
        groups = [(runs / f"n{n}" / Path(args.split).stem / args.tag, sp["train"], sp[args.eval_set])]
    else:
        groups = [(runs / f"n{n}" / f"fold{k}" / args.tag, sorted(set(range(P)) - set(folds[k])), folds[k])
                  for k in range(len(folds))]
    keep = np.array(sorted(p for _, _, te in groups for p in te))                 # 输出的扰动
    models = []
    for d, tr, te in groups:
        sd = torch.load(d / args.ckpt, map_location=dev)
        obj = "jepa" if any(key.startswith("target_patch.") for key in sd) else "recon"   # 不做 jepa 的模型没有 target_patch
        m = PertFold3(Config3(objective=obj, **json.loads((d / "grid.json").read_text())), data.esm).to(dev).eval()
        m.load_state_dict(sd)
        dj = d / "data.json"
        dset = json.loads(dj.read_text()) if dj.exists() else {}
        data.ev_mode = dset.get("ev_mode", "delta")                                # B2：证据空间与训练相同
        if data.ev_mode == "fc" and not hasattr(data, "mu_counts"):
            data.build_aug_tables()
        if dset.get("space", "dp") == "bulk" and data.space != "bulk":            # v10：打分空间的标签与证据
            data.use_bulk()
        if dset.get("space", "dp") != data.space:
            raise SystemExit("各折模型的空间不一致")
        data.set_evidence(torch.tensor(tr, device=dev), dset.get("relation", "string"))   # 本模型训练扰动中的邻居与证据
        ev = data.evidence if dset.get("use_evidence", True) else None
        nbd = data.nb_inputs(torch.tensor(te, device=dev), with_c=False) if m.c.nb_learn else None   # 只依赖训练扰动的标签
        nbd = None if nbd is None else (None, nbd)
        models.append((m, data.str_idx.clone(), data.str_mask.clone(), (ev, nbd), te))
        amp = not dset.get("fp32", False)                                          # T3：训练时不用 autocast，预测也不用

    for r in range(args.reps):
        g = np.random.default_rng(10_000 * n + r)
        perm = g.permutation(ctrl)
        ctrl_pred, ctrl_real = np.sort(perm[: len(perm) // 2]), np.sort(perm[len(perm) // 2: 2 * (len(perm) // 2)])
        x_idx = sample(ctrl_pred, P, n, g)
        real_idx = np.stack([g.choice(cells[p], n, replace=False) for p in range(P)])
        xbar = data.X[torch.from_numpy(x_idx).to(dev)].mean(1)                     # [P, G]
        yhat = torch.empty_like(xbar)
        dhat = torch.zeros_like(xbar)                                              # v10：读出的 δ̂（模型的空间）
        de_prob = torch.zeros_like(xbar)                                           # DE 头：P(DE)
        dp_prob = None                                                             # delta 分箱的 softmax 概率 [P, G, K]
        for m, sidx, smask, (ev, nbd), te in models:
            t = torch.tensor(te, device=dev)
            # 相关行：本折 STRING 邻居扰动各 n 个真实细胞的平均（与训练相同）
            rel = torch.zeros(t.numel(), sidx.shape[1], data.X.shape[1], device=dev)
            for b, j in smask[t].nonzero().tolist():
                pool = cells[int(sidx[t[b], j])]
                rel[b, j] = data.X[torch.from_numpy(g.choice(pool, n, replace=False)).to(dev)].mean(0)
            evid = None
            if ev is not None:
                evid = {k: v[t] for k, v in ev.items()}
                evid["ctrl"] = data.ctrl_mean.expand(t.numel(), -1)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                o = m(gene_idx=data.gene_idx, x=xbar[t], relate_x=rel, relate_x_mask=smask[t],
                      pert_gene=data.pert_vocab[t], relate_genes=data.pert_vocab[sidx[t]],
                      relate_genes_mask=smask[t], evidence=evid,
                      nb=None if nbd is None else nbd[1])                    # nbd 按同一个 te 顺序构建
            if args.readout == "exp":
                d = o.delta.float() if o.dp_logits is None else o.dp_logits.float().softmax(-1) @ dp_centers(m.c, dev)   # v7 affine
                if o.dp_base is not None and o.dp_logits is not None:              # v8 nb_resid：δ̂ = nbw + E[残差]
                    d = d + o.dp_base.float()
            elif args.readout == "cont":
                d = o.dp_cont.float() * m.s_dp
            elif args.readout == "nbw":
                d = o.nbw_dp.float() * m.s_dp
                if data.ev_mode == "fc":                                            # fold change → δ：以对照水平为起点
                    c = torch.expm1(data.ctrl_mean)
                    d = torch.log1p(c * torch.exp2(d)) - torch.log1p(c)
            else:
                d = o.delta.float()
            yhat[t] = xbar[t] + d                                                   # y_hat = x + delta（fp32 相加）
            dhat[t] = d
            de_prob[t] = o.de_logits.float().softmax(-1)[..., 1]
            if o.dp_logits is None:                                                # v7 affine：没有分箱概率
                dp_prob = False
            if dp_prob is None:
                dp_prob = torch.zeros(*xbar.shape, o.dp_logits.shape[-1], device=dev)
            if dp_prob is not False and dp_prob.shape[-1] == o.dp_logits.shape[-1]:
                dp_prob[t] = o.dp_logits.float().softmax(-1)
            else:
                dp_prob = False                                                     # 各折分箱数不同（5 折时）：不保存
        suffix = ("" if args.readout == "argmax" else f"_{args.readout}") + ("_val" if args.eval_set == "val" else "")
        out = runs / f"n{n}" / f"pred_{args.tag or 'model'}_{Path(args.ckpt).stem}{suffix}_rep{r}.npz"
        extra = {"dp_prob": dp_prob[keep].half().cpu().numpy()} if dp_prob is not False else {}
        np.savez(out, perts=keep, x_idx=x_idx[keep], real_idx=real_idx[keep], ctrl_pred=ctrl_pred, ctrl_real=ctrl_real,
                 xbar=xbar[keep].cpu().numpy(), yhat=yhat[keep].cpu().numpy(), de_prob=de_prob[keep].cpu().numpy(),
                 dhat=dhat[keep].cpu().numpy(), space=data.space, **extra)
        print(f"n={n} rep {r} → {out}", flush=True)


if __name__ == "__main__":
    main()
