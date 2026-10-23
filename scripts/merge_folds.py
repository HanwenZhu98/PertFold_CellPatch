"""把 train_h1_v12.py --fold 0..4 的测试预测合并成 147 个扰动的一个文件（每次重复一个），供 eval_bulk_h1.py 与 summarize_bench.py。
用法：python merge_folds.py <tag_s种子，如 Z0_s0> → runs3/h1_v12/n32/pred_<tag_s种子>_k_rep<r>.npz"""
import sys
from pathlib import Path
import numpy as np
V = Path("/cluster2/home/hanwen/vcc_26/pertfold/runs3/h1_v12/n32")
ts = sys.argv[1]
for r in range(5):
    parts = [np.load(V / f"pred_{ts}_fold{k}_test_rep{r}.npz") for k in range(5)]
    perts = np.concatenate([p["perts"] for p in parts]); o = np.argsort(perts)
    np.savez(V / f"pred_{ts}_k_rep{r}.npz", perts=perts[o], x_idx=np.concatenate([p["x_idx"] for p in parts])[o],
             real_idx=np.concatenate([p["real_idx"] for p in parts])[o], ctrl_pred=parts[0]["ctrl_pred"], ctrl_real=parts[0]["ctrl_real"],
             dhat_all=np.concatenate([p["dhat_all"] for p in parts])[o], space="bulk")
    assert all(np.array_equal(p["ctrl_pred"], parts[0]["ctrl_pred"]) for p in parts)
print("ok", ts)
