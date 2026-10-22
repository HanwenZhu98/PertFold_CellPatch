#!/bin/bash
# 用本目录的代码对原 R4 权重（种子 0）重新预测第 0 次重复，与 2026-10-20 当时的预测逐元素比对（应完全相同）。
set -eu
cd /cluster2/home/hanwen/vcc_26/pertfold_r4/scripts
PY=/cluster2/home/hanwen/miniconda3/envs/vcc26/bin/python
SPLIT=/cluster2/home/hanwen/vcc_26/pertfold/data/h1/split_811.json
CUDA_VISIBLE_DEVICES=${GPU:-0} $PY -u predict_h1_v5.py 32 stage2_ft_lr0.0001_best_pearson_exp.pt --split $SPLIT --tag R4_orig_s0 \
    --reps 1 --gpu 0 --runs ../runs --readout exp > ../runs/verify_predict.log 2>&1
$PY - <<'PYEOF'
import numpy as np
a = np.load("/cluster2/home/hanwen/vcc_26/pertfold_r4/checkpoints/R4_s0/ref_pred_exp_rep0.npz")
b = np.load("/cluster2/home/hanwen/vcc_26/pertfold_r4/runs/n32/pred_R4_orig_s0_stage2_ft_lr0.0001_best_pearson_exp_exp_rep0.npz")
same = all(np.array_equal(a[k], b[k]) for k in ("perts", "x_idx", "real_idx"))
d = float(np.abs(a["yhat"] - b["yhat"]).max()); e = float(np.abs(a["de_prob"] - b["de_prob"]).max())
print(f"抽样一致 {same}；yhat 最大差 {d:.2e}；de_prob 最大差 {e:.2e} → {'通过：与原 R4 逐位相同' if same and d == 0 and e == 0 else '不一致！'}")
PYEOF
