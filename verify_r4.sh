#!/bin/bash
# 用本目录的代码对已有 R4 权重（种子 0）重新预测第 0 次重复，与当时的预测逐元素比对（应完全相同）。
# 用法：GPU=0 ./verify_r4.sh [v2|v1]（默认 v2）
#   v2 = 现在的 R4（证据只留 nbw）：权重 checkpoints/R4v2_s0 ← pertfold/runs3/h1_v7/.../V7_S1c修正_s0（2026-10-21）
#   v1 = 原 R4（证据含对照均值、ctx、str、nbw）：权重 checkpoints/R4_s0 ← pertfold/runs3/h1_v6/.../V6_R4_jepa_IDM_SIG_s0（2026-10-20）
set -eu
VER=${1:-v2}
case $VER in v2) CKD=R4v2_s0; TAG=R4v2_orig_s0 ;; v1) CKD=R4_s0; TAG=R4_orig_s0 ;; *) echo "版本只能是 v1 或 v2"; exit 1 ;; esac
cd /cluster2/home/hanwen/vcc_26/pertfold_r4/scripts
PY=/cluster2/home/hanwen/miniconda3/envs/vcc26/bin/python
SPLIT=/cluster2/home/hanwen/vcc_26/pertfold/data/h1/split_811.json
CUDA_VISIBLE_DEVICES=${GPU:-0} $PY -u predict_h1_v5.py 32 stage2_ft_lr0.0001_best_pearson_exp.pt --split $SPLIT --tag $TAG \
    --reps 1 --gpu 0 --runs ../runs --readout exp > ../runs/verify_predict_$VER.log 2>&1
CKD=$CKD TAG=$TAG VER=$VER $PY - <<'PYEOF'
import os
import numpy as np
r = "/cluster2/home/hanwen/vcc_26/pertfold_r4"
a = np.load(f"{r}/checkpoints/{os.environ['CKD']}/ref_pred_exp_rep0.npz")
b = np.load(f"{r}/runs/n32/pred_{os.environ['TAG']}_stage2_ft_lr0.0001_best_pearson_exp_exp_rep0.npz")
same = all(np.array_equal(a[k], b[k]) for k in ("perts", "x_idx", "real_idx"))
d = float(np.abs(a["yhat"] - b["yhat"]).max()); e = float(np.abs(a["de_prob"] - b["de_prob"]).max())
ok = same and d == 0 and e == 0
print(f"R4 {os.environ['VER']}：抽样一致 {same}；yhat 最大差 {d:.2e}；de_prob 最大差 {e:.2e} → {'通过：逐位相同' if ok else '不一致！'}")
PYEOF
