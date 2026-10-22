#!/bin/bash
# 从头训练 R4（v2 = 2026-10-22 起的 R4：在原 R4 上去掉 decoder 的冗余证据输入，只留 nbw），预测（期望值与 nbw 两种读出，5 次重复）并 cell-eval。
#   v2 的参数 = 原 R4（runs3/h1_v6 V6_R4_jepa_IDM_SIG）+ --ev-drop ctx,str,ctrl --ev-family-dropout 0.015625
#   （与 pertfold/runs3/h1_v7 V7_S1c修正、h1_v7k K_S1c修正 相同）。原 R4 见 git 标签 r4-v1。
# 注意：GPU 训练不是逐位确定的；同配置同种子重跑，split_811 上 PDS 可差 0.02–0.03，结论以 5 折为准。
# 用法：
#   GPU=0 ./run_r4.sh [种子 ...]          split_811，默认种子 0 1 → runs/n32/split_811/R4v2_s<种子>、scripts/report/compare_r4.csv
#   GPU=0 MODE=kfold ./run_r4.sh           5 折（147 个扰动，种子 0，每折留 23 个训练扰动作 val）→ runs/n32/fold<k>/R4v2_k、scripts/report/kfold_r4.csv
set -u
cd /cluster2/home/hanwen/vcc_26/pertfold_r4/scripts
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/cluster2/home/hanwen/miniconda3/envs/vcc26/bin/python
EV=/cluster2/home/hanwen/vcc_26/backbone/.venv-eval/bin/python
SPLIT=/cluster2/home/hanwen/vcc_26/pertfold/data/h1/split_811.json
REF=/cluster2/home/hanwen/vcc_26/pertfold/runs3/split_811_ref/n32
CK=stage2_ft_lr0.0001_best_pearson_exp
TRAIN="--s1 3000 --lr1 3e-4 --s2 5000 --stage2 ft:1e-4 --eval-every 25"
R4="--relation string+perturbseq_k562+depmap --nb-learn --value-enc mul --value-norm --norm-bulk --jepa-mode sg --idm 1 --jepa-sigreg 0.1 --mask-targets \
    --ev-drop ctx,str,ctrl --ev-family-dropout 0.015625"
mkdir -p ../runs
if [ "${MODE:-split}" = kfold ]; then
  for k in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=${GPU:-0} $PY -u train_h1_v5.py 32 $k --inner-val 23 --gpu 0 $TRAIN --seed 0 --tag R4v2_k $R4 --out ../runs \
        > ../runs/train_R4v2_k_fold$k.log 2>&1 || { echo "训练失败 第 $k 折"; exit 1; }
  done
  for ro in exp nbw; do
    CUDA_VISIBLE_DEVICES=${GPU:-0} $PY -u predict_h1_v5.py 32 ${CK}.pt --tag R4v2_k --reps 5 --gpu 0 --runs ../runs --readout $ro >> ../runs/predict_R4v2_k.log 2>&1
  done
  for f in ../runs/n32/pred_R4v2_k_*_rep?.npz; do $EV -u eval_model_h1.py $f > ${f%.npz}_eval.log 2>&1 & done; wait
  $EV summarize_kfold.py kfold_r4 "R4 完整模型=../runs/n32/pred_R4v2_k_${CK}_exp" "R4 nbw=../runs/n32/pred_R4v2_k_${CK}_nbw"
  exit
fi
SEEDS=${@:-0 1}
for s in $SEEDS; do
  CUDA_VISIBLE_DEVICES=${GPU:-0} $PY -u train_h1_v5.py 32 --split $SPLIT --gpu 0 $TRAIN --seed $s --tag R4v2_s$s $R4 --out ../runs \
      > ../runs/train_R4v2_s$s.log 2>&1 || { echo "训练失败 s$s"; exit 1; }
  for ro in exp nbw; do
    CUDA_VISIBLE_DEVICES=${GPU:-0} $PY -u predict_h1_v5.py 32 ${CK}.pt --split $SPLIT --tag R4v2_s$s --reps 5 --gpu 0 --runs ../runs \
        --readout $ro >> ../runs/predict_R4v2_s$s.log 2>&1
  done
done
for f in ../runs/n32/pred_R4v2_s*_rep?.npz; do $EV -u eval_model_h1.py $f > ${f%.npz}_eval.log 2>&1 & done; wait
ROWS=("STRING kNN=$REF/pred_string_knn_rep?_per_pert.parquet")
for s in $SEEDS; do for ro in exp nbw; do ROWS+=("R4 种子 $s $ro=../runs/n32/pred_R4v2_s${s}_${CK}_${ro}_rep?_per_pert.parquet"); done; done
$EV compare_split.py $SPLIT compare_r4 "${ROWS[@]}"
