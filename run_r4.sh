#!/bin/bash
# 从头训练 R4（与 2026-10-20 runs3/h1_v6 V6_R4_jepa_IDM_SIG_s{0,1} 的命令行参数完全相同），再预测（期望值读出与 nbw 读出，5 次重复）与 cell-eval。
# 注意：GPU 训练不是逐位确定的；同配置同种子重跑，split_811 test 上 PDS 曾为 0.705 / 0.716（原跑 0.736 / 0.735）。
# 用法：GPU=0 ./run_r4.sh [种子 ...]（默认 0 1）；输出 runs/n32/split_811/R4_s<种子>、runs/n32/pred_*、scripts/report/compare_r4.csv
set -u
cd /cluster2/home/hanwen/vcc_26/pertfold_r4/scripts
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/cluster2/home/hanwen/miniconda3/envs/vcc26/bin/python
EV=/cluster2/home/hanwen/vcc_26/backbone/.venv-eval/bin/python
SPLIT=/cluster2/home/hanwen/vcc_26/pertfold/data/h1/split_811.json
REF=/cluster2/home/hanwen/vcc_26/pertfold/runs3/split_811_ref/n32
CK=stage2_ft_lr0.0001_best_pearson_exp
R4="--relation string+perturbseq_k562+depmap --nb-learn --value-enc mul --value-norm --norm-bulk --jepa-mode sg --idm 1 --jepa-sigreg 0.1 --mask-targets"
SEEDS=${@:-0 1}
for s in $SEEDS; do
  CUDA_VISIBLE_DEVICES=${GPU:-0} $PY -u train_h1_v5.py 32 --split $SPLIT --gpu 0 --s1 3000 --lr1 3e-4 --s2 5000 --stage2 ft:1e-4 \
      --eval-every 25 --seed $s --tag R4_s$s $R4 --out ../runs > ../runs/train_R4_s$s.log 2>&1 || { echo "训练失败 s$s"; exit 1; }
  for ro in exp nbw; do
    CUDA_VISIBLE_DEVICES=${GPU:-0} $PY -u predict_h1_v5.py 32 ${CK}.pt --split $SPLIT --tag R4_s$s --reps 5 --gpu 0 --runs ../runs \
        --readout $ro >> ../runs/predict_R4_s$s.log 2>&1
  done
done
for f in ../runs/n32/pred_R4_s*_rep?.npz; do $EV -u eval_model_h1.py $f > ${f%.npz}_eval.log 2>&1 & done; wait
ROWS=("STRING kNN=$REF/pred_string_knn_rep?_per_pert.parquet")
for s in $SEEDS; do for ro in exp nbw; do ROWS+=("R4 种子 $s $ro=../runs/n32/pred_R4_s${s}_${CK}_${ro}_rep?_per_pert.parquet"); done; done
$EV compare_split.py $SPLIT compare_r4 "${ROWS[@]}"
