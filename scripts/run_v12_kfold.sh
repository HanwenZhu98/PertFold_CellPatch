#!/bin/bash
# run_v12_kfold.sh <GPU> <tag> <种子> <折...> -- <额外参数>：在当前机器上依次训练 5 折模式（等该 GPU 上没有 train_h1_v12 进程后再开始）
cd /cluster2/home/hanwen/vcc_26/pertfold_r4/scripts && mkdir -p logs/v12
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/cluster2/home/hanwen/miniconda3/envs/vcc26/bin/python
g=$1; tag=$2; seed=$3; shift 3; folds=(); while [ "$1" != "--" ] && [ -n "$1" ]; do folds+=($1); shift; done; shift
while [ "$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F, -v g=$g '$1==g{print ($2>8000)}')" = "1" ]; do sleep 60; done
for k in "${folds[@]}"; do
  echo "$(date +%F' '%T) $(hostname) GPU$g 开始 ${tag}_s${seed}_fold$k" >> logs/v12/run_kfold.log
  CUDA_VISIBLE_DEVICES=$g $PY -u train_h1_v12.py --fold $k --seed $seed --tag $tag "$@" > logs/v12/${tag}_s${seed}_fold$k.log 2>&1 \
    && echo "$(date +%F' '%T) $(hostname) ${tag}_s${seed}_fold$k 完成" >> logs/v12/run_kfold.log \
    || echo "$(date +%F' '%T) $(hostname) ${tag}_s${seed}_fold$k 失败" >> logs/v12/run_kfold.log
done
