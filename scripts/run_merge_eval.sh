#!/bin/bash
# run_merge_eval.sh <tag_s种子>：等 5 折测试预测齐全 → 合并 → run_decode_eval.sh（s 网格 1–8，交叉拟合，5 次重复）
cd /cluster2/home/hanwen/vcc_26/pertfold_r4/scripts && mkdir -p logs/v12
EV=/cluster2/home/hanwen/vcc_26/backbone/.venv-eval/bin/python
OB=/cluster2/home/hanwen/vcc_26/pertfold/runs3/h1_v12/n32
until [ $(ls $OB/pred_$1_fold?_test_rep4.npz 2>/dev/null | wc -l) -ge 5 ]; do sleep 60; done
$EV merge_folds.py $1 && echo "$(date +%F' '%T) $1 5 折合并完成" >> logs/v12/decode.log
exec ./run_decode_eval.sh $OB pred_$1_k
