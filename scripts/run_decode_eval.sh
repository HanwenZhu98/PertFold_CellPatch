#!/bin/bash
# run_decode_eval.sh <目录> <stem>：第 0 次重复扫 s ∈ {1,1.5,2,3,4,6,8} → 交叉拟合 → 5 次重复（与 run_bench 相同口径）
set -u
cd /cluster2/home/hanwen/vcc_26/pertfold_r4/scripts && mkdir -p logs/v12
EV=/cluster2/home/hanwen/vcc_26/backbone/.venv-eval/bin/python
OB=$1; st=$2
log() { echo "$(date +%F' '%T) $*" | tee -a logs/v12/decode.log; }
ev() { local f=$1; shift; EVAL_THREADS=10 $EV -u eval_bulk_h1.py $f "$@" > logs/v12/eval_$(basename $f .npz)_$(echo "$@" | tr ' ,' '__').log 2>&1; }
for s in 1 1.5 2 3 4 6 8; do [ -f $OB/${st}_rep0_s${s}_agg.csv ] || ev $OB/${st}_rep0.npz --scale $s & sleep 2; done; wait
FS=$($EV summarize_bench.py --pick $OB $st); log "$st：每折 s = $FS"
for r in 0 1 2 3 4; do ev $OB/${st}_rep$r.npz --fold-scale $FS & sleep 2; done; wait
log "$st 完成"
