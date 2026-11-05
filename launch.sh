#!/bin/bash
# Usage: bash launch.sh <config> <gpus> <master_port> [extra train args...]
# Activate your Python environment first.  Logs go to logs/train_<config>.log.
cfg=$1; gpus=$2; port=$3; shift 3
name=$(basename "$cfg" .yaml)
cd "$(dirname "$0")"
mkdir -p logs
export OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$gpus
n=$(echo "$gpus" | tr ',' '\n' | wc -l)
exec torchrun --nproc_per_node "$n" --master_port "$port" -m scripts.train --config "$cfg" "$@" >> logs/train_$name.log 2>&1
