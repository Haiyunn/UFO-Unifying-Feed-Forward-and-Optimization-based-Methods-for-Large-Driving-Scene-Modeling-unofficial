#!/bin/bash
# Depth prior for half of the Waymo cache.
# Usage: bash prior_waymo.sh <gpu> <shard 0|1> [waymo_root] [infinidepth_repo] [infinidepth_ckpt]
cd "$(dirname "$0")"
root=${3:-data/waymo/processed/training}
repo=${4:-third_party/InfiniDepth}
ckpt=${5:-$repo/checkpoints/depth/infinidepth_depthsensor.ckpt}
export CUDA_VISIBLE_DEVICES=$1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
scenes=$(ls cache/waymo_160x240 | awk -v s=$2 "NR % 2 == s")
python -u -m ufo.data.depth_prior --cache cache/waymo_160x240 --dataset waymo \
  --root "$root" --depth-repo "$repo" --depth-ckpt "$ckpt" --scenes $scenes
echo PRIOR_DONE
