#!/bin/bash
# $1 = gpu, $2 = shard index (0/1): depth prior for half of the Waymo cache.
cd /mnt/data/why/UFO
source /root/miniconda3/etc/profile.d/conda.sh && conda activate drivestudio
export CUDA_VISIBLE_DEVICES=$1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
scenes=$(ls cache/waymo_160x240 | awk -v s=$2 "NR % 2 == s")
python -u -m ufo.data.depth_prior --cache cache/waymo_160x240 --dataset waymo \
  --root /mnt/data/dataset/notr/processed/training --scenes $scenes
echo PRIOR_DONE
