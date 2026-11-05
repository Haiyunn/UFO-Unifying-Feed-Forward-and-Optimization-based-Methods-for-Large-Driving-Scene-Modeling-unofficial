#!/bin/bash
# Wait for both Waymo prior shards, then compute the Argoverse prior on GPU 1 and 7 (two shards).
cd /mnt/data/why/UFO
until grep -q PRIOR_DONE logs/prior_waymo_0.log && grep -q PRIOR_DONE logs/prior_waymo_1.log; do sleep 60; done
source /root/miniconda3/etc/profile.d/conda.sh && conda activate drivestudio
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
for shard in 0 1; do
  gpu=$([ $shard = 0 ] && echo 1 || echo 7)
  scenes=$(ls cache/argoverse_160x240 | awk -v s=$shard "NR % 2 == s")
  CUDA_VISIBLE_DEVICES=$gpu python -u -m ufo.data.depth_prior --cache cache/argoverse_160x240 --dataset argoverse \
    --root /mnt/data/dataset/argoverse/processed/training --scenes $scenes > logs/prior_argo_$shard.log 2>&1 &
done
wait
echo ARGO_PRIOR_DONE >> logs/prior_argo_0.log
