#!/bin/bash
# Sample the 30k stage-1 model on every unseen Waymo scene (11 PF val + out-of-split 003).
cd /mnt/data/why/UFO
source /root/miniconda3/bin/activate drivestudio
export OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=${1:-2}
OMP_NUM_THREADS=1 python -m ufo.data.preprocess --scenes 003 --workers 1
for s in 005 025 049 082 102 146 213 245 314 438 581 003; do
  python -m scripts.sample_scene --ckpt runs/stage1_2s/ckpt_030000.pt --cache-root cache/waymo_160x240 \
      --scene $s --seconds 2 --out results/samples/s1_030000/waymo_$s 2>&1 | grep -E "window|target_psnr|video"
done
echo ALL_DONE
