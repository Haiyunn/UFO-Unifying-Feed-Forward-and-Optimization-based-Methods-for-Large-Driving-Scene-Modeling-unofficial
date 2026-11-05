#!/bin/bash
# Evaluate the 8 s cosine fine-tune: intermediate ckpts (2s/8s, quick) as they appear,
# final ckpt on 2s/8s/16s + one continuous full-scene render per val scene (no stitching).
cd /mnt/data/why/UFO
source /root/miniconda3/bin/activate drivestudio
export OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=${1:-6}
R=runs/ft_8s_cosine; mkdir -p results/ft_8s_cosine
for s in 004000 008000; do
  until [ -f $R/ckpt_$s.pt ]; do sleep 120; done; sleep 30
  python -m scripts.eval --ckpt $R/ckpt_$s.pt --seconds 2 8 --no-lpips --out results/ft_8s_cosine/eval_$s.json > logs/eval_ft_$s.log 2>&1
done
until [ -f $R/ckpt_012000.pt ]; do sleep 120; done; sleep 30
python -m scripts.eval --ckpt $R/ckpt_012000.pt --seconds 2 8 16 --out results/ft_8s_cosine/eval_012000.json > logs/eval_ft_012000.log 2>&1
for s in 005 025 049 082 102 146 213 245 314 438 581 003; do
  python -m scripts.sample_scene --ckpt $R/ckpt_012000.pt --cache-root cache/waymo_160x240 --scene $s \
      --out results/samples/ft_012000/waymo_$s > /dev/null 2>&1
done
python -m scripts.sample_scene --ckpt $R/ckpt_012000.pt --cache-root cache/argoverse_160x240 --scene 017 \
    --out results/samples/ft_012000/argoverse_017 > /dev/null 2>&1
echo "$(date) ft pipeline done"
