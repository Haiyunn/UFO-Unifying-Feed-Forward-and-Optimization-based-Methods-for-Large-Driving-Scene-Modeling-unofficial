#!/bin/bash
# One continuous recurrent pass over every unseen scene (no window stitching), 30k stage-1 model.
cd /mnt/data/why/UFO
source /root/miniconda3/bin/activate drivestudio
export OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=${1:-2}
for s in 005 025 049 082 102 146 213 245 314 438 581 003; do
  python -m scripts.sample_scene --ckpt runs/stage1_2s/ckpt_030000.pt --cache-root cache/waymo_160x240 \
      --scene $s --out results/samples/s1_030000/waymo_$s 2>&1 | grep -E "window|target_psnr|context_psnr|_full.mp4|Error"
done
python -m scripts.sample_scene --ckpt runs/stage1_2s/ckpt_030000.pt --cache-root cache/argoverse_160x240 \
    --scene 017 --out results/samples/s1_030000/argoverse_017 2>&1 | grep -E "window|target_psnr|context_psnr|_full.mp4|Error"
echo ALL_DONE
