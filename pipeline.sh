#!/bin/bash
# Chain: wait for stage 1 -> evaluate it -> stage 2 (8 s fine-tune) -> evaluate -> scalability.
# Launch detached:  setsid nohup bash pipeline.sh 6,7 > logs/pipeline.log 2>&1 < /dev/null &
gpus=${1:-6,7}
first=${gpus%%,*}
cd /mnt/data/why/UFO
source /root/miniconda3/bin/activate drivestudio
export OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1

s1=runs/stage1_2s/ckpt_050000.pt
s2=runs/stage2_8s/ckpt_020000.pt
until [ -f "$s1" ]; do sleep 120; done
echo "$(date) stage 1 done"; sleep 60

# Stage 2 first so the GPUs never idle; stage-1 eval runs on one GPU alongside.
bash launch.sh configs/stage2_8s.yaml "$gpus" 29612 --resume --init "$s1" &
CUDA_VISIBLE_DEVICES=$first python -m scripts.eval --ckpt "$s1" --seconds 2 8 16 \
    --out results/stage1_2s.json --save-images results/images/stage1 > logs/eval_stage1.log 2>&1
wait
echo "$(date) stage 2 done"

CUDA_VISIBLE_DEVICES=$first python -m scripts.eval --ckpt "$s2" --seconds 2 8 16 \
    --out results/stage2_8s.json --save-images results/images/stage2 > logs/eval_stage2.log 2>&1
CUDA_VISIBLE_DEVICES=$first python -m scripts.bench_scalability --ckpt "$s2" \
    --out results/scalability.json > logs/bench.log 2>&1
echo "$(date) all done"
