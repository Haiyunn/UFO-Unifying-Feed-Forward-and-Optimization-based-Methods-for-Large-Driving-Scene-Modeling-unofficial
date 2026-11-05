#!/bin/bash
# Improvement-loop arm: train from scratch on one GPU, then UFO-protocol 2s evals.
# $1 = run dir (holds config.yaml), $2 = GPU, $3 = extra eval: none | train | heldout30
set -e
cd /mnt/data/why/UFO
source /root/miniconda3/etc/profile.d/conda.sh && conda activate drivestudio
export CUDA_VISIBLE_DEVICES=$2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
RUN=$1
python -u -m scripts.train --config $RUN/config.yaml --resume
CK=$RUN/ckpt_latest.pt
python -u -m scripts.eval --ckpt $CK --config $RUN/config.yaml --seconds 2 --split val --no-lpips --out $RUN/eval_val_2s.json
case $3 in
  train) python -u -m scripts.eval --ckpt $CK --config $RUN/config.yaml --seconds 2 --split train --max-windows 88 --no-lpips --out $RUN/eval_train88_2s.json ;;
  heldout30) python -u -m scripts.eval --ckpt $CK --config $RUN/config_heldout30.yaml --seconds 2 --split val --no-lpips --out $RUN/eval_heldout30_2s.json ;;
esac
echo ARM_DONE
