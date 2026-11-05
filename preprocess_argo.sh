#!/bin/bash
# Build cache/argoverse_160x240 for all 50 base scenes, 8 scenes in parallel (OMP_NUM_THREADS=1).
cd /mnt/data/why/UFO
source /root/miniconda3/etc/profile.d/conda.sh && conda activate drivestudio
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
scenes=$(python3 -c "import json; d=json.load(open(\"splits/argoverse.json\")); print(\" \".join(sorted(set(d[\"train\"]+d[\"val\"]+d[\"test\"]))))")
printf "%s\n" $scenes | xargs -P 8 -I{} sh -c "test -f cache/argoverse_160x240/{}/boxes.npz && echo {}: cached || python -m ufo.data.preprocess_argoverse --scenes {}"
echo PREPROCESS_DONE
