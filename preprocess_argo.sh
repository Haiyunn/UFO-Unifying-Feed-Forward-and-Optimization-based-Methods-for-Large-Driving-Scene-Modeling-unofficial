#!/bin/bash
# Build cache/argoverse_160x240 for every scene in splits/argoverse.json, 8 scenes in parallel.
# Usage: bash preprocess_argo.sh [argoverse_root]
cd "$(dirname "$0")"
root=${1:-data/argoverse/processed/training}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
scenes=$(python3 -c "import json; d=json.load(open(\"splits/argoverse.json\")); print(\" \".join(sorted(set(d[\"train\"]+d[\"val\"]+d[\"test\"]))))")
printf "%s\n" $scenes | xargs -P 8 -I{} sh -c "test -f cache/argoverse_160x240/{}/boxes.npz && echo {}: cached || python -m ufo.data.preprocess_argoverse --root '$root' --scenes {}"
echo PREPROCESS_DONE
