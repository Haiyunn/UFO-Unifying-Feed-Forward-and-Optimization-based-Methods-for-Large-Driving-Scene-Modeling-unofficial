"""Paper Fig. 4: reconstruction time and peak memory vs number of context frames.

Batch size 1, 3 cameras, Gaussian rendering excluded (forward_sequence +
decode only).  Context frames are taken every 5th frame from one val log, so
n context frames span 0.5 * n seconds.

    python -m scripts.bench_scalability --ckpt runs/stage2_8s/ckpt_latest.pt
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from scripts.train import build_model, to_device
from ufo.data.dataset import UFOSequences, collate


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--counts", type=int, nargs="+", default=[4, 8, 12, 16, 20, 24, 28, 32])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    state = torch.load(args.ckpt, map_location="cuda")
    cfg = state["cfg"]
    cfg["model"]["grad_checkpoint"] = False
    model = build_model(cfg).cuda().eval()
    model.load_state_dict(state["model"])
    d = cfg["data"]
    rows = []
    for n in args.counts:
        ds = UFOSequences(d["cache_root"], d["split_file"], "val", seq_frames=5 * n, train=False, eval_span=5 * n)
        batch = to_device(collate([ds[0]]), torch.device("cuda"))
        times = []
        for r in range(args.repeats + 1):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            mems, _, _ = model.forward_sequence(batch)
            model.decode(mems[0], batch["boxes"], 0)
            torch.cuda.synchronize()
            if r:  # first run is warm-up
                times.append(time.time() - t0)
        row = {"context_frames": n, "seconds": n * 0.5, "time_s": sum(times) / len(times),
               "peak_mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30, "tokens": mems[0].n}
        rows.append(row)
        print(row, flush=True)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(rows, fh, indent=2)


if __name__ == "__main__":
    main()
