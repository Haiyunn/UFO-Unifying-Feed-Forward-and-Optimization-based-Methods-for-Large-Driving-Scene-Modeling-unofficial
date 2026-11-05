"""Forward / loss / backward smoke test on real cached data, with timing + memory.

    python -m tests.smoke --seq 20 --batch 2
"""
import argparse
import time

import torch

from scripts.train import Wrapper, sequence_loss, to_device
from ufo.data.dataset import UFOSequences, collate
from ufo.losses import LPIPS
from ufo.model import UFO, UFOConfig

ap = argparse.ArgumentParser()
ap.add_argument("--seq", type=int, default=20)
ap.add_argument("--batch", type=int, default=2)
ap.add_argument("--iters", type=int, default=3)
ap.add_argument("--gpt", type=int, default=16)
args = ap.parse_args()
dev = torch.device("cuda")
ds = UFOSequences("cache/waymo_160x240", "splits/waymo_ds.json", "train", seq_frames=args.seq, num_target=4)
print(f"{len(ds)} sequences over {len(ds.scenes)} scenes")
core = UFO(UFOConfig(gaussians_per_token=args.gpt)).to(dev)
print(f"params {sum(p.numel() for p in core.parameters())/1e6:.1f}M")
model = Wrapper(core).train()
opt = torch.optim.AdamW(core.parameters(), lr=1e-4)
lp = LPIPS().to(dev)
for it in range(args.iters):
    batch = to_device(collate([ds[i * 997 % len(ds)] for i in range(it * args.batch, (it + 1) * args.batch)]), dev)
    torch.cuda.synchronize(); t0 = time.time()
    loss, logs = sequence_loss(model, batch, lp, use_lpips=True)
    opt.zero_grad(); loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0)
    opt.step(); torch.cuda.synchronize()
    nograd = [n for n, p in core.named_parameters() if p.grad is None]
    print(f"it {it} {time.time()-t0:.2f}s mem {torch.cuda.max_memory_allocated()/2**30:.1f}G loss {float(loss):.4f} gn {float(gn):.2f}")
    print("   ", {k: round(v, 4) for k, v in logs.items()})
    if it == 0:
        print("    params without grad:", nograd)
