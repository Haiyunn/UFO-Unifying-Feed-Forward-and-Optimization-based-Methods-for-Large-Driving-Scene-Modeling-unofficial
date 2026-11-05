"""Train UFO (single GPU or torchrun DDP).

    torchrun --nproc_per_node 2 -m scripts.train --config configs/stage1_2s.yaml
    torchrun --nproc_per_node 2 -m scripts.train --config configs/stage2_8s.yaml \
        --init runs/stage1_2s/ckpt_latest.pt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from ufo.data.dataset import build_train_set, collate
from ufo.losses import LPIPS, gaussian_losses, psnr_per_image, render_losses, total
from ufo.model import UFO, UFOConfig
from ufo.render import render_targets


def to_device(x, dev):
    if isinstance(x, torch.Tensor):
        return x.to(dev, non_blocking=True)
    if isinstance(x, dict):
        return {k: to_device(v, dev) for k, v in x.items()}
    return x


def build_model(cfg: dict) -> UFO:
    return UFO(UFOConfig(**cfg.get("model", {})))


def sequence_loss(model: UFO, batch: dict, lpips_fn, use_lpips: bool) -> tuple[torch.Tensor, dict]:
    core = (model.module if isinstance(model, DDP) else model).model
    # DDP copies forward() inputs, so anything the loss also reads is set here.
    batch["boxes"]["pose_inv"] = torch.linalg.inv(batch["boxes"]["pose"])
    # Decoding happens inside forward() so DDP sees every parameter used there.
    mems, sky, affine, gs = model(batch)
    b = len(mems)
    parts: dict[str, list] = {}
    for i in range(b):
        out, g = render_targets(core, mems[i], sky, affine, batch, i, batch["target"], g=gs[i])
        gt = {k: batch["target"][k][i].flatten(0, 1) for k in ("images", "depth", "sky")}
        flat = {k: v.flatten(0, 1) for k, v in out.items()}
        losses = render_losses(flat, gt, lpips_fn if use_lpips else None)
        # Forward flow of token centres between the first target frame and the next.
        f0 = int(batch["target"]["frame"][i, 0])
        f1 = min(f0 + 1, batch["seq_times"].shape[1] - 1)
        flow = None
        if core.cfg.use_boxes and mems[i].n:
            assign = g["assign_logits"].softmax(-1)
            T = core.motion(assign, mems[i], batch["boxes"], i, torch.tensor([f0, f1], device=assign.device))
            p = mems[i].pos
            flow = (T[1, :, :3, :3] @ p[:, :, None]).squeeze(-1) + T[1, :, :3, 3] - (
                (T[0, :, :3, :3] @ p[:, :, None]).squeeze(-1) + T[0, :, :3, 3])
        losses.update(gaussian_losses(g, flow))
        losses["_psnr"] = psnr_per_image(flat["rgb"].detach(), gt["images"]).mean()
        losses["_tokens"] = torch.tensor(float(mems[i].n))
        for k, v in losses.items():
            parts.setdefault(k, []).append(v)
    mean = {k: torch.stack([torch.as_tensor(x, device=batch["context"]["images"].device) for x in v]).mean()
            for k, v in parts.items()}
    loss = total({k: v for k, v in mean.items() if not k.startswith("_")}, use_lpips)
    return loss, {k: float(v) for k, v in mean.items()}


def load_compatible(core: UFO, state: dict) -> tuple[list, list, list]:
    """Warm-start across the depth_prior / pixel_aligned extensions.

    * patch_embed gains a 10th per-pixel input channel: the old weights fill the
      first 9 channels of every pixel, the new channel starts at zero, so the
      image tokens are unchanged at init.
    * depth_head changes meaning (absolute log distance -> log residual on the
      prior) and the Gaussian decoder's last layer changes size (M = 64): both
      keep their fresh initialisation.
    """
    own = core.state_dict()
    state = dict(state)
    reinit = []
    w_old = state.get("patch_embed.weight")
    if w_old is not None and w_old.shape != own["patch_embed.weight"].shape:
        pix = core.cfg.patch ** 2
        c_old = w_old.shape[1] // pix
        w_new = torch.zeros_like(own["patch_embed.weight"])
        w_new.view(w_new.shape[0], pix, core.in_ch)[:, :, :c_old] = w_old.view(w_old.shape[0], pix, c_old)
        state["patch_embed.weight"] = w_new
    if core.cfg.depth_prior:
        for k in ("depth_head.weight", "depth_head.bias"):
            state.pop(k, None)
            reinit.append(k)
    for k in list(state):
        if k in own and state[k].shape != own[k].shape:
            state.pop(k)
            reinit.append(k)
    missing, unexpected = core.load_state_dict(state, strict=False)
    return missing, unexpected, reinit


class Wrapper(torch.nn.Module):
    """DDP needs forward() to be the entry point."""

    def __init__(self, model: UFO) -> None:
        super().__init__()
        self.model = model
        self.cfg = model.cfg

    def forward(self, batch):
        mems, sky, affine = self.model.forward_sequence(batch)
        gs = [self.model.decode(m, batch["boxes"], i) for i, m in enumerate(mems)]
        return mems, sky, affine, gs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--init", default=None, help="warm-start weights (e.g. 2 s ckpt for 8 s fine-tuning)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    else:
        rank, world = 0, 1
    dev = torch.device("cuda")
    torch.manual_seed(cfg.get("seed", 0) + rank)
    out_dir = cfg["output"]
    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "config.yaml"), "w") as fh:
            yaml.safe_dump(cfg, fh)

    d, t = cfg["data"], cfg["train"]
    ds = build_train_set(cfg)
    sampler = DistributedSampler(ds, shuffle=True, seed=cfg.get("seed", 0)) if ddp else None
    loader = DataLoader(ds, batch_size=t["batch_size"], sampler=sampler, shuffle=sampler is None,
                        num_workers=t.get("num_workers", 6), collate_fn=collate, drop_last=True,
                        pin_memory=True, persistent_workers=True)
    if rank == 0:
        parts = getattr(ds, "datasets", [ds])
        print(f"train: {len(ds)} sequences over {sum(len(p.scenes) for p in parts)} scenes "
              f"({', '.join(str(len(p.scenes)) for p in parts)}), world={world}", flush=True)

    core = build_model(cfg).to(dev)
    if rank == 0:
        print(f"params: {sum(p.numel() for p in core.parameters()) / 1e6:.1f}M", flush=True)
    step = 0
    opt = torch.optim.AdamW(core.parameters(), lr=t["lr"], weight_decay=t.get("weight_decay", 0.05),
                            betas=(0.9, 0.95))
    latest = os.path.join(out_dir, "ckpt_latest.pt")
    if args.resume and os.path.exists(latest):
        state = torch.load(latest, map_location=dev)
        core.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        step = state["step"]
        if rank == 0:
            print(f"resumed at step {step}", flush=True)
    elif args.init:
        state = torch.load(args.init, map_location=dev)
        missing, unexpected, reinit = load_compatible(core, state["model"])
        if rank == 0:
            print(f"init from {args.init}: missing {len(missing)} unexpected {len(unexpected)} "
                  f"re-initialised (shape/meaning changed) {reinit}", flush=True)

    model = Wrapper(core)
    if ddp:
        model = DDP(model, device_ids=[dev.index], find_unused_parameters=True)
    lpips_fn = LPIPS().to(dev)

    steps = args.max_steps or t["steps"]
    warm = t.get("warmup_steps", 5000)
    lpips_after = t.get("lpips_after", 5000)
    log_every, save_every = t.get("log_every", 20), t.get("save_every", 1000)
    lr_base = t["lr"]
    # "constant" (stage 1/2 default) or "cosine": decay to lr_min after warm-up.
    schedule, lr_min = t.get("schedule", "constant"), t.get("lr_min", 0.0)
    epoch = 0
    acc: dict[str, float] = {}
    tic = time.time()
    model.train()
    while step < steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            if step >= steps:
                break
            batch = to_device(batch, dev)
            lr = lr_base * min(1.0, (step + 1) / warm)
            if schedule == "cosine" and step >= warm:
                prog = (step - warm) / max(1, steps - warm)
                lr = lr_min + (lr_base - lr_min) * 0.5 * (1.0 + math.cos(math.pi * prog))
            for gparam in opt.param_groups:
                gparam["lr"] = lr
            loss, logs = sequence_loss(model, batch, lpips_fn, use_lpips=step >= lpips_after)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(core.parameters(), t.get("grad_clip", 1.0))
            if torch.isfinite(gn):
                opt.step()
            step += 1
            logs["loss"] = float(loss)
            logs["gnorm"] = float(gn)
            for k, v in logs.items():
                acc[k] = acc.get(k, 0.0) + v
            if step % log_every == 0 and rank == 0:
                el = time.time() - tic
                msg = " ".join(f"{k} {v / log_every:.4f}" for k, v in sorted(acc.items()))
                mem = torch.cuda.max_memory_allocated() / 2 ** 30
                print(f"step {step} lr {lr:.2e} {el / log_every:.2f}s/it mem {mem:.1f}G | {msg}", flush=True)
                with open(os.path.join(out_dir, "log.jsonl"), "a") as fh:
                    fh.write(json.dumps({"step": step, "lr": lr, **{k: v / log_every for k, v in acc.items()}}) + "\n")
                acc, tic = {}, time.time()
            if rank == 0 and (step % save_every == 0 or step == steps):
                state = {"model": core.state_dict(), "opt": opt.state_dict(), "step": step, "cfg": cfg}
                torch.save(state, latest + ".tmp")
                os.replace(latest + ".tmp", latest)
                if step % t.get("keep_every", 10000) == 0 or step == steps:
                    torch.save({"model": core.state_dict(), "step": step, "cfg": cfg},
                               os.path.join(out_dir, f"ckpt_{step:06d}.pt"))
        epoch += 1
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
