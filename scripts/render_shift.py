"""Render a whole cached scene from laterally shifted camera rigs.

The scene is reconstructed ONCE in a single recurrent pass (context = every
5th frame, as in training), then every frame is rendered from the original
rig and from rigs translated along the ego lateral axis (ego +y = left, as in
Waymo/Argoverse ego frames). Shifted views have no ground truth, so metrics
are reported only for the unshifted rig.

    python -m scripts.render_shift --ckpt runs/v2_prior_pix_2s/ckpt_030000.pt \
        --cache-root cache/argoverse_160x240 --scene 017 --shifts 2 -2 \
        --out results/samples/argoverse_017_shift2m
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

from scripts.sample_scene import depth_vis, to_u8
from scripts.train import build_model, to_device
from ufo.data.dataset import SceneCache, UFOSequences, collate
from ufo.losses import depth_rmse_per_image, psnr_per_image, ssim_per_image
from ufo.render import gaussians_at, render_views


def label(img: np.ndarray, text: str) -> np.ndarray:
    im = Image.fromarray(img)
    ImageDraw.Draw(im).text((4, 2), text, fill=(255, 255, 0))
    return np.asarray(im)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--shifts", type=float, nargs="+", default=[2.0, -2.0], help="metres along ego +y (left)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=10)
    args = ap.parse_args()

    state = torch.load(args.ckpt, map_location="cuda")
    cfg = state["cfg"]
    cfg["model"]["grad_checkpoint"] = False
    model = build_model(cfg).cuda().eval()
    model.load_state_dict(state["model"])

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump({"val": [args.scene]}, fh)
        split = fh.name
    sc = SceneCache(os.path.join(args.cache_root, args.scene))
    n = len(sc)
    ds = UFOSequences(args.cache_root, split, "val", seq_frames=n, ctx_stride=cfg["data"].get("ctx_stride", 5),
                      max_objects=cfg["model"].get("max_objects", 32), train=False,
                      load_prior=bool(cfg["model"].get("depth_prior", False)))
    os.remove(split)
    assert len(ds) == 1
    dev = torch.device("cuda")
    item = collate([ds[0]])
    batch = to_device(item, dev)
    # Feed-forward time = recurrent reconstruction + Gaussian decoding (no rendering);
    # the first pass warms up CUDA kernels and is not timed.
    model.forward_sequence(batch)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    mems, sky, affine = model.forward_sequence(batch)
    mem = mems[0]
    g = model.decode(mem, batch["boxes"], 0)
    torch.cuda.synchronize()
    ff_seconds = time.time() - t0
    ff_peak_gb = torch.cuda.max_memory_allocated() / 2**30
    n_ctx = int(batch["context"]["frame"].shape[1])
    print(f"reconstructed {n} frames ({n_ctx} context steps) in one pass: {mem.n} scene tokens, "
          f"feed-forward {ff_seconds:.3f}s, peak {ff_peak_gb:.2f} GB", flush=True)

    # All frames (context + target) with their GT images and cameras, in time order.
    frames = torch.cat([batch["context"]["frame"][0], batch["target"]["frame"][0]])
    times = torch.cat([batch["context"]["time"][0], batch["target"]["time"][0]])
    images = torch.cat([batch["context"]["images"][0], batch["target"]["images"][0]])
    K = torch.cat([batch["context"]["K"][0], batch["target"]["K"][0]])
    n_ctx_frames = batch["context"]["frame"].shape[1]
    is_ctx = torch.cat([torch.ones_like(batch["context"]["frame"][0]), torch.zeros_like(batch["target"]["frame"][0])])
    order = frames.argsort()

    world_from = np.linalg.inv(sc.ego_to_world[0])
    shifts = [0.0] + list(args.shifts)
    os.makedirs(args.out, exist_ok=True)
    cam_order = [1, 0, 2]  # left / front / right as the car sees them
    panels, per_shift = [], {s: [] for s in shifts}
    psnr_nov, psnr_ctx, ssim_nov, ssim_ctx, drmse_nov = [], [], [], [], []
    for i in order.tolist():
        f = int(frames[i])
        gs = gaussians_at(model, g, mem, batch["boxes"], 0, f, float(times[i]))
        rows = [label(np.concatenate(list(to_u8(images[i])[cam_order]), axis=1), f"GT f{f:03d}")]
        depth0 = None
        for s in shifts:
            shift = np.eye(4)
            shift[1, 3] = s
            c2w = world_from @ sc.ego_to_world[f] @ shift @ sc.cam_to_ego  # (C, 4, 4)
            out = render_views(model, gs, sky[0], affine[0], K[i], torch.from_numpy(c2w).float().to(dev))
            rgb = out["rgb"].clamp(0, 1)
            if s == 0.0:
                (psnr_ctx if is_ctx[i] else psnr_nov).extend(psnr_per_image(rgb, images[i]).tolist())
                (ssim_ctx if is_ctx[i] else ssim_nov).extend(ssim_per_image(rgb, images[i]).tolist())
                if not is_ctx[i]:
                    gt_d = batch["target"]["depth"][0, i - n_ctx_frames].unsqueeze(1)
                    dr, ok = depth_rmse_per_image(out["depth"], gt_d)
                    drmse_nov.extend(dr[ok].tolist())
                depth0 = out["depth"]
            strip = np.concatenate(list(to_u8(rgb)[cam_order]), axis=1)
            name = "original 0m" if s == 0 else (f"left {s:g}m" if s > 0 else f"right {-s:g}m")
            per_shift[s].append(label(strip, name))
        dstrip = np.concatenate(list(depth_vis(depth0)[cam_order]), axis=1)
        # Row order: GT, left, original, right, depth(original).
        left = [per_shift[s][-1] for s in shifts if s > 0]
        right = [per_shift[s][-1] for s in shifts if s < 0]
        panels.append(np.concatenate(rows + left + [per_shift[0.0][-1]] + right + [dstrip], axis=0))

    imageio.mimwrite(os.path.join(args.out, f"{args.scene}_shift_panel.mp4"), panels, fps=args.fps, quality=8,
                     macro_block_size=1)
    for s in shifts:
        tag = "0m" if s == 0 else (f"left{s:g}m" if s > 0 else f"right{-s:g}m")
        imageio.mimwrite(os.path.join(args.out, f"{args.scene}_{tag}.mp4"), per_shift[s], fps=args.fps, quality=8,
                         macro_block_size=1)
    for k in (0, len(panels) // 2, len(panels) - 1):
        imageio.imwrite(os.path.join(args.out, f"panel_{k:03d}.png"), panels[k])
    res = {"ckpt": args.ckpt, "step": state.get("step"), "scene": args.scene, "frames": n, "shifts_m_ego_y": shifts,
           "single_pass": True, "scene_tokens": int(mem.n), "context_steps": n_ctx,
           "feedforward_seconds": ff_seconds, "feedforward_ms_per_context_step": 1000 * ff_seconds / n_ctx,
           "feedforward_peak_gpu_gb": ff_peak_gb,
           "original_rig_novel_psnr": float(np.mean(psnr_nov)), "original_rig_novel_ssim": float(np.mean(ssim_nov)),
           "original_rig_novel_drmse": float(np.mean(drmse_nov)),
           "original_rig_context_psnr": float(np.mean(psnr_ctx)), "original_rig_context_ssim": float(np.mean(ssim_ctx)),
           "note": "shifted rigs have no ground truth; metrics only for the original rig"}
    with open(os.path.join(args.out, "metrics.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
