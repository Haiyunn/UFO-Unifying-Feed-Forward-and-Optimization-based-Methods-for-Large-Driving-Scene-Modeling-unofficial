"""Sample (reconstruct + render) one cached scene with a UFO checkpoint.

By default the whole scene is reconstructed in ONE recurrent pass (context =
every 5th frame) and every frame is rendered from that single state, giving
one continuous video.  With ``--seconds`` the scene is cut into independent
non-overlapping windows (the UFO eval protocol); each window gets its own
video -- independent reconstructions are never concatenated.  Writes per-frame
panels (GT / prediction / depth, one column per camera), mp4s and metrics
split into novel (target) frames and context frames.

    python -m scripts.sample_scene --ckpt runs/stage1_2s/ckpt_latest.pt \
        --cache-root cache/argoverse_160x240 --scene 017 \
        --out results/samples/argoverse_017
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile

import imageio.v2 as imageio
import matplotlib
import numpy as np
import torch

from scripts.train import build_model, to_device
from ufo.data.dataset import SceneCache, UFOSequences, collate
from ufo.losses import depth_rmse_per_image, psnr_per_image, ssim_per_image
from ufo.render import render_targets


def depth_vis(d: torch.Tensor, max_d: float = 80.0) -> np.ndarray:
    """(V, 1, H, W) metric depth -> (V, H, W, 3) uint8 turbo, inverse-depth scaled."""
    inv = (1.0 / d.clamp_min(1.0)).squeeze(1).cpu().numpy()
    x = np.clip((inv - 1.0 / max_d) / (1.0 - 1.0 / max_d), 0, 1) ** 0.5
    return (matplotlib.colormaps["turbo"](x)[..., :3] * 255).astype(np.uint8)


def to_u8(x: torch.Tensor) -> np.ndarray:
    return (x.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--seconds", type=int, default=None, help="independent windows; default: whole scene in one pass")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--ctx-stride", type=int, default=None, help="context every n-th frame (default: training value)")
    args = ap.parse_args()

    state = torch.load(args.ckpt, map_location="cuda")
    cfg = state["cfg"]
    cfg["model"]["grad_checkpoint"] = False
    model = build_model(cfg).cuda().eval()
    model.load_state_dict(state["model"])
    step = state.get("step")
    print(f"loaded {args.ckpt} (step {step})", flush=True)

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump({"val": [args.scene]}, fh)
        split = fh.name
    d = cfg["data"]
    seq_frames = args.seconds * 10 if args.seconds else len(SceneCache(os.path.join(args.cache_root, args.scene)))
    stride = args.ctx_stride or d.get("ctx_stride", 5)
    tag_len = (f"{args.seconds}s" if args.seconds else "full") + (f"_ctx{stride}" if stride != d.get("ctx_stride", 5) else "")
    ds = UFOSequences(args.cache_root, split, "val", seq_frames=seq_frames, ctx_stride=stride,
                      max_objects=cfg["model"].get("max_objects", 32), train=False,
                      load_prior=bool(cfg["model"].get("depth_prior", False)))
    os.remove(split)
    assert len(ds), f"scene {args.scene} not found in {args.cache_root} or shorter than {seq_frames} frames"
    frame_dir = os.path.join(args.out, f"{tag_len}_frames")
    os.makedirs(frame_dir, exist_ok=True)
    dev = torch.device("cuda")

    stats: dict[str, list] = {}
    videos = []
    for w in range(len(ds)):
        panels = []
        item = collate([ds[w]])
        batch = to_device(item, dev)
        start = item["start"][0]
        mems, sky, affine = model.forward_sequence(batch)
        g = model.decode(mems[0], batch["boxes"], 0)
        for kind in ("context", "target"):
            views = batch[kind]
            for s in range(0, views["frame"].shape[1], 8):
                sub = {k: v[:, s: s + 8] for k, v in views.items()}
                out, _ = render_targets(model, mems[0], sky, affine, batch, 0, sub, g=g)
                for j in range(sub["frame"].shape[1]):
                    pred, gt = out["rgb"][j].clamp(0, 1), sub["images"][0, j]
                    stats.setdefault(f"{kind}_psnr", []).extend(psnr_per_image(pred, gt).tolist())
                    stats.setdefault(f"{kind}_ssim", []).extend(ssim_per_image(pred, gt).tolist())
                    if kind == "target":
                        dr, ok = depth_rmse_per_image(out["depth"][j], sub["depth"][0, j].unsqueeze(1))
                        stats.setdefault("target_drmse", []).extend(dr[ok].tolist())
                    # Columns are cameras ordered left / front / right like the car sees them.
                    order = [1, 0, 2]
                    rows = [to_u8(gt)[order], to_u8(pred)[order], depth_vis(out["depth"][j])[order]]
                    panel = np.concatenate([np.concatenate(list(r), axis=1) for r in rows], axis=0)
                    f_abs = start + int(sub["frame"][0, j])
                    tag = "ctx" if kind == "context" else "nov"
                    imageio.imwrite(os.path.join(frame_dir, f"{f_abs:03d}_{tag}.png"), panel)
                    panels.append((f_abs, panel))
        n_t = batch["target"]["frame"].shape[1] * 3
        print(f"  window {w + 1}/{len(ds)} @{start}: tokens {mems[0].n} "
              f"novel PSNR {np.mean(stats['target_psnr'][-n_t:]):.2f}", flush=True)
        panels.sort(key=lambda p: p[0])
        name = f"{args.scene}_{tag_len}.mp4" if len(ds) == 1 and not args.seconds else \
            f"{args.scene}_{tag_len}_win{start:03d}-{start + seq_frames - 1:03d}.mp4"
        videos.append(os.path.join(args.out, name))
        imageio.mimwrite(videos[-1], [p for _, p in panels], fps=args.fps, quality=8, macro_block_size=1)

    res = {"ckpt": args.ckpt, "step": step, "scene": args.scene, "seq_frames": seq_frames, "ctx_stride": stride, "windows": len(ds),
           "frames": len(ds) * seq_frames}
    res.update({k: float(np.mean(v)) for k, v in stats.items()})
    with open(os.path.join(args.out, f"metrics_{tag_len}.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps(res, indent=2))
    print("videos", *videos, sep="\n  ")


if __name__ == "__main__":
    main()
