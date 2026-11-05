"""Precompute a dense metric depth prior for every cached frame / camera.

UFO itself learns each new token's distance from scratch (initialised at 15 m),
which on 60 training scenes leaves a 7 m validation D-RMSE.  PointForward's
answer is a frozen depth foundation model; this script runs the same one --
InfiniDepth, the ``InfiniDepth_DepthSensor`` variant prompted with that frame's
projected LiDAR sweep -- once over the cache, so training never pays for it.

The model sees the source image at 3x the cache resolution (480x720, with the
same crop the cache used) and its implicit depth field is queried directly at
the cache resolution.  Output per scene::

    prior.npy  float16 (F, C, H, W)  z-depth in metres, 0 = no prediction

Only a frame's *own* sweep is used, so the prior of a context frame carries no
information about any target frame.

    python -m ufo.data.depth_prior --cache cache/waymo_160x240 --dataset waymo \
        --root data/waymo/processed/training --scenes 016 017
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

from ufo.data import preprocess, preprocess_argoverse

INFINIDEPTH_ROOT = "third_party/InfiniDepth"
INFINIDEPTH_CKPT = "third_party/InfiniDepth/checkpoints/depth/infinidepth_depthsensor.ckpt"
WORK_SCALE = 3


def dense_query_coord(h: int, w: int, device) -> torch.Tensor:
    """(H*W, 2) (y, x) pixel-centre coordinates in [-1, 1] (InfiniDepth convention)."""

    def seq(n: int) -> torch.Tensor:
        r = 1.0 / n
        return -1.0 + r + 2.0 * r * torch.arange(n, device=device, dtype=torch.float32)

    yy, xx = torch.meshgrid(seq(h), seq(w), indexing="ij")
    return torch.stack((yy, xx), dim=-1).reshape(-1, 2)


def load_infinidepth(repo_root: str = INFINIDEPTH_ROOT, checkpoint: str = INFINIDEPTH_CKPT):
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from InfiniDepth.utils.model_utils import build_model

    model = build_model("InfiniDepth_DepthSensor", model_path=checkpoint, encoder="vitl16")
    model.requires_grad_(False)
    model.eval()
    return model


def load_view(dataset: str, path: str, hw) -> np.ndarray:
    """Source image at ``hw`` with the cache's crop (Argoverse) or plain resize (Waymo)."""
    if dataset == "waymo":
        return preprocess._load(path, hw, False)
    with Image.open(path) as im:
        w0, h0 = im.size
    box = preprocess_argoverse.crop_box(w0, h0, hw[1] / hw[0])
    return preprocess_argoverse._load(path, box, hw, False)


def read_sweep(dataset: str, path: str):
    return preprocess.read_lidar(path) if dataset == "waymo" else preprocess_argoverse.read_lidar(path)


@torch.no_grad()
def process_scene(model, cache_dir: str, src_dir: str, dataset: str, cameras) -> str:
    out_path = os.path.join(cache_dir, "prior.npy")
    if os.path.exists(out_path):
        return f"{os.path.basename(cache_dir)}: cached"
    meta = np.load(os.path.join(cache_dir, "meta.npz"))
    frames, K, c2e = meta["frames"], meta["intrinsics"], meta["cam_to_ego"]
    h, w = np.load(os.path.join(cache_dir, "images.npy"), mmap_mode="r").shape[2:4]
    hw_work = (h * WORK_SCALE, w * WORK_SCALE)
    K_work = K.copy()
    K_work[:, :2] *= WORK_SCALE
    coord = dense_query_coord(h, w, "cuda")
    prior = np.zeros((len(frames), len(cameras), h, w), np.float16)
    no_lidar = 0
    for i, f in enumerate(frames):
        imgs = np.stack([load_view(dataset, os.path.join(src_dir, "images", f"{f:03d}_{c}.jpg"), hw_work)
                         for c in cameras])
        pts = read_sweep(dataset, os.path.join(src_dir, "lidar", f"{f:03d}.bin"))
        if pts is None or len(pts) == 0:
            no_lidar += 1
            continue
        sparse = np.stack([preprocess.project_depth(pts, c2e[j], K_work[j], hw_work) for j in range(len(cameras))])
        x = torch.from_numpy(imgs).cuda().permute(0, 3, 1, 2).float() / 255.0
        sd = torch.from_numpy(sparse).cuda().unsqueeze(1)
        mask = (sd > 0).float()
        disp = sd.clone()
        disp[sd > 0] = 1.0 / sd[sd > 0]
        out = model.inference(x, coord.unsqueeze(0).expand(len(cameras), -1, -1), use_batch_infer=True,
                              prompt_depth=disp, gt_depth=disp, prompt_mask=mask, gt_depth_mask=mask)
        depth = out[0] if isinstance(out, (tuple, list)) else out
        prior[i] = depth.reshape(len(cameras), h, w).float().clamp(0, 1000).cpu().numpy().astype(np.float16)
    np.save(out_path + ".tmp.npy", prior)
    os.replace(out_path + ".tmp.npy", out_path)
    lidar = np.load(os.path.join(cache_dir, "depth.npy"), mmap_mode="r").astype(np.float32)
    ok = (lidar > 0) & (prior > 0)
    rel = float(np.abs(prior[ok].astype(np.float32) - lidar[ok]).mean() / lidar[ok].mean()) if ok.any() else float("nan")
    return f"{os.path.basename(cache_dir)}: {len(frames)} frames, no-lidar frames {no_lidar}, rel err vs lidar {rel:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--dataset", choices=("waymo", "argoverse"), required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--depth-repo", default=INFINIDEPTH_ROOT, help="cloned InfiniDepth repository")
    ap.add_argument("--depth-ckpt", default=INFINIDEPTH_CKPT, help="InfiniDepth checkpoint")
    args = ap.parse_args()
    model = load_infinidepth(args.depth_repo, args.depth_ckpt)
    for s in args.scenes:
        t0 = time.time()
        msg = process_scene(model, os.path.join(args.cache, s), os.path.join(args.root, s), args.dataset,
                            preprocess.CAMERAS)
        print(f"{msg}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
