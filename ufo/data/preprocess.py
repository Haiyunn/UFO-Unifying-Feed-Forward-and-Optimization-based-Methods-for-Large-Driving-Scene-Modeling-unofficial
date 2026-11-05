"""Build the low-resolution UFO cache from the DriveStudio / NOTR Waymo layout.

UFO trains at 160x240 on the three front cameras.  An 8 s training sequence
touches 80 frames x 3 cameras, so decoding 1920x1280 JPEGs on the fly would
starve the GPUs; everything is resized once and stored as memory-mappable
arrays.

Per scene ``<out>/<scene>/``::

    images.npy   uint8   (F, C, H, W, 3)
    sky.npy      uint8   (F, C, H, W)      1 = sky
    dynamic.npy  uint8   (F, C, H, W)      1 = dynamic object (eval only)
    depth.npy    float16 (F, C, H, W)      projected LiDAR depth, 0 = no return
    meta.npz     frames (F,), intrinsics (C,3,3) at H x W, cam_to_ego (C,4,4)
                 in OpenCV axes, ego_to_world (F,4,4) float64
    boxes.npz    obj_to_world (O,F,4,4) float64, size (O,F,3), valid (O,F),
                 lidar_count (O,F), cls (O,) [0 vehicle,1 pedestrian,2 cyclist]

Usage:
    python -m ufo.data.preprocess --root /mnt/data/dataset/notr/processed/training \
        --split splits/waymo_ds.json --out cache/waymo_160x240 --workers 32
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

CAMERAS = (0, 1, 2)
CLASSES = {"Vehicle": 0, "Pedestrian": 1, "Cyclist": 2}

# Stored extrinsics are camera->ego in Waymo camera axes (x fwd, y left, z up);
# right-multiplying converts them to OpenCV axes (x right, y down, z fwd).
OPENCV2WAYMO = np.array(
    [[0.0, 0.0, 1.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
)


def read_lidar(path: str) -> np.ndarray | None:
    """(N, 3) ego-frame points, or None for missing / placeholder sweeps."""
    if not os.path.exists(path):
        return None
    raw = np.fromfile(path, dtype=np.float32)
    for cols in (14, 10):
        if raw.size % cols:
            continue
        a = raw.reshape(-1, cols)
        laser = np.unique(a[:, cols - 1])
        ground = np.unique(a[:, cols - 4])
        if len(laser) > 5 or not np.all(np.isin(laser, (0, 1, 2, 3, 4))):
            continue
        if not np.all(np.isin(ground, (0, 1))):
            continue
        return a[:, 3:6].astype(np.float64)
    return None


def project_depth(pts_ego: np.ndarray, cam_to_ego: np.ndarray, k: np.ndarray, hw) -> np.ndarray:
    h, w = hw
    depth = np.zeros((h, w), np.float32)
    cam_from_ego = np.linalg.inv(cam_to_ego)
    p = pts_ego @ cam_from_ego[:3, :3].T + cam_from_ego[:3, 3]
    z = p[:, 2]
    front = z > 1e-3
    p, z = p[front], z[front]
    if len(z) == 0:
        return depth
    uv = (p / z[:, None]) @ k.T
    u = np.floor(uv[:, 0]).astype(np.int64)
    v = np.floor(uv[:, 1]).astype(np.int64)
    keep = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v, z = u[keep], v[keep], z[keep]
    order = np.argsort(-z)  # far first so near returns win
    depth[v[order], u[order]] = z[order]
    return depth


def _load(path: str, hw, gray: bool) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("L" if gray else "RGB")
        im = im.resize((hw[1], hw[0]), Image.NEAREST if gray else Image.BILINEAR)
        return np.asarray(im)


def process_scene(args) -> str:
    root, scene, out, hw = args
    src = os.path.join(root, scene)
    dst = os.path.join(out, scene)
    if os.path.exists(os.path.join(dst, "boxes.npz")):
        return f"{scene}: cached"
    os.makedirs(dst, exist_ok=True)

    frames = sorted(
        int(f.split("_")[0])
        for f in os.listdir(os.path.join(src, "images"))
        if f.endswith(f"_{CAMERAS[0]}.jpg")
    )
    frames = [
        f for f in frames
        if all(os.path.exists(os.path.join(src, "images", f"{f:03d}_{c}.jpg")) for c in CAMERAS)
    ]
    n_f, n_c = len(frames), len(CAMERAS)
    h, w = hw

    intr, c2e = [], []
    for c in CAMERAS:
        fx, fy, cx, cy = np.loadtxt(os.path.join(src, "intrinsics", f"{c}.txt")).reshape(-1)[:4]
        orig_w, orig_h = _orig_size(os.path.join(src, "images", f"{frames[0]:03d}_{c}.jpg"))
        sx, sy = w / orig_w, h / orig_h
        intr.append(np.array([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1]]))
        c2e.append(np.loadtxt(os.path.join(src, "extrinsics", f"{c}.txt")).reshape(4, 4) @ OPENCV2WAYMO)
    intr, c2e = np.stack(intr), np.stack(c2e)
    ego = np.stack([np.loadtxt(os.path.join(src, "ego_pose", f"{f:03d}.txt")).reshape(4, 4) for f in frames])

    images = np.zeros((n_f, n_c, h, w, 3), np.uint8)
    sky = np.zeros((n_f, n_c, h, w), np.uint8)
    dyn = np.zeros((n_f, n_c, h, w), np.uint8)
    depth = np.zeros((n_f, n_c, h, w), np.float16)
    lidar_world = []
    for i, f in enumerate(frames):
        pts = read_lidar(os.path.join(src, "lidar", f"{f:03d}.bin"))
        for j, c in enumerate(CAMERAS):
            images[i, j] = _load(os.path.join(src, "images", f"{f:03d}_{c}.jpg"), hw, False)
            sp = os.path.join(src, "sky_masks", f"{f:03d}_{c}.png")
            if os.path.exists(sp):
                sky[i, j] = _load(sp, hw, True) > 127
            dp = os.path.join(src, "dynamic_masks", "all", f"{f:03d}_{c}.png")
            if os.path.exists(dp):
                dyn[i, j] = _load(dp, hw, True) > 127
            if pts is not None:
                depth[i, j] = project_depth(pts, c2e[j], intr[j], hw)
        if pts is not None:
            lidar_world.append(pts @ ego[i, :3, :3].T + ego[i, :3, 3])
        else:
            lidar_world.append(np.zeros((0, 3)))

    # Boxes: every annotated instance, with per-frame LiDAR support so the
    # loader can rank them like the paper (number of points inside the box).
    with open(os.path.join(src, "instances", "instances_info.json")) as fh:
        info = json.load(fh)
    slot = {f: i for i, f in enumerate(frames)}
    o2w, size, valid, count, cls = [], [], [], [], []
    for entry in info.values():
        ann = entry.get("frame_annotations", {})
        if not ann.get("frame_idx"):
            continue
        P = np.tile(np.eye(4), (n_f, 1, 1))
        S = np.ones((n_f, 3))
        V = np.zeros(n_f, bool)
        N = np.zeros(n_f, np.int32)
        for f, pose, sz in zip(ann["frame_idx"], ann["obj_to_world"], ann["box_size"]):
            if int(f) not in slot:
                continue
            i = slot[int(f)]
            P[i] = np.asarray(pose).reshape(4, 4)
            S[i] = np.asarray(sz)
            V[i] = True
            pts = lidar_world[i]
            if len(pts):
                inv = np.linalg.inv(P[i])
                loc = pts @ inv[:3, :3].T + inv[:3, 3]
                N[i] = int(np.all(np.abs(loc) <= S[i] / 2, axis=1).sum())
        if not V.any():
            continue
        o2w.append(P), size.append(S), valid.append(V), count.append(N)
        cls.append(CLASSES.get(entry.get("class_name"), 0))

    np.save(os.path.join(dst, "images.npy"), images)
    np.save(os.path.join(dst, "sky.npy"), sky)
    np.save(os.path.join(dst, "dynamic.npy"), dyn)
    np.save(os.path.join(dst, "depth.npy"), depth)
    np.savez(os.path.join(dst, "meta.npz"), frames=np.asarray(frames), intrinsics=intr, cam_to_ego=c2e, ego_to_world=ego)
    o = len(o2w)
    np.savez(
        os.path.join(dst, "boxes.npz"),
        obj_to_world=np.stack(o2w) if o else np.zeros((0, n_f, 4, 4)),
        size=np.stack(size) if o else np.zeros((0, n_f, 3)),
        valid=np.stack(valid) if o else np.zeros((0, n_f), bool),
        lidar_count=np.stack(count) if o else np.zeros((0, n_f), np.int32),
        cls=np.asarray(cls, np.int64),
    )
    return f"{scene}: {n_f} frames, {o} objects"


def _orig_size(path: str):
    from PIL import Image

    with Image.open(path) as im:
        return im.size


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/data/dataset/notr/processed/training")
    ap.add_argument("--split", default="splits/waymo_ds.json")
    ap.add_argument("--out", default="cache/waymo_160x240")
    ap.add_argument("--height", type=int, default=160)
    ap.add_argument("--width", type=int, default=240)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--scenes", nargs="*", default=None)
    args = ap.parse_args()

    with open(args.split) as fh:
        split = json.load(fh)
    scenes = args.scenes or sorted(set(split["train"]) | set(split["val"]))
    os.makedirs(args.out, exist_ok=True)
    jobs = [(args.root, s, args.out, (args.height, args.width)) for s in scenes]
    with ProcessPoolExecutor(args.workers) as ex:
        for msg in ex.map(process_scene, jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
