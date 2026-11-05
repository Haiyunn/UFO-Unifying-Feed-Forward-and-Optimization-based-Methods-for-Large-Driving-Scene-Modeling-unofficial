"""Build a UFO cache (same layout as :mod:`ufo.data.preprocess`) from the
DriveStudio-processed Argoverse-2 layout, for cross-dataset sampling of the
Waymo-trained model.

Differences from Waymo handled here:
  * extrinsics are already camera->ego in OpenCV axes (no OPENCV2WAYMO);
  * ring_front_center (cam 0) is portrait 1550x2048 and the side cameras are
    2048x1550, so every camera is centre-cropped to the model's 3:2 aspect
    before resizing (intrinsics shifted accordingly);
  * LiDAR .bin is float32 (x, y, z, intensity) in the ego frame;
  * Argoverse class names.

    python -m ufo.data.preprocess_argoverse --scenes 017
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from PIL import Image

from ufo.data.preprocess import CAMERAS, project_depth

CLASSES = {
    "REGULAR_VEHICLE": 0, "LARGE_VEHICLE": 0, "BUS": 0, "BOX_TRUCK": 0, "TRUCK": 0, "TRUCK_CAB": 0,
    "VEHICULAR_TRAILER": 0, "ARTICULATED_BUS": 0, "SCHOOL_BUS": 0, "MOTORCYCLE": 2, "BICYCLE": 2,
    "WHEELED_DEVICE": 2, "PEDESTRIAN": 1,
}


def read_lidar(path: str) -> np.ndarray | None:
    if not os.path.exists(path):
        return None
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % 4:
        return None
    return raw.reshape(-1, 4)[:, :3].astype(np.float64)


def crop_box(w0: int, h0: int, aspect: float) -> tuple[int, int, int, int]:
    """Centre crop (left, top, width, height) of a w0 x h0 image to width/height = aspect."""
    if w0 / h0 < aspect:
        hc = int(round(w0 / aspect))
        return 0, (h0 - hc) // 2, w0, hc
    wc = int(round(h0 * aspect))
    return (w0 - wc) // 2, 0, wc, h0


def _load(path: str, box, hw, gray: bool) -> np.ndarray:
    left, top, wc, hc = box
    with Image.open(path) as im:
        im = im.convert("L" if gray else "RGB").crop((left, top, left + wc, top + hc))
        im = im.resize((hw[1], hw[0]), Image.NEAREST if gray else Image.BILINEAR)
        return np.asarray(im)


def process_scene(root: str, scene: str, out: str, hw) -> str:
    src, dst = os.path.join(root, scene), os.path.join(out, scene)
    os.makedirs(dst, exist_ok=True)
    h, w = hw
    frames = sorted(
        int(f.split("_")[0]) for f in os.listdir(os.path.join(src, "images")) if f.endswith(f"_{CAMERAS[0]}.jpg")
    )
    frames = [f for f in frames if all(os.path.exists(os.path.join(src, "images", f"{f:03d}_{c}.jpg")) for c in CAMERAS)]
    n_f, n_c = len(frames), len(CAMERAS)

    intr, c2e, boxes_px = [], [], []
    for c in CAMERAS:
        fx, fy, cx, cy = np.loadtxt(os.path.join(src, "intrinsics", f"{c}.txt")).reshape(-1)[:4]
        with Image.open(os.path.join(src, "images", f"{frames[0]:03d}_{c}.jpg")) as im:
            w0, h0 = im.size
        box = crop_box(w0, h0, w / h)
        sx, sy = w / box[2], h / box[3]
        intr.append(np.array([[fx * sx, 0, (cx - box[0]) * sx], [0, fy * sy, (cy - box[1]) * sy], [0, 0, 1]]))
        c2e.append(np.loadtxt(os.path.join(src, "extrinsics", f"{c}.txt")).reshape(4, 4))
        boxes_px.append(box)
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
            images[i, j] = _load(os.path.join(src, "images", f"{f:03d}_{c}.jpg"), boxes_px[j], hw, False)
            sp = os.path.join(src, "sky_masks", f"{f:03d}_{c}.png")
            if os.path.exists(sp):
                sky[i, j] = _load(sp, boxes_px[j], hw, True) > 127
            dp = os.path.join(src, "dynamic_masks", "all", f"{f:03d}_{c}.png")
            if os.path.exists(dp):
                dyn[i, j] = _load(dp, boxes_px[j], hw, True) > 127
            if pts is not None:
                depth[i, j] = project_depth(pts, c2e[j], intr[j], hw)
        lidar_world.append(pts @ ego[i, :3, :3].T + ego[i, :3, 3] if pts is not None else np.zeros((0, 3)))

    with open(os.path.join(src, "instances", "instances_info.json")) as fh:
        info = json.load(fh)
    slot = {f: i for i, f in enumerate(frames)}
    o2w, size, valid, count, cls = [], [], [], [], []
    for entry in info.values():
        ann = entry.get("frame_annotations", {})
        if not ann.get("frame_idx"):
            continue
        P, S = np.tile(np.eye(4), (n_f, 1, 1)), np.ones((n_f, 3))
        V, N = np.zeros(n_f, bool), np.zeros(n_f, np.int32)
        for f, pose, sz in zip(ann["frame_idx"], ann["obj_to_world"], ann["box_size"]):
            if int(f) not in slot:
                continue
            i = slot[int(f)]
            P[i], S[i], V[i] = np.asarray(pose).reshape(4, 4), np.asarray(sz), True
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
    hit = float((depth > 0).mean())
    sky_hit = float(((depth > 0) & (sky > 0)).sum() / max((depth > 0).sum(), 1))
    return f"{scene}: {n_f} frames, {o} objects, lidar px {hit:.3%}, lidar-on-sky {sky_hit:.2%}, crops {boxes_px}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/argoverse/processed/training")
    ap.add_argument("--out", default="cache/argoverse_160x240")
    ap.add_argument("--height", type=int, default=160)
    ap.add_argument("--width", type=int, default=240)
    ap.add_argument("--scenes", nargs="+", required=True)
    args = ap.parse_args()
    for s in args.scenes:
        print(process_scene(args.root, s, args.out, (args.height, args.width)), flush=True)


if __name__ == "__main__":
    main()
