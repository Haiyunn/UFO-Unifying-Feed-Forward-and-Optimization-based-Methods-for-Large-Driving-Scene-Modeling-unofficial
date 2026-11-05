"""Convention check: splat the target frame's own LiDAR (coloured from the image)
through the UFO renderer and compare to the image; also check box/lidar overlap.

    python -m tests.check_geometry --scene 005
"""
import argparse

import numpy as np
import torch

from ufo.data.dataset import UFOSequences, collate
from ufo.geometry import pixel_rays
from ufo.losses import psnr_per_image
from ufo.model import UFO, UFOConfig
from ufo.render import render_views

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="val")
args = ap.parse_args()
dev = torch.device("cuda")
ds = UFOSequences("cache/waymo_160x240", "splits/waymo_ds.json", args.split, seq_frames=20, train=False, eval_span=160)
item = collate([ds[0]])
tg = item["target"]
K, c2w, img, depth = tg["K"][0, 0].to(dev), tg["c2w"][0, 0].to(dev), tg["images"][0, 0].to(dev), tg["depth"][0, 0].to(dev)
# Back-project the LiDAR depth of all 3 cams into world points, colour them.
o, d = pixel_rays(K, c2w, 160, 240)
# pixel_rays returns unit dirs; LiDAR depth is z-depth -> convert via cos.
zdir = torch.linalg.inv(c2w)[:, None, None, :3, :3] @ d.unsqueeze(-1)
dist = depth / zdir[..., 2, 0].clamp_min(1e-6)
m = depth > 0
pts = (o + d * dist[..., None])[m]
col = img.permute(0, 2, 3, 1)[m]
n = pts.shape[0]
model = UFO(UFOConfig(grad_checkpoint=False)).to(dev)
gs = {"means": pts, "quats": torch.tensor([1.0, 0, 0, 0], device=dev).expand(n, 4),
      "scales": torch.full((n, 3), 0.03, device=dev), "opacity": torch.ones(n, device=dev), "rgb": col}
out = render_views(model, gs, torch.zeros(model.sky_param_count, device=dev), torch.zeros(3, 12, device=dev), K, c2w)
cover = out["alpha"][:, 0] > 0.5
err = ((out["rgb"] - img) ** 2).mean(1)[cover].mean()
print(f"lidar pts {n}, render coverage {cover.float().mean():.3f}, PSNR on covered px {-10*torch.log10(err):.2f}")
dm = (m & (out["alpha"][:, 0] > 0.5))
print(f"rendered-depth vs lidar abs err (median) {((out['depth'][:,0]-depth).abs()[dm]).median():.3f} m")
b = item["boxes"]
print(f"objects present {int(b['present'].sum())}, valid frames {int(b['valid'].sum())}")
# Box sanity: box centres of present objects should project in front of the front camera mostly.
