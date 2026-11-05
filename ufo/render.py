"""Render decoded UFO scenes at arbitrary target times / cameras with gsplat."""

from __future__ import annotations

import torch
from gsplat import rasterization
from torch import Tensor

from .geometry import pixel_rays, transform_points
from .model import UFO, SceneMemory, affine_matrix, rotate_quats


def gaussians_at(model: UFO, g: dict, mem: SceneMemory, boxes: dict, b: int, frame: int, time: float) -> dict:
    """Move canonical Gaussians to ``frame`` (Eq. 6) and apply lifespans (Eq. 9)."""
    assign = g["assign_logits"].softmax(-1)
    f = torch.tensor([frame], device=mem.pos.device)
    T = model.motion(assign, mem, boxes, b, f)[0]                                       # (N, 4, 4)
    means = transform_points(T[:, None], g["means"])
    quats = rotate_quats(T[:, None], g["quats"])
    opacity = g["opacity"]
    if model.cfg.use_lifespan:
        opacity = opacity * torch.exp(-((time - g["t0"]) ** 2) / (2 * g["lifespan"] ** 2))
    return {"means": means, "quats": quats, "opacity": opacity, "scales": g["scales"], "rgb": g["rgb"]}


def render_views(
    model: UFO, gs: dict, sky_params: Tensor, affine: Tensor, K: Tensor, c2w: Tensor, cams: Tensor | None = None
) -> dict:
    """Render one time instant for V cameras.

    gs: Gaussians already moved to that time; K (V, 3, 3); c2w (V, 4, 4);
    cams (V,) camera ids indexing the affine tokens (default 0..V-1).
    Returns rgb (V, 3, H, W), depth (V, 1, H, W), alpha (V, 1, H, W).
    """
    h, w = model.cfg.height, model.cfg.width
    v = K.shape[0]
    cams = torch.arange(v, device=K.device) if cams is None else cams
    viewmats = torch.linalg.inv(c2w)
    colors, meta_alpha, _ = rasterization(
        means=gs["means"].reshape(-1, 3).float(),
        quats=gs["quats"].reshape(-1, 4).float(),
        scales=gs["scales"].reshape(-1, 3).float(),
        opacities=gs["opacity"].reshape(-1).float(),
        colors=gs["rgb"].reshape(-1, 3).float(),
        viewmats=viewmats.float(),
        Ks=K.float(),
        width=w,
        height=h,
        near_plane=0.1,
        far_plane=1000.0,
        render_mode="RGB+ED",
        packed=False,
    )
    rgb, depth = colors[..., :3], colors[..., 3:4]
    alpha = meta_alpha
    _, dirs = pixel_rays(K, c2w, h, w)
    sky = model.sky_color(sky_params, dirs)
    rgb = rgb + (1.0 - alpha) * sky
    A, bvec = affine_matrix(affine[cams])
    rgb = torch.einsum("vij,vhwj->vhwi", A, rgb) + bvec[:, None, None]
    return {
        "rgb": rgb.permute(0, 3, 1, 2),
        "depth": depth.permute(0, 3, 1, 2),
        "alpha": alpha.permute(0, 3, 1, 2),
    }


def render_targets(model: UFO, mem: SceneMemory, sky_params: Tensor, affine: Tensor, batch: dict, b: int,
                   views: dict, g: dict | None = None) -> tuple[dict, dict]:
    """Render every target frame of sequence ``b``; returns stacked (T, C, ...) outputs."""
    boxes = batch["boxes"]
    if g is None:
        g = model.decode(mem, boxes, b)
    outs = []
    for j in range(views["frame"].shape[1]):
        gs = gaussians_at(model, g, mem, boxes, b, int(views["frame"][b, j]), float(views["time"][b, j]))
        outs.append(render_views(model, gs, sky_params[b], affine[b], views["K"][b, j], views["c2w"][b, j]))
    return {k: torch.stack([o[k] for o in outs]) for k in outs[0]}, g
