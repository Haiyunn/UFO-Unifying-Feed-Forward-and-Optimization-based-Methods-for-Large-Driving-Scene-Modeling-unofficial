"""UFO training objective (Eq. 11 + supplementary Sec. 7) and evaluation metrics."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# Supplementary Sec. 7.
WEIGHTS = {
    "rgb": 1.0,          # MSE
    "lpips": 0.05,       # after the LPIPS warm-up
    "depth": 1.0,        # L1 vs LiDAR
    "flow": 0.005,       # MSE on forward flow
    "lifespan": 1e-4,    # mean 1/beta
    "sky_depth": 0.01,   # MSE in sky
    "sky_opacity": 0.1,  # L1 in sky
    "obj": 1.0,          # cross-entropy of the soft assignment
}

# Depth losses are computed on depth / DEPTH_SCALE so a 1.0 weight is commensurate
# with the photometric terms (the paper does not state its depth units).
DEPTH_SCALE = 80.0


class LPIPS(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        import lpips

        self.net = lpips.LPIPS(net="vgg", verbose=False).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)

    def forward(self, pred: Tensor, gt: Tensor) -> Tensor:
        return self.net(pred * 2 - 1, gt * 2 - 1).mean()


def render_losses(out: dict, gt: dict, lpips_fn: LPIPS | None) -> dict:
    """out / gt flattened over (views): rgb (V,3,H,W), depth/alpha (V,1,H,W)."""
    pred = out["rgb"]
    img = gt["images"]
    losses = {"rgb": F.mse_loss(pred, img)}
    if lpips_fn is not None:
        losses["lpips"] = lpips_fn(pred.clamp(0, 1), img)

    sky = gt["sky"].unsqueeze(1)
    lidar = gt["depth"].unsqueeze(1)
    valid = (lidar > 0) & (sky < 0.5)
    if valid.any():
        d = (out["depth"] - lidar).abs() / DEPTH_SCALE
        losses["depth"] = d[valid].mean()
    if (sky > 0.5).any():
        s = sky > 0.5
        losses["sky_opacity"] = out["alpha"][s].mean()
        losses["sky_depth"] = ((out["depth"] * out["alpha"] / DEPTH_SCALE)[s] ** 2).mean()
    return losses


def gaussian_losses(g: dict, flow: Tensor | None) -> dict:
    losses = {
        "lifespan": (1.0 / g["lifespan"]).mean(),
        "obj": F.cross_entropy(g["assign_logits"], g["assign_labels"]),
    }
    if flow is not None:
        losses["flow"] = (flow ** 2).sum(-1).mean()
    return losses


def total(losses: dict, use_lpips: bool) -> Tensor:
    return sum(WEIGHTS[k] * v for k, v in losses.items() if k != "lpips" or use_lpips)


# ------------------------------------------------------------------ metrics
# Same definitions as PointForward's eval (per-image PSNR, Gaussian-window SSIM,
# D-RMSE on LiDAR pixels) so numbers are comparable across the two codebases.


@torch.no_grad()
def psnr_per_image(pred: Tensor, gt: Tensor) -> Tensor:
    mse = ((pred.clamp(0, 1) - gt.clamp(0, 1)) ** 2).flatten(1).mean(1).clamp_min(1e-10)
    return -10.0 * torch.log10(mse)


@torch.no_grad()
def ssim_per_image(pred: Tensor, gt: Tensor, window_size: int = 11, sigma: float = 1.5) -> Tensor:
    pred, gt = pred.clamp(0, 1), gt.clamp(0, 1)
    c = pred.shape[1]
    coords = torch.arange(window_size, device=pred.device, dtype=pred.dtype) - (window_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    win = g.outer(g).expand(c, 1, window_size, window_size).contiguous()
    pad = window_size // 2
    mu_x = F.conv2d(pred, win, padding=pad, groups=c)
    mu_y = F.conv2d(gt, win, padding=pad, groups=c)
    sxx = F.conv2d(pred * pred, win, padding=pad, groups=c) - mu_x ** 2
    syy = F.conv2d(gt * gt, win, padding=pad, groups=c) - mu_y ** 2
    sxy = F.conv2d(pred * gt, win, padding=pad, groups=c) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))
    return m.flatten(1).mean(1)


@torch.no_grad()
def depth_rmse_per_image(pred: Tensor, lidar: Tensor) -> tuple[Tensor, Tensor]:
    """Returns per-image RMSE and a flag for images that have LiDAR returns."""
    m = (lidar > 0).float()
    n = m.flatten(1).sum(1)
    se = (((pred - lidar) ** 2) * m).flatten(1).sum(1)
    return (se / n.clamp_min(1)).sqrt(), n > 0
