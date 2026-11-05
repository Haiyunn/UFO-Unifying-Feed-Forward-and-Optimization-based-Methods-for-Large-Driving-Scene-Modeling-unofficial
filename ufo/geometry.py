"""Camera / transform helpers shared by the model, renderer and loader.

Conventions: cameras are OpenCV (x right, y down, z forward); ``c2w`` maps
camera to the sequence world frame (the ego pose of the sequence's first frame);
all poses are 4x4 row-major homogeneous matrices.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def transform_points(mat: Tensor, pts: Tensor) -> Tensor:
    """Apply (..., 4, 4) to (..., 3)."""
    return (mat[..., :3, :3] @ pts.unsqueeze(-1)).squeeze(-1) + mat[..., :3, 3]


def pixel_rays(K: Tensor, c2w: Tensor, h: int, w: int, stride: int = 1) -> tuple[Tensor, Tensor]:
    """Ray origins / unit directions at pixel centres.

    K (V, 3, 3), c2w (V, 4, 4) -> origins, dirs (V, h/stride, w/stride, 3).
    With ``stride`` > 1 the rays go through patch centres.
    """
    dev, dt = K.device, K.dtype
    ys = torch.arange(h // stride, device=dev, dtype=dt) * stride + (stride - 1) / 2 + 0.5
    xs = torch.arange(w // stride, device=dev, dtype=dt) * stride + (stride - 1) / 2 + 0.5
    v, u = torch.meshgrid(ys, xs, indexing="ij")
    fx, fy = K[:, 0, 0, None, None], K[:, 1, 1, None, None]
    cx, cy = K[:, 0, 2, None, None], K[:, 1, 2, None, None]
    d = torch.stack([(u - cx) / fx, (v - cy) / fy, torch.ones_like(u - cx)], dim=-1)
    d = F.normalize(d, dim=-1)
    rot = c2w[:, None, None, :3, :3]
    dirs = (rot @ d.unsqueeze(-1)).squeeze(-1)
    origins = c2w[:, None, None, :3, 3].expand_as(dirs)
    return origins, dirs


def plucker(origins: Tensor, dirs: Tensor) -> Tensor:
    """Plucker coordinates (d, o x d), (..., 6)."""
    return torch.cat([dirs, torch.cross(origins, dirs, dim=-1)], dim=-1)


def contract(p: Tensor, scale: float = 20.0) -> Tensor:
    """Squash metric positions into the unit ball: p/|p| * |p|/(|p|+scale)."""
    n = p.norm(dim=-1, keepdim=True)
    return p / (n + scale)


def fourier(x: Tensor, n_freq: int, include_input: bool = True) -> Tensor:
    """NeRF-style Fourier features of inputs roughly in [-1, 1]."""
    freqs = 2.0 ** torch.arange(n_freq, device=x.device, dtype=x.dtype) * math.pi
    xb = x.unsqueeze(-1) * freqs
    out = torch.cat([xb.sin(), xb.cos()], dim=-1).flatten(-2)
    return torch.cat([x, out], dim=-1) if include_input else out


def fourier_dim(d_in: int, n_freq: int, include_input: bool = True) -> int:
    return d_in * (2 * n_freq + (1 if include_input else 0))


def sincos_1d(x: Tensor, dim: int, max_period: float = 100.0) -> Tensor:
    """Transformer-style sinusoidal embedding of a scalar (e.g. seconds)."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=x.device, dtype=torch.float32) / half)
    args = x.float().unsqueeze(-1) * freqs * 2 * math.pi
    return torch.cat([args.sin(), args.cos()], dim=-1)


def sincos_2d(h: int, w: int, dim: int, device) -> Tensor:
    """(h*w, dim) fixed 2-D positional embedding for patch grids."""
    ys = torch.arange(h, device=device, dtype=torch.float32)
    xs = torch.arange(w, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    ey = sincos_1d(yy.flatten(), dim // 2, max_period=4 * h)
    ex = sincos_1d(xx.flatten(), dim // 2, max_period=4 * w)
    return torch.cat([ey, ex], dim=-1)


def quat_to_mat(q: Tensor) -> Tensor:
    """(..., 4) wxyz -> (..., 3, 3)."""
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def mat_to_quat(m: Tensor) -> Tensor:
    """(..., 3, 3) rotation -> (..., 4) wxyz, Shepperd's method (largest-pivot branch)."""
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    tr = m00 + m11 + m22
    cand = torch.stack([tr, m00, m11, m22], dim=-1).argmax(-1)
    s0 = torch.sqrt((1 + tr).clamp_min(1e-6)) * 2
    s1 = torch.sqrt((1 + m00 - m11 - m22).clamp_min(1e-6)) * 2
    s2 = torch.sqrt((1 + m11 - m00 - m22).clamp_min(1e-6)) * 2
    s3 = torch.sqrt((1 + m22 - m00 - m11).clamp_min(1e-6)) * 2
    q0 = torch.stack([s0 / 4, (m21 - m12) / s0, (m02 - m20) / s0, (m10 - m01) / s0], -1)
    q1 = torch.stack([(m21 - m12) / s1, s1 / 4, (m01 + m10) / s1, (m02 + m20) / s1], -1)
    q2 = torch.stack([(m02 - m20) / s2, (m01 + m10) / s2, s2 / 4, (m12 + m21) / s2], -1)
    q3 = torch.stack([(m10 - m01) / s3, (m02 + m20) / s3, (m12 + m21) / s3, s3 / 4], -1)
    q = torch.stack([q0, q1, q2, q3], dim=-2)
    q = torch.gather(q, -2, cand[..., None, None].expand(cand.shape + (1, 4))).squeeze(-2)
    return F.normalize(q, dim=-1)


def orthonormalize(m: Tensor) -> Tensor:
    """Gram-Schmidt the first two columns of (..., 3, 3) into a rotation."""
    a, b = m[..., :, 0], m[..., :, 1]
    e1 = F.normalize(a, dim=-1)
    e2 = F.normalize(b - (e1 * b).sum(-1, keepdim=True) * e1, dim=-1)
    e3 = torch.cross(e1, e2, dim=-1)
    return torch.stack([e1, e2, e3], dim=-1)


def quat_mul(a: Tensor, b: Tensor) -> Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )
