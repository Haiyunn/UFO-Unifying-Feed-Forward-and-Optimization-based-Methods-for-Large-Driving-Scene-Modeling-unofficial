"""UFO: recurrent scene-token reconstruction (Tan et al., arXiv 2602.20943).

Faithful to the paper where it is specific, with the remaining choices made
explicit (see docs/IMPLEMENTATION.md):

* Scene tokens s = (xyz in the sequence world frame, f in R^768)       Sec. 3.1
* Image tokens: 8x8 patches of RGB + Plucker rays, linear-projected,
  plus 2-D patch position, camera and time embeddings                  Sec. 3.1
* Box tokens: Fourier embedding of box centre + 8 corners, 2-layer MLP  Sec. 3.1
* Auxiliary tokens: sky tokens -> sky-MLP weights; one affine token per
  camera -> 3x3 + 3 colour transform; carried across steps            Sec. 3.1
* T_update: 12-layer ViT encoder (768-d, 12 heads)                     Sec. 3.2
* Visibility filter: tokens inside any current camera frustum, nearest
  K = 3600 to the camera centre                                        Eq. 2
* Local frame: selected tokens / rays are expressed in the current
  front-camera frame before the update                                 Sec. 3.2
* Update: refined visible tokens replace their old versions, every image
  token output becomes a new scene token                               Eq. 3-4
* Decoding: per-token 2-layer MLP -> M Gaussians; soft object assignment
  A = softmax(q(s).k(B)) over {static} U boxes, motion T = sum_i A_i T_i  Eq. 5-7
* Lifespan opacity sigma(t) = sigma exp(-(t-t0)^2 / 2 beta^2)           Eq. 9

Two extensions beyond the paper, both off by default:

* ``depth_prior``: a precomputed dense metric depth (InfiniDepth prompted with
  the frame's own LiDAR, :mod:`ufo.data.depth_prior`) enters each image token as
  a 10th per-pixel channel, and a new token is placed at its patch's prior
  depth times a learned residual (zero-init), instead of a distance learned
  from scratch.
* ``pixel_aligned``: a token decodes one Gaussian per pixel of its patch
  (M = patch^2).  Gaussian j is anchored at pixel j's ray times its prior depth
  (stored per token as offsets from the token position, rotated with the token
  when it is refined), the decoder predicting only a small residual and a
  pixel-footprint-relative scale.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from .geometry import (
    contract,
    fourier,
    fourier_dim,
    orthonormalize,
    pixel_rays,
    plucker,
    sincos_1d,
    sincos_2d,
    transform_points,
)

# Box corner signs in the box-local frame (x length, y width, z height).
_CORNERS = torch.tensor(
    [[sx, sy, sz] for sx in (-0.5, 0.5) for sy in (-0.5, 0.5) for sz in (-0.5, 0.5)], dtype=torch.float32
)

# Token-type ids.
_AUX, _BOX, _SCENE, _IMAGE = 0, 1, 2, 3


@dataclass
class UFOConfig:
    height: int = 160
    width: int = 240
    num_cams: int = 3
    patch: int = 8
    dim: int = 768
    depth: int = 12
    heads: int = 12
    mlp_ratio: float = 4.0
    visible_k: int = 3600
    max_objects: int = 32
    gaussians_per_token: int = 16
    num_sky_tokens: int = 4
    sky_hidden: int = 32
    sky_freq: int = 4
    pos_freq: int = 8
    box_freq: int = 6
    assign_dim: int = 64
    frustum_margin: float = 0.1
    grad_checkpoint: bool = True
    # Ablation switches (paper Table 2).
    use_scene_tokens: bool = True      # (2) w/o scene tokens
    use_refinement: bool = True        # (3) w/o refinement
    use_lifespan: bool = True          # (4) w/o lifespan
    use_boxes: bool = True             # (5) w/o bounding box
    # Extensions (see module docstring).
    depth_prior: bool = False
    prior_residual: float = 1.0        # |log residual| bound on the prior depth
    pixel_aligned: bool = False
    # Match M=16's per-token opacity mass and coverage at init when M != 16
    # (opacity x 16/M, scale x sqrt(16/M)); False keeps the original init.
    init_density_match: bool = False


# ----------------------------------------------------------------- blocks


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: Tensor, key_mask: Tensor | None) -> Tensor:
        b, n, d = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        attn_mask = None if key_mask is None else key_mask[:, None, None, :]
        y = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], attn_mask=attn_mask)
        x = x + self.proj(y.transpose(1, 2).reshape(b, n, d))
        return x + self.mlp(self.norm2(x))


def mlp2(d_in: int, d_hidden: int, d_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, d_hidden), nn.GELU(), nn.Linear(d_hidden, d_out))


# ----------------------------------------------------------------- memory


@dataclass
class SceneMemory:
    """Scene tokens of one sequence.  All tensors share the leading N."""

    pos: Tensor        # (N, 3) world position at reference frame f_ref
    feat: Tensor       # (N, D)
    f_ref: Tensor      # (N,) frame index (sequence-relative) the position refers to
    t_create: Tensor   # (N,) creation time in seconds (lifespan centre t0)
    footprint: Tensor  # (N,) metric size of one patch at the token's distance
    anchor: Tensor     # (N, A, 3) pixel-aligned Gaussian anchors minus pos (A = 0 when off)

    @property
    def n(self) -> int:
        return self.pos.shape[0]

    @staticmethod
    def empty(dim: int, device, anchors: int = 0) -> "SceneMemory":
        z = torch.zeros(0, device=device)
        return SceneMemory(torch.zeros(0, 3, device=device), torch.zeros(0, dim, device=device),
                           z.long(), z, z, torch.zeros(0, anchors, 3, device=device))

    def replace_and_append(self, idx: Tensor, refined: "SceneMemory", new: "SceneMemory") -> "SceneMemory":
        """S_t = (S_{t-1} minus V) U V' U dS  (Eq. 4), out of place for autograd."""
        keep = torch.ones(self.n, dtype=torch.bool, device=self.pos.device)
        keep[idx] = False
        return SceneMemory.cat([self.select(keep), refined, new])

    def select(self, idx: Tensor) -> "SceneMemory":
        return SceneMemory(*[getattr(self, k)[idx] for k in _MEMORY_FIELDS])

    @staticmethod
    def cat(parts: list["SceneMemory"]) -> "SceneMemory":
        return SceneMemory(*[torch.cat([getattr(p, k) for p in parts]) for k in _MEMORY_FIELDS])


_MEMORY_FIELDS = ("pos", "feat", "f_ref", "t_create", "footprint", "anchor")


# ----------------------------------------------------------------- model


class UFO(nn.Module):
    def __init__(self, cfg: UFOConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d, p = cfg.dim, cfg.patch
        self.gh, self.gw = cfg.height // p, cfg.width // p

        # Image tokens: per-pixel RGB + Plucker (9 ch, + prior log-depth) in 8x8 patches.
        if cfg.pixel_aligned and cfg.gaussians_per_token != p * p:
            raise ValueError("pixel_aligned needs gaussians_per_token == patch * patch")
        if cfg.pixel_aligned and not cfg.depth_prior:
            raise ValueError("pixel_aligned anchors need depth_prior")
        self.in_ch = 9 + int(cfg.depth_prior)
        self.patch_embed = nn.Linear(p * p * self.in_ch, d)
        self.cam_embed = nn.Parameter(torch.zeros(cfg.num_cams, d))
        self.image_time = mlp2(64, d, d)
        self.register_buffer("patch_pos", sincos_2d(self.gh, self.gw, d, "cpu"), persistent=False)
        self.type_embed = nn.Parameter(torch.zeros(4, d))

        # Scene-token re-entry: feature + local position + relative times.
        self.scene_pos = mlp2(fourier_dim(3, cfg.pos_freq), d, d)
        self.scene_time = mlp2(2 * 64, d, d)

        # Box tokens (Sec. 3.1, 2-layer MLP over Fourier(centre, corners)).
        self.box_encoder = mlp2(fourier_dim(27 + 3, cfg.box_freq), d, d)
        self.slot_embed = nn.Parameter(torch.randn(cfg.max_objects, d) * 0.02)

        # Auxiliary tokens carried across the recurrence.
        self.sky_tokens = nn.Parameter(torch.randn(cfg.num_sky_tokens, d) * 0.02)
        self.affine_tokens = nn.Parameter(torch.randn(cfg.num_cams, d) * 0.02)

        self.blocks = nn.ModuleList([Block(d, cfg.heads, cfg.mlp_ratio) for _ in range(cfg.depth)])
        self.norm = nn.LayerNorm(d)

        # Heads.
        self.depth_head = nn.Linear(d, 1)            # new tokens: distance along patch ray
        self.delta_head = nn.Linear(d, 3)            # refined tokens: position update
        self.gauss_dec = mlp2(d, d, cfg.gaussians_per_token * 15)
        sky_in = fourier_dim(3, cfg.sky_freq)
        self.sky_param_count = sky_in * cfg.sky_hidden + cfg.sky_hidden + cfg.sky_hidden * 3 + 3
        self.sky_head = nn.Linear(d * cfg.num_sky_tokens, self.sky_param_count)
        self.affine_head = nn.Linear(d, 12)

        # Object assignment (Eq. 5): q from scene tokens, k from box embeddings.
        a = cfg.assign_dim
        self.assign_q = mlp2(d, d // 2, a)
        self.assign_k = mlp2(fourier_dim(3, cfg.box_freq), 128, a)
        self.slot_k = nn.Linear(d, a)
        self.static_key = nn.Parameter(torch.randn(a) * 0.02)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.depth_head.weight)
        # With a depth prior the head is a log residual on it, zero at init.
        nn.init.constant_(self.depth_head.bias, 0.0 if self.cfg.depth_prior else math.log(15.0))
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.zeros_(self.affine_head.weight)
        nn.init.zeros_(self.affine_head.bias)
        nn.init.normal_(self.sky_head.weight, std=1e-3)
        self._init_gauss_bias()

    def _init_gauss_bias(self) -> None:
        m = self.cfg.gaussians_per_token
        bias = torch.zeros(m, 15)
        bias[:, 3:6] = 0.0          # log scale relative to footprint
        bias[:, 6] = 1.0            # quaternion w
        bias[:, 10] = -1.0          # opacity logit
        bias[:, 14] = 4.0           # lifespan pre-softplus -> ~4 s
        if self.cfg.init_density_match and m != 16:
            ratio = 16.0 / m
            o = torch.sigmoid(torch.tensor(-1.0)) * ratio
            bias[:, 10] = torch.log(o / (1 - o))
            bias[:, 3:6] = 0.5 * math.log(ratio)
        last = self.gauss_dec[-1]
        nn.init.normal_(last.weight, std=1e-3)
        with torch.no_grad():
            last.bias.copy_(bias.flatten())

    # ------------------------------------------------------------ tokens

    def image_tokens(self, images: Tensor, K: Tensor, c2w_local: Tensor, t_now: Tensor,
                     prior: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        """images (B, C, 3, H, W), K (B, C, 3, 3), c2w_local (B, C, 4, 4),
        prior (B, C, H, W) metric depth (0 = none), required with ``depth_prior``.

        Returns tokens (B, C*h*w, D) and patch-centre ray origins / dirs in the
        local frame (B, C*h*w, 3) used to place new scene tokens.
        """
        b, c, _, h, w = images.shape
        p = self.cfg.patch
        o, dvec = pixel_rays(K.flatten(0, 1), c2w_local.flatten(0, 1), h, w)
        pl = plucker(o, dvec).permute(0, 3, 1, 2)                       # (BC, 6, H, W)
        chans = [images.flatten(0, 1) * 2 - 1, pl]
        if self.cfg.depth_prior:
            pr = prior.flatten(0, 1).unsqueeze(1)
            chans.append(torch.where(pr > 0, (pr.clamp(0.5, 300.0).log() - math.log(15.0)) / 2.0,
                                     torch.zeros_like(pr)))
        x = torch.cat(chans, dim=1)                                      # (BC, 9|10, H, W)
        ch = self.in_ch
        x = x.reshape(b * c, ch, self.gh, p, self.gw, p).permute(0, 2, 4, 3, 5, 1).reshape(b * c, self.gh * self.gw, p * p * ch)
        tok = self.patch_embed(x) + self.patch_pos.to(x.dtype)
        tok = tok.reshape(b, c, -1, self.cfg.dim) + self.cam_embed[None, :c, None]
        tok = tok + self.image_time(sincos_1d(t_now, 64, 64.0))[:, None, None]
        po, pd = pixel_rays(K.flatten(0, 1), c2w_local.flatten(0, 1), h, w, stride=p)
        return (tok.flatten(1, 2) + self.type_embed[_IMAGE],
                po.reshape(b, -1, 3), pd.reshape(b, -1, 3))

    def box_tokens(self, pose: Tensor, size: Tensor, world_to_local: Tensor) -> Tensor:
        """pose (B, O, 4, 4), size (B, O, 3) at the current frame -> (B, O, D)."""
        corners = _CORNERS.to(pose)[None, None] * size[:, :, None]                      # (B, O, 8, 3)
        corners_w = transform_points(pose[:, :, None], corners)
        centre_w = pose[:, :, :3, 3]
        pts = torch.cat([centre_w[:, :, None], corners_w], dim=2)                      # (B, O, 9, 3)
        pts_l = transform_points(world_to_local[:, None, None], pts)
        feat = torch.cat([contract(pts_l).flatten(2), size / 10.0], dim=-1)
        return self.box_encoder(fourier(feat, self.cfg.box_freq)) + self.slot_embed[None] + self.type_embed[_BOX]

    def scene_tokens(self, mem: SceneMemory, pos_now: Tensor, world_to_local: Tensor, t_now: float, f_now: int) -> Tensor:
        pos_l = transform_points(world_to_local, pos_now)
        dt = torch.stack([(mem.t_create - t_now), (mem.f_ref.float() - f_now) / 10.0], dim=-1)
        temb = torch.cat([sincos_1d(dt[:, 0], 64, 64.0), sincos_1d(dt[:, 1], 64, 64.0)], dim=-1)
        return (mem.feat + self.scene_pos(fourier(contract(pos_l), self.cfg.pos_freq))
                + self.scene_time(temb) + self.type_embed[_SCENE])

    # ------------------------------------------------------------ assignment / motion

    def assignment(self, mem: SceneMemory, boxes: dict, b: int) -> tuple[Tensor, Tensor]:
        """Logits of Eq. 5 over {static} U objects, and the Eq. 8 labels.

        Returns logits (N, 1 + O) and labels (N,) where label o + 1 means the
        token lies inside object o's box at its reference frame and 0 means
        background.  Only token/box pairs within a margin of the box are scored
        by the key MLP; farther pairs are masked to -inf, which keeps the cost
        linear in N instead of N x O.
        """
        a = self.cfg.assign_dim
        q = self.assign_q(mem.feat)                                                    # (N, a)
        n = mem.n
        static = (q @ self.static_key)[:, None] / math.sqrt(a)
        labels = torch.zeros(n, dtype=torch.long, device=q.device)
        if not self.cfg.use_boxes:
            return static, labels
        o = boxes["pose"].shape[1]
        with torch.no_grad():
            inv = boxes["pose_inv"][b][:, mem.f_ref]                                   # (O, N, 4, 4)
            size = boxes["size"][b][:, mem.f_ref]                                      # (O, N, 3)
            rel_m = transform_points(inv, mem.pos.detach()[None].expand(o, n, 3))
            present = boxes["present"][b][:, None]
            near = (rel_m.abs() <= size * 0.75 + 1.0).all(-1) & present                # (O, N)
            inside = (rel_m.abs() <= size * 0.5).all(-1) & present & boxes["valid"][b][:, mem.f_ref]
            has = inside.any(0)
            labels[has] = inside.float().argmax(0)[has] + 1
            oi, ni = near.nonzero(as_tuple=True)
        obj = q.new_full((n, o), -1e4)
        if oi.numel():
            # Token geometry inside the box; detached so L_obj cannot drag token positions.
            rel = rel_m[oi, ni] / size[oi, ni].clamp_min(0.1)
            k = self.assign_k(fourier(rel.clamp(-2, 2) / 2, self.cfg.box_freq)) + self.slot_k(self.slot_embed)[oi]
            obj = obj.index_put((ni, oi), (k * q[ni]).sum(-1) / math.sqrt(a))
        return torch.cat([static, obj], dim=1), labels

    def motion(self, assign: Tensor, mem: SceneMemory, boxes: dict, b: int, f_target: Tensor) -> Tensor:
        """Per-token motion from the token's reference frame to each target frame.

        Eq. 6-7 blend absolute box poses and invert the blend; with a static
        (identity) column and objects facing ~180 deg from the sequence frame
        that blend is near-singular.  We blend the relative motions instead,
            T = A_static I + sum_o A_o  T_{t1,o} T_{t0,o}^{-1},
        which equals Eq. 6-7 for one-hot assignments and never inverts a blend.

        assign (N, 1+O) probabilities, f_target (F,) -> (F, N, 4, 4).
        """
        n = mem.n
        eye = torch.eye(4, device=assign.device)
        out = eye.expand(len(f_target), n, 4, 4) * assign[None, :, :1, None]
        if assign.shape[1] == 1:
            return eye.expand(len(f_target), n, 4, 4)
        pose = boxes["pose"][b]                                                        # (O, Fseq, 4, 4)
        rel = pose[:, f_target][:, :, None] @ boxes["pose_inv"][b][:, mem.f_ref][:, None]  # (O, F, N, 4, 4)
        return out + torch.einsum("no,ofnij->fnij", assign[:, 1:], rel)

    # ------------------------------------------------------------ recurrence

    def _run_transformer(self, x: Tensor, key_mask: Tensor) -> Tensor:
        for blk in self.blocks:
            if self.cfg.grad_checkpoint and self.training:
                x = checkpoint(blk, x, key_mask, use_reentrant=False)
            else:
                x = blk(x, key_mask)
        return self.norm(x)

    def visible(self, pos: Tensor, K: Tensor, c2w: Tensor) -> Tensor:
        """Indices of the nearest K tokens inside any camera frustum (Eq. 2)."""
        if pos.shape[0] == 0:
            return torch.zeros(0, dtype=torch.long, device=pos.device)
        h, w, m = self.cfg.height, self.cfg.width, self.cfg.frustum_margin
        w2c = torch.linalg.inv(c2w)                                                    # (C, 4, 4)
        pc = transform_points(w2c[:, None], pos[None])                                 # (C, N, 3)
        z = pc[..., 2]
        uv = (K[:, None] @ pc.unsqueeze(-1)).squeeze(-1)
        u, v = uv[..., 0] / z.clamp_min(1e-6), uv[..., 1] / z.clamp_min(1e-6)
        inside = (z > 0.1) & (u > -m * w) & (u < (1 + m) * w) & (v > -m * h) & (v < (1 + m) * h)
        inside = inside.any(0)
        dist = (pos - c2w[0, :3, 3]).norm(dim=-1)
        dist = torch.where(inside, dist, torch.full_like(dist, float("inf")))
        k = min(self.cfg.visible_k, int(inside.sum()))
        if k == 0:
            return torch.zeros(0, dtype=torch.long, device=pos.device)
        return torch.topk(dist, k, largest=False).indices

    def forward_sequence(self, batch: dict) -> tuple[list[SceneMemory], Tensor, Tensor]:
        """Run Algorithm 1 over the context frames of every sequence in the batch.

        Returns the final memories, and the sky / affine parameters
        (B, P_sky), (B, C, 12).
        """
        cfg = self.cfg
        ctx = batch["context"]
        boxes = batch["boxes"]
        b, t_steps, c = ctx["images"].shape[:3]
        dev = ctx["images"].device
        n_anchor = cfg.patch * cfg.patch if cfg.pixel_aligned else 0
        mems = [SceneMemory.empty(cfg.dim, dev, n_anchor) for _ in range(b)]
        boxes.setdefault("pose_inv", torch.linalg.inv(boxes["pose"]))
        aux = torch.cat([self.sky_tokens, self.affine_tokens])[None].expand(b, -1, -1) + self.type_embed[_AUX]
        n_aux = aux.shape[1]

        for t in range(t_steps):
            c2w = ctx["c2w"][:, t]                                                     # (B, C, 4, 4)
            local_from_world = torch.linalg.inv(c2w[:, 0])                            # front-camera frame
            c2w_local = local_from_world[:, None] @ c2w
            f_now = ctx["frame"][:, t]
            t_now = ctx["time"][:, t]

            prior = ctx["prior"][:, t] if cfg.depth_prior else None
            img_tok, ray_o, ray_d = self.image_tokens(ctx["images"][:, t], ctx["K"][:, t], c2w_local, t_now, prior)

            if cfg.use_boxes:
                bp = boxes["pose"][torch.arange(b), :, f_now]                         # (B, O, 4, 4)
                bs = boxes["size"][torch.arange(b), :, f_now]
                box_tok = self.box_tokens(bp, bs, local_from_world)
                box_mask = boxes["valid"][torch.arange(b), :, f_now] & boxes["present"]
            else:
                box_tok = img_tok.new_zeros(b, 0, cfg.dim)
                box_mask = torch.zeros(b, 0, dtype=torch.bool, device=dev)

            # Visibility filtering + local frame, per sequence.
            sel, scene_in, pos_now_sel, rot_now_sel = [], [], [], []
            for i in range(b):
                mem = mems[i]
                if not cfg.use_scene_tokens or mem.n == 0:
                    idx = torch.zeros(0, dtype=torch.long, device=dev)
                    pos_now, T_now = mem.pos, None
                else:
                    T_now = self._motion_now(mem, boxes, i, int(f_now[i]))
                    pos_now = mem.pos if T_now is None else transform_points(T_now, mem.pos)
                    idx = self.visible(pos_now.detach(), ctx["K"][i, t], c2w[i])
                sel.append(idx)
                rot_now_sel.append(None if T_now is None or not idx.numel() else orthonormalize(T_now[idx, :3, :3]))
                if idx.numel():
                    pos_now_sel.append(pos_now[idx])
                    scene_in.append(self.scene_tokens(mem.select(idx), pos_now[idx], local_from_world[i],
                                                      float(t_now[i]), int(f_now[i])))
                else:
                    pos_now_sel.append(mem.pos[:0])
                    scene_in.append(img_tok.new_zeros(0, cfg.dim))
            n_scene = max(int(s.shape[0]) for s in scene_in)
            scene_pad = img_tok.new_zeros(b, n_scene, cfg.dim)
            scene_mask = torch.zeros(b, n_scene, dtype=torch.bool, device=dev)
            for i, s in enumerate(scene_in):
                scene_pad[i, : s.shape[0]] = s
                scene_mask[i, : s.shape[0]] = True

            x = torch.cat([aux, box_tok, scene_pad, img_tok], dim=1)
            key_mask = torch.cat([
                torch.ones(b, n_aux, dtype=torch.bool, device=dev), box_mask, scene_mask,
                torch.ones(b, img_tok.shape[1], dtype=torch.bool, device=dev)], dim=1)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                y = self._run_transformer(x, key_mask)
            y = y.float()

            n_box = box_tok.shape[1]
            aux = y[:, :n_aux]
            y_scene = y[:, n_aux + n_box: n_aux + n_box + n_scene]
            y_img = y[:, n_aux + n_box + n_scene:]

            # New tokens: one per image patch, placed along its ray.
            anchors = None
            if cfg.depth_prior:
                dist, anchors = self._place_on_prior(y_img, prior, ctx["K"][:, t], c2w_local, ray_o, ray_d)
            else:
                dist = self.depth_head(y_img).squeeze(-1).clamp(-2.0, 6.0).exp()       # (B, P) metres
            f_mean = ctx["K"][:, t, :, 0, 0].mean(-1, keepdim=True)                     # (B, 1)
            for i in range(b):
                world_from_local = c2w[i, 0]
                p_local = ray_o[i] + ray_d[i] * dist[i, :, None]
                if anchors is not None:
                    anchor = (world_from_local[:3, :3] @ anchors[i].unsqueeze(-1)).squeeze(-1)
                else:
                    anchor = p_local.new_zeros(p_local.shape[0], 0, 3)
                new = SceneMemory(
                    pos=transform_points(world_from_local, p_local),
                    feat=y_img[i],
                    f_ref=torch.full_like(dist[i], int(f_now[i]), dtype=torch.long),
                    t_create=torch.full_like(dist[i], float(t_now[i])),
                    footprint=dist[i] * cfg.patch / f_mean[i],
                    anchor=anchor,
                )
                idx = sel[i]
                if not cfg.use_scene_tokens:
                    mems[i] = SceneMemory.cat([mems[i], new])
                    continue
                if idx.numel() and cfg.use_refinement:
                    old = mems[i].select(idx)
                    ys = y_scene[i, : idx.numel()]
                    p_now = pos_now_sel[i]
                    d_cam = (p_now - world_from_local[:3, 3]).norm(dim=-1, keepdim=True)
                    delta_l = torch.tanh(self.delta_head(ys)) * (0.1 * d_cam + 0.2)
                    delta_w = (world_from_local[:3, :3] @ delta_l.unsqueeze(-1)).squeeze(-1)
                    # Anchors are world-frame offsets: they turn with the token's motion.
                    rot = rot_now_sel[i]
                    anchor = old.anchor if rot is None or old.anchor.shape[1] == 0 else \
                        torch.einsum("nij,naj->nai", rot, old.anchor)
                    refined = SceneMemory(
                        pos=p_now + delta_w,
                        feat=ys,
                        f_ref=torch.full_like(old.f_ref, int(f_now[i])),
                        t_create=old.t_create,
                        footprint=torch.minimum(old.footprint, d_cam.squeeze(-1) * cfg.patch / f_mean[i]),
                        anchor=anchor,
                    )
                    mems[i] = mems[i].replace_and_append(idx, refined, new)
                else:
                    empty = mems[i].select(idx[:0])
                    mems[i] = mems[i].replace_and_append(idx[:0], empty, new)

        sky_params = self.sky_head(aux[:, : cfg.num_sky_tokens].flatten(1))
        affine = self.affine_head(aux[:, cfg.num_sky_tokens:])
        return mems, sky_params, affine

    def _motion_now(self, mem: SceneMemory, boxes: dict, b: int, f: int) -> Tensor | None:
        """(N, 4, 4) motion of each token to frame f under its current soft
        assignment, or None without boxes (identity)."""
        if not self.cfg.use_boxes:
            return None
        assign = self.assignment(mem, boxes, b)[0].softmax(-1)
        return self.motion(assign, mem, boxes, b, torch.tensor([f], device=mem.pos.device))[0]

    def _positions_at(self, mem: SceneMemory, boxes: dict, b: int, f: int) -> Tensor:
        """Token positions moved to frame f with their current soft assignment."""
        T = self._motion_now(mem, boxes, b, f)
        return mem.pos if T is None else transform_points(T, mem.pos)

    def _place_on_prior(self, y_img: Tensor, prior: Tensor, K: Tensor, c2w_local: Tensor,
                        ray_o: Tensor, ray_d: Tensor) -> tuple[Tensor, Tensor | None]:
        """Distance of each new token along its patch-centre ray from the depth
        prior, and (pixel_aligned) its per-pixel anchors in the local frame.

        The prior is z-depth; a patch's depth is the median of its valid pixels
        (15 m when it has none), scaled by exp(r) with r the bounded learned
        log residual (zero at init), then converted to distance along the ray.

        Returns dist (B, P) and anchors (B, P, p*p, 3) or None.
        """
        cfg = self.cfg
        p = cfg.patch
        b, c, h, w = prior.shape
        z = prior.reshape(b, c, self.gh, p, self.gw, p).permute(0, 1, 2, 4, 3, 5).reshape(b, c * self.gh * self.gw, p * p)
        valid = z > 0
        z_patch = torch.nanmedian(torch.where(valid, z, torch.full_like(z, float("nan"))), dim=-1).values
        z_patch = torch.where(valid.any(-1), z_patch, torch.full_like(z_patch, 15.0)).clamp(0.5, 300.0)
        bound = cfg.prior_residual
        scale = (bound * torch.tanh(self.depth_head(y_img).squeeze(-1) / bound)).exp()          # (B, P)
        # Optical axis of each patch's camera in the local frame, for z -> ray distance.
        fwd = c2w_local[:, :, None, :3, 2].expand(b, c, self.gh * self.gw, 3).reshape(b, -1, 3)
        dist = z_patch * scale / (ray_d * fwd).sum(-1).clamp_min(1e-3)
        if not cfg.pixel_aligned:
            return dist, None
        po, pd = pixel_rays(K.flatten(0, 1), c2w_local.flatten(0, 1), h, w)                   # (BC, H, W, 3)
        pd = pd.reshape(b, c, self.gh, p, self.gw, p, 3).permute(0, 1, 2, 4, 3, 5, 6).reshape(b, -1, p * p, 3)
        po = po.reshape(b, c, self.gh, p, self.gw, p, 3).permute(0, 1, 2, 4, 3, 5, 6).reshape(b, -1, p * p, 3)
        z_pix = torch.where(valid, z.clamp(0.5, 300.0), z_patch[..., None]) * scale[..., None]
        p_pix = po + pd * (z_pix / (pd * fwd[:, :, None]).sum(-1).clamp_min(1e-3))[..., None]
        centre = ray_o + ray_d * dist[..., None]
        return dist, p_pix - centre[:, :, None]

    # ------------------------------------------------------------ decoding

    def decode(self, mem: SceneMemory, boxes: dict, b: int) -> dict:
        """Scene tokens -> canonical Gaussians (at each token's reference frame)."""
        cfg = self.cfg
        m = cfg.gaussians_per_token
        raw = self.gauss_dec(mem.feat).reshape(mem.n, m, 15)
        fp = mem.footprint.clamp(0.01, 20.0)[:, None, None]
        if cfg.pixel_aligned:
            # One Gaussian per pixel: anchored on the prior, sized by the pixel footprint.
            fp = fp / cfg.patch
            offset = mem.anchor + torch.tanh(raw[..., 0:3]) * fp * 1.5
            scale = (raw[..., 3:6].clamp(-6, 3).exp() * fp * 0.7).clamp(1e-4, 50.0)
        else:
            offset = torch.tanh(raw[..., 0:3]) * fp * 1.5
            scale = (raw[..., 3:6].clamp(-6, 3).exp() * fp * 0.5).clamp(1e-4, 50.0)
        quat = F.normalize(raw[..., 6:10], dim=-1)
        opacity = torch.sigmoid(raw[..., 10])
        rgb = torch.sigmoid(raw[..., 11:14])
        lifespan = F.softplus(raw[..., 14]) + 0.05
        logits, labels = self.assignment(mem, boxes, b)
        return {
            "means": mem.pos[:, None] + offset,
            "scales": scale,
            "quats": quat,
            "opacity": opacity,
            "rgb": rgb,
            "lifespan": lifespan,
            "t0": mem.t_create[:, None].expand(-1, m),
            "assign_logits": logits,
            "assign_labels": labels,
        }

    def sky_color(self, params: Tensor, dirs: Tensor) -> Tensor:
        """Hyper-network sky MLP (STORM-style): params (P,), dirs (..., 3) -> (..., 3)."""
        cfg = self.cfg
        x = fourier(dirs, cfg.sky_freq)
        d_in, hdn = x.shape[-1], cfg.sky_hidden
        i = 0
        w1 = params[i: i + d_in * hdn].reshape(d_in, hdn); i += d_in * hdn
        b1 = params[i: i + hdn]; i += hdn
        w2 = params[i: i + hdn * 3].reshape(hdn, 3); i += hdn * 3
        b2 = params[i: i + 3]
        return torch.sigmoid(F.gelu(x @ w1 + b1) @ w2 + b2)


def affine_matrix(affine: Tensor) -> tuple[Tensor, Tensor]:
    """(..., 12) -> A = I + dA (..., 3, 3), b (..., 3)."""
    a = affine[..., :9].reshape(affine.shape[:-1] + (3, 3)) + torch.eye(3, device=affine.device)
    return a, affine[..., 9:]


def rotate_quats(T: Tensor, quats: Tensor) -> Tensor:
    """Apply the rotation part of (possibly non-rigid, averaged) T to wxyz quats."""
    from .geometry import mat_to_quat, quat_mul

    r = mat_to_quat(orthonormalize(T[..., :3, :3]))
    return quat_mul(r, quats)
