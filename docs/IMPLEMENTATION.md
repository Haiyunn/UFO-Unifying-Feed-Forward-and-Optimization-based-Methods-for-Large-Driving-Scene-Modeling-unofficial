# UFO reproduction — implementation notes

From-scratch reimplementation of **UFO: Unifying Feed-Forward and Optimization-based
Methods for Large Driving Scene Modeling** (Tan et al., arXiv 2602.20943, CVPR 2026).
The official repo (github.com/wm-research/UFO) ships only a README (no code, no
weights) as of 2026-10-21, so everything here is written from the paper.

## Data / split / protocol

| item | this repo | paper |
|---|---|---|
| dataset | Waymo NOTR processed (`/mnt/data/dataset/notr/processed/training`) | WOD |
| split | **PointForward `splits/waymo_ds.json`: 60 train / 11 val scenes** | official WOD train/val |
| cameras | front, front-left, front-right (0,1,2) | same |
| resolution | 160 x 240 | same |
| context | every 5th frame (0,5,10,...) — one recurrent step per context frame | same |
| targets | all other frames, 3 cams | same |
| eval windows | first 16 s (160 frames) of each val log: 2 s → 8 windows, 8 s → 2, 16 s → 1 | not stated |
| metrics | PSNR / SSIM (Gaussian 11x11) / D-RMSE (expected depth vs LiDAR px) per image, + LPIPS, dynamic-px PSNR | PSNR / SSIM / D-RMSE |
| boxes | Waymo GT tracks (all classes), top-32 per sequence by LiDAR points in box | off-the-shelf detector+tracker, top-32 per frame by LiDAR points |

The split has far fewer training scenes (60) than WOD train (798) — the largest
expected source of a gap to the paper's numbers, together with compute.

## Architecture (`ufo/model.py`)

Paper-specified: scene tokens (xyz + 768-d), image tokens = 8x8 patches of
RGB + Plücker rays (linear projection + positional/time encodings), box tokens via
Fourier embedding of centre + 8 corners through a 2-layer MLP, sky tokens decoded
into sky-MLP parameters, per-camera affine tokens, 12-layer ViT update transformer
(768-d), visibility filter (frustum + nearest K = 3600), local camera-centric
coordinates, Eq. 4 replace-and-append update, soft object assignment (Eq. 5),
box-weighted motion (Eq. 6-7), object-assignment CE (Eq. 8), lifespan opacity
(Eq. 9) with SoftPlus, lifespan regulariser (Eq. 10), loss weights of supp. Sec. 7,
2-layer-MLP Gaussian decoder, gsplat backend, AdamW, 1e-4 with linear warm-up.
Total parameters: **92.0 M** (paper: ~92 M).

Choices the paper leaves open (all documented in code):

1. **New tokens** — every image-token output becomes a new scene token, placed at
   the patch-centre ray times a predicted distance (`depth_head`, exp-parametrised,
   init 15 m).  Refined tokens get a bounded position update (`delta_head`,
   tanh × (0.1·dist + 0.2 m), zero-init).
2. **Gaussians per token** — M = 16 Gaussians per token from the 2-layer MLP:
   offset (tanh, ±1.5 patch footprints), scale (relative to the footprint),
   rotation, opacity, RGB, lifespan.  Footprint = distance × patch / f.
3. **Local frame** — the front camera of the current step.
3b. **Image-token time encoding** — MLP(sinusoidal(absolute time in the sequence, s)).
4. **Token re-entry encoding** — stored feature + MLP(Fourier(contracted local
   position)) + MLP(sinusoidal Δt to creation / reference time).
5. **Assignment keys** — logits q(s)·k(B) where k encodes the token position inside
   the box (box-normalised) plus a slot embedding; only token/box pairs within the
   box + margin are scored (others −inf), keeping cost linear in N.  A learnable
   "static" key forms column 0 (identity motion).  Label for Eq. 8: object whose
   box contains the token at its reference frame, otherwise static.
5b. **Motion blend** — Eq. 6-7 literally average absolute box poses and invert
   the average; with the identity (static) column this is near-singular for
   objects facing ~180° from the sequence frame (oncoming / opposite-lane cars)
   at mixed assignments.  We blend relative motions instead,
   `T = A_static·I + Σ_o A_o T_{t1,o} T_{t0,o}^{-1}` — identical for one-hot
   assignments.  The assignment key geometry uses detached token positions so
   L_obj cannot move geometry.
6. **Motion reference time** — each token stores the frame its position refers to
   (creation frame, reset to the current frame when refined; refined positions
   are first moved to the current frame with the token's assignment).  Lifespan
   centre t0 = creation time (never reset).
7. **Box tracks** — poses of frames where a track is unannotated hold the nearest
   annotated pose; slot ids are shuffled per training sequence.
8. **Depth units** — depth L1 / sky-depth are computed on depth / 80 m so the
   paper's weights (1.0 / 0.01) are commensurate with the RGB MSE.
9. **Flow regulariser** — MSE of token-centre displacement between the first
   target frame and the next frame.
10. **Supervision** — all target frames are rendered from the final state S_T
    (4 sampled target frames x 3 cams per training sequence).

## Training schedule (reduced compute)

Paper: 16 x H200, global batch 64, 2 s 100 k it → 8 s 50 k it, ~1 day.
Here: 2 x A100-80G (GPU6 + GPU7, both shared with other jobs), global batch 8 (2 s)
/ 4 (8 s): **stage 1 2 s 50 k it → stage 2 8 s 20 k it**, LPIPS on after 2 k it,
16 s evaluated zero-shot from the 8 s model (as in the paper).
