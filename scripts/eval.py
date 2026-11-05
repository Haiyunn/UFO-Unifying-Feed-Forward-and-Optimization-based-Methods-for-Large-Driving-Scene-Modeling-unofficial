"""UFO-protocol evaluation (paper Table 1) on the PointForward val split.

For every val scene the first 16 s (160 frames) are cut into non-overlapping
windows of the requested length (2 s -> 8 windows, 8 s -> 2, 16 s -> 1).  In
each window every 5th frame is context (one recurrent step per context frame,
3 front cameras); every other frame is a novel-view target, rendered for all 3
cameras and scored at 160x240.  Metrics are averaged over all target images:

    PSNR, SSIM (Gaussian 11x11, PointForward definition), D-RMSE (rendered
    expected depth vs projected LiDAR, LiDAR pixels only), LPIPS (VGG),
    plus PSNR restricted to dynamic-object pixels.

Reconstruction time / peak memory exclude Gaussian rendering, as in Fig. 4.

    python -m scripts.eval --ckpt runs/stage2_8s/ckpt_latest.pt --seconds 2 8 16
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
import yaml

from scripts.train import build_model, to_device
from ufo.data.dataset import UFOSequences, collate, data_sources
from ufo.losses import LPIPS, depth_rmse_per_image, psnr_per_image, ssim_per_image
from ufo.render import render_targets


@torch.no_grad()
def evaluate(model, cfg: dict, seconds: int, lpips_fn, max_windows: int | None, save_dir: str | None,
             split: str = "val", source: dict | None = None) -> dict:
    d = cfg["data"]
    src = source or data_sources(d)[0]
    ds = UFOSequences(src["cache_root"], src["split_file"], split, seq_frames=seconds * 10,
                      ctx_stride=d.get("ctx_stride", 5), max_objects=cfg["model"].get("max_objects", 32),
                      train=False, eval_span=160, load_prior=bool(cfg["model"].get("depth_prior", False)))
    dev = torch.device("cuda")
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    per_scene: dict[str, dict] = {}
    recon_t, peak = [], []

    def add(key, vals, scene):
        vals = vals.float()
        sums[key] = sums.get(key, 0.0) + float(vals.sum())
        counts[key] = counts.get(key, 0) + int(vals.numel())
        ps = per_scene.setdefault(scene, {})
        ps.setdefault(key, []).extend(vals.tolist())

    n = len(ds) if max_windows is None else min(max_windows, len(ds))
    for w in range(n):
        item = collate([ds[w]])
        batch = to_device(item, dev)
        scene = item["scene"][0]
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        mems, sky, affine = model.forward_sequence(batch)
        g = model.decode(mems[0], batch["boxes"], 0)
        torch.cuda.synchronize()
        recon_t.append(time.time() - t0)
        peak.append(torch.cuda.max_memory_allocated() / 2 ** 30)

        tg = batch["target"]
        nt = tg["frame"].shape[1]
        # Render in chunks of target frames to bound memory on 16 s windows.
        for s in range(0, nt, 8):
            sub = {k: v[:, s: s + 8] for k, v in tg.items()}
            out, _ = render_targets(model, mems[0], sky, affine, batch, 0, sub, g=g)
            pred = out["rgb"].flatten(0, 1).clamp(0, 1)
            gt = sub["images"][0].flatten(0, 1)
            add("psnr", psnr_per_image(pred, gt), scene)
            add("ssim", ssim_per_image(pred, gt), scene)
            dr, ok = depth_rmse_per_image(out["depth"].flatten(0, 1), sub["depth"][0].flatten(0, 1).unsqueeze(1))
            add("drmse", dr[ok], scene)
            if lpips_fn is not None:
                for j in range(pred.shape[0]):
                    add("lpips", lpips_fn(pred[j: j + 1], gt[j: j + 1]).reshape(1), scene)
            dyn = sub["dynamic"][0].flatten(0, 1)
            has = dyn.flatten(1).sum(1) > 50
            if has.any():
                m = dyn[has].unsqueeze(1)
                se = (((pred[has] - gt[has]) ** 2) * m).flatten(1).sum(1) / (3 * m.flatten(1).sum(1))
                add("psnr_dynamic", -10 * torch.log10(se.clamp_min(1e-10)), scene)
            if save_dir and s == 0:
                import torchvision

                os.makedirs(save_dir, exist_ok=True)
                grid = torch.cat([gt[:3], pred[:3]], dim=0)
                torchvision.utils.save_image(grid, os.path.join(save_dir, f"{seconds}s_{scene}_{item['start'][0]:03d}.png"), nrow=3)
        print(f"  [{seconds}s] window {w + 1}/{n} {scene}@{item['start'][0]}: tokens {mems[0].n} "
              f"recon {recon_t[-1]:.2f}s peak {peak[-1]:.1f}G running PSNR {sums['psnr'] / counts['psnr']:.2f}", flush=True)

    res = {k: sums[k] / counts[k] for k in sums}
    res.update({
        "seconds": seconds,
        "windows": n,
        "target_images": counts.get("psnr", 0),
        "recon_time_s": sum(recon_t) / len(recon_t),
        "peak_mem_gb": max(peak),
        "per_scene": {s: {k: sum(v) / len(v) for k, v in m.items()} for s, m in per_scene.items()},
    })
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default=None, help="defaults to the config stored in the ckpt")
    ap.add_argument("--seconds", type=int, nargs="+", default=[2, 8, 16])
    ap.add_argument("--max-windows", type=int, default=None)
    ap.add_argument("--no-lpips", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--save-images", default=None)
    ap.add_argument("--split", default="val", help="split to score (train = fit check)")
    args = ap.parse_args()

    state = torch.load(args.ckpt, map_location="cuda")
    if args.config:
        with open(args.config) as fh:
            cfg = yaml.safe_load(fh)
    else:
        cfg = state["cfg"]
    cfg["model"]["grad_checkpoint"] = False
    model = build_model(cfg).cuda().eval()
    model.load_state_dict(state["model"])
    print(f"loaded {args.ckpt} (step {state.get('step')})")
    lpips_fn = None if args.no_lpips else LPIPS().cuda()

    results = {"ckpt": args.ckpt, "step": state.get("step")}
    sources = data_sources(cfg["data"])
    for sec in args.seconds:
        for src in sources:
            r = evaluate(model, cfg, sec, lpips_fn, args.max_windows, args.save_images, args.split, src)
            # One source keeps the original key; several are keyed by name.
            key = f"{sec}s" if len(sources) == 1 else f"{sec}s_{src['name']}"
            results[key] = r
            print(f"== {key}: PSNR {r['psnr']:.2f} SSIM {r['ssim']:.3f} D-RMSE {r.get('drmse', float('nan')):.2f} "
                  f"LPIPS {r.get('lpips', float('nan')):.3f} dynPSNR {r.get('psnr_dynamic', float('nan')):.2f} "
                  f"| recon {r['recon_time_s']:.2f}s peak {r['peak_mem_gb']:.1f}G", flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"written {args.out}")


if __name__ == "__main__":
    main()
