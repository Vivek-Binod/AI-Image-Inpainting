"""Controlled evaluation of the *website* inference path (backend/app) on the test split.

It answers questions raised by deploying the 128-px models on real photos, using the
same test images and the same seeded masks as scripts/evaluate.py (seed 42 + 2,000,000 + i):

  r128        128-px image and mask, network at 128 (the report's protocol, via the web
              code). Reproduces the published numbers -> baseline check.
  n256_w128   native 256-px image, mask upsampled x2 (nearest), network at 128, prediction
              upscaled and composited at 256 (the website default).
  n256_w128_dilate  as n256_w128 but with the "dilate" preprocessing (hole enlarged by
              the bicubic support) instead of the default normalised resampling.
  n256_w256   same input, network run directly at 256 px (an untested resolution).
  ns_cli      non-square 256x176 crop, ORIGINAL scripts/inpaint.py path (squashes to 128x128).
  ns_web      same crop, website path (aspect preserved: 128x88).
  telea256    OpenCV Telea at 256 px, as a non-learned reference for LPIPS.

Metrics per image: full PSNR, SSIM, hole PSNR (inside the mask) and LPIPS (AlexNet,
lower is better). Nothing here is used for training or model selection.

    .venv/bin/python scripts/web_eval.py [--max-images N] [--out-dir results/web_eval]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from backend.app.config import Settings  # noqa: E402
from backend.app.engine import Engine  # noqa: E402
from src.datasets.inpainting_dataset import load_image, read_split  # noqa: E402
from src.evaluation.baselines import telea_inpaint  # noqa: E402
from src.evaluation.evaluator import bucket_name  # noqa: E402
from src.evaluation.metrics import LPIPSMetric, masked_psnr, psnr, ssim  # noqa: E402
from src.inference.inpainter import Inpainter  # noqa: E402
from src.masks import mask_config_from_dict  # noqa: E402
from src.masks.generator import generate_mask  # noqa: E402
from src.utils.common import load_config, resolve_path, save_json  # noqa: E402

CROP = (40, 216)  # rows kept for the non-square condition (256 x 176)


def f32(a: np.ndarray) -> np.ndarray:
    return a.astype(np.float32) / 255.0


def metrics(out8: np.ndarray, gt8: np.ndarray, mask: np.ndarray, lp) -> dict:
    out, gt = f32(out8), f32(gt8)
    return {"psnr": psnr(out, gt), "ssim": ssim(out, gt), "hole_psnr": masked_psnr(out, gt, mask.astype(np.float32)),
            "lpips": lp(out, gt) if lp else float("nan")}


def ci95(x: np.ndarray) -> float:
    return float(1.96 * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--out-dir", default="results/web_eval")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-lpips", action="store_true")
    args = ap.parse_args()

    cfg = load_config(resolve_path("configs/default.yaml"))
    settings = Settings.from_env()
    settings.device = args.device
    engine = Engine(settings)
    assert engine.available == ["unet", "gated_unet"], engine.describe()
    ckpt_cfg = engine.models["unet"].info
    mask_cfg = mask_config_from_dict(cfg["mask"])
    seed_base = cfg["seed"] + cfg["eval"]["test_mask_seed_offset"]
    cli = {n: Inpainter(settings.checkpoints[n], str(engine.device)) for n in engine.available}
    lp = None if args.no_lpips else LPIPSMetric(str(engine.device))

    files = read_split(resolve_path(cfg["data"]["splits_dir"]) / "test.txt")
    if args.max_images:
        files = files[: args.max_images]
    root = resolve_path(cfg["data"]["root"])
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    t0 = time.time()
    for i, rel in enumerate(files):
        img128 = (load_image(root / rel, 128) * 255).round().astype(np.uint8)
        img256 = (load_image(root / rel, 256) * 255).round().astype(np.uint8)
        m128, family = generate_mask(128, 128, np.random.default_rng(seed_base + i), mask_cfg)
        m128 = m128.astype(np.uint8)
        m256 = np.kron(m128, np.ones((2, 2), np.uint8))
        crop_img, crop_m = img256[CROP[0]:CROP[1]], m256[CROP[0]:CROP[1]]
        base = {"index": i, "path": rel, "mask_family": family, "hole_ratio": float(m128.mean()),
                "hole_bucket": bucket_name(float(m128.mean())), "crop_hole_ratio": float(crop_m.mean())}

        for name in engine.available:
            for cond, im, m, work, rs in (
                    ("r128", img128, m128, 128, "normalized"),
                    ("n256_w128", img256, m256, 128, "normalized"),
                    ("n256_w128_dilate", img256, m256, 128, "dilate"),
                    ("n256_w256", img256, m256, 256, "normalized"),
                    ("ns_web", crop_img, crop_m, 128, "normalized")):
                if not m.any():
                    continue
                res = engine.inpaint(im, m, name, work, rs).result
                rows.append({**base, "method": name, "condition": cond, **metrics(res, im, m, lp)})
            if crop_m.any():
                out = cli[name](f32(crop_img), crop_m.astype(np.float32))["composite"]
                out8 = (np.clip(out, 0, 1) * 255).round().astype(np.uint8)
                rows.append({**base, "method": name, "condition": "ns_cli", **metrics(out8, crop_img, crop_m, lp)})
        tel = (np.clip(telea_inpaint(f32(img256) * (1 - m256[..., None]), m256.astype(np.float32)), 0, 1) * 255
               ).round().astype(np.uint8)
        rows.append({**base, "method": "telea", "condition": "telea256", **metrics(tel, img256, m256, lp)})
        if (i + 1) % 250 == 0:
            print(f"{i + 1}/{len(files)} images, {time.time() - t0:.0f}s", flush=True)

    with open(out_dir / "per_image.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ---- aggregate ----
    keys = ("psnr", "ssim", "hole_psnr", "lpips")
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["method"], r["condition"]), []).append(r)
    summary = {"protocol": {"split": "test", "num_images": len(files), "mask_seed_base": seed_base,
                            "mask_config": cfg["mask"], "checkpoints": {n: str(settings.checkpoints[n]) for n in cli},
                            "training_resolution": ckpt_cfg["training_resolution"], "device": str(engine.device),
                            "lpips": None if lp is None else "lpips 0.1.4, AlexNet, inputs in [-1, 1]",
                            "crop_rows": CROP},
               "overall": {}, "by_mask_family": {}, "by_hole_bucket": {}, "paired": {}}
    for (meth, cond), rs in groups.items():
        k = f"{meth}/{cond}"
        summary["overall"][k] = {"n": len(rs), **{m: {"mean": float(np.nanmean([r[m] for r in rs])),
                                                     "ci95": ci95(np.array([r[m] for r in rs]))} for m in keys}}
        for field, dest in (("mask_family", "by_mask_family"), ("hole_bucket", "by_hole_bucket")):
            for val in sorted({r[field] for r in rs}):
                sub = [r for r in rs if r[field] == val]
                summary[dest].setdefault(k, {})[val] = {"n": len(sub), **{m: float(np.nanmean([r[m] for r in sub]))
                                                                            for m in keys}}

    def paired(a: str, b: str) -> dict:
        ra = {r["index"]: r for r in groups.get(tuple(a.split("/")), [])}
        rb = {r["index"]: r for r in groups.get(tuple(b.split("/")), [])}
        idx = sorted(set(ra) & set(rb))
        out = {"n": len(idx)}
        for m in keys:
            d = np.array([rb[i][m] - ra[i][m] for i in idx])
            out[m] = {"mean_diff": float(d.mean()), "ci95": ci95(d), "b_better_frac":
                      float((d < 0).mean() if m == "lpips" else (d > 0).mean())}
        return out

    for name in cli:
        summary["paired"][f"{name}: n256_w128_dilate -> n256_w128"] = paired(f"{name}/n256_w128_dilate", f"{name}/n256_w128")
        summary["paired"][f"{name}: n256_w128 -> n256_w256"] = paired(f"{name}/n256_w128", f"{name}/n256_w256")
        summary["paired"][f"{name}: ns_cli -> ns_web"] = paired(f"{name}/ns_cli", f"{name}/ns_web")
    summary["paired"]["unet -> gated_unet (n256_w128)"] = paired("unet/n256_w128", "gated_unet/n256_w128")

    # Reproduction check against the published per-image test metrics.
    ref = resolve_path("results/metrics/per_image_metrics.csv")
    if ref.is_file():
        with open(ref) as f:
            published = {(int(r["index"]), r["method"]): float(r["hole_psnr"]) for r in csv.DictReader(f)}
        for name in cli:
            d = [abs(r["hole_psnr"] - published[(r["index"], name)]) for r in groups[(name, "r128")]
                 if (r["index"], name) in published]
            summary.setdefault("reproduction", {})[name] = {
                "n": len(d), "mean_abs_hole_psnr_diff_db": float(np.mean(d)), "max_abs_diff_db": float(np.max(d))}
    save_json(summary, out_dir / "summary.json")

    lines = ["| method / condition | n | PSNR (dB) | SSIM | hole PSNR (dB) | LPIPS ↓ |", "|---|---|---|---|---|---|"]
    for k, v in summary["overall"].items():
        lines.append(f"| {k} | {v['n']} | {v['psnr']['mean']:.2f} ± {v['psnr']['ci95']:.2f} | "
                     f"{v['ssim']['mean']:.4f} | {v['hole_psnr']['mean']:.2f} ± {v['hole_psnr']['ci95']:.2f} | "
                     f"{v['lpips']['mean']:.4f} ± {v['lpips']['ci95']:.4f} |")
    lines += ["", "Paired differences (B − A, mean ± 95% CI):", ""]
    for k, v in summary["paired"].items():
        lines.append(f"- {k} (n={v['n']}): hole PSNR {v['hole_psnr']['mean_diff']:+.3f} ± {v['hole_psnr']['ci95']:.3f} dB, "
                     f"PSNR {v['psnr']['mean_diff']:+.3f} ± {v['psnr']['ci95']:.3f} dB, SSIM {v['ssim']['mean_diff']:+.4f}, "
                     f"LPIPS {v['lpips']['mean_diff']:+.4f} ± {v['lpips']['ci95']:.4f}")
    if "reproduction" in summary:
        lines += ["", "Reproduction of published test hole PSNR (r128 vs results/metrics):", ""]
        for k, v in summary["reproduction"].items():
            lines.append(f"- {k}: mean |diff| {v['mean_abs_hole_psnr_diff_db']:.4f} dB, max {v['max_abs_diff_db']:.4f} dB (n={v['n']})")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Done in {time.time() - t0:.0f}s -> {out_dir}")


if __name__ == "__main__":
    main()
