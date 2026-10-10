"""Evaluate trained models and baselines on the same test images and the same masks."""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.datasets.inpainting_dataset import tensor_to_image
from src.evaluation.baselines import BASELINES
from src.evaluation.metrics import LPIPSMetric, masked_mse, mse, psnr_from_mse, ssim

HOLE_BUCKETS = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 1.01)]


def bucket_name(ratio: float) -> str:
    for lo, hi in HOLE_BUCKETS:
        if lo <= ratio < hi:
            return f"{int(lo * 100)}-{min(int(hi * 100), 100)}%"
    return "unknown"


def image_metrics(gt: np.ndarray, raw: np.ndarray, comp: np.ndarray, mask: np.ndarray,
                  lpips_fn=None) -> dict:
    """Metrics for one image.

    full_*     : composite (known pixels copied from input) vs ground truth, whole image
    raw_full_* : raw network output vs ground truth, whole image (no pixels copied)
    hole_*     : missing region only (raw == composite there)
    """
    full_mse = mse(comp, gt)
    raw_mse = mse(raw, gt)
    hole_mse = masked_mse(raw, gt, mask)
    out = {
        "full_mse": full_mse, "full_psnr": psnr_from_mse(full_mse), "full_ssim": ssim(comp, gt),
        "raw_full_mse": raw_mse, "raw_full_psnr": psnr_from_mse(raw_mse), "raw_full_ssim": ssim(raw, gt),
        "hole_mse": hole_mse, "hole_psnr": psnr_from_mse(hole_mse),
    }
    if lpips_fn is not None:
        out["full_lpips"] = lpips_fn(comp, gt)
    return out


def save_png(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if arr.ndim == 2:
        Image.fromarray((np.clip(arr, 0, 1) * 255).round().astype(np.uint8), "L").save(path)
    else:
        Image.fromarray((np.clip(arr, 0, 1) * 255).round().astype(np.uint8)).save(path)


@torch.no_grad()
def evaluate(dataset, models: dict, device: torch.device, out_dir: Path, baselines=("mean_fill", "telea"),
             use_lpips: bool = True, save_examples: int = 24, batch_size: int = 32) -> tuple[list[dict], dict]:
    """Run all methods on ``dataset`` and write per-image and aggregate metrics.

    ``models`` maps a method name to a model in eval mode. Returns (rows, summary).
    """
    out_dir = Path(out_dir)
    lpips_fn = None
    if use_lpips:
        try:
            lpips_fn = LPIPSMetric(str(device))
        except Exception as exc:  # optional dependency / weights download may fail offline
            print(f"LPIPS unavailable ({exc}); continuing without it.")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    rows: list[dict] = []
    done = 0
    for batch in loader:
        image, mask, masked = batch["image"], batch["mask"], batch["masked"]
        preds = {name: m(masked.to(device), mask.to(device)).float().cpu() for name, m in models.items()}
        for j in range(image.shape[0]):
            idx = int(batch["index"][j])
            gt = tensor_to_image(image[j])
            mk = mask[j, 0].numpy()
            mi = tensor_to_image(masked[j])
            outputs = {}
            for name, p in preds.items():
                raw = tensor_to_image(p[j])
                outputs[name] = (raw, mk[..., None] * raw + (1 - mk[..., None]) * gt)
            for name in baselines:
                filled = BASELINES[name](mi, mk)
                outputs[name] = (filled, filled)
            base = {"index": idx, "path": batch["path"][j], "mask_family": batch["mask_family"][j],
                    "hole_ratio": float(mk.mean())}
            base["hole_bucket"] = bucket_name(base["hole_ratio"])
            for name, (raw, comp) in outputs.items():
                rows.append({**base, "method": name, **image_metrics(gt, raw, comp, mk, lpips_fn)})
                if idx < save_examples:
                    save_png(comp, out_dir / "reconstructed" / name / f"{idx:04d}_composite.png")
                    if name in preds:
                        save_png(raw, out_dir / "reconstructed" / name / f"{idx:04d}_raw.png")
            if idx < save_examples:
                save_png(gt, out_dir / "reconstructed" / "inputs" / f"{idx:04d}_original.png")
                save_png(mk, out_dir / "reconstructed" / "inputs" / f"{idx:04d}_mask.png")
                save_png(mi + mk[..., None] * 0.8, out_dir / "reconstructed" / "inputs" / f"{idx:04d}_masked.png")
        done += image.shape[0]
        if done % (batch_size * 20) == 0 or done == len(dataset):
            print(f"  evaluated {done}/{len(dataset)} images", flush=True)

    summary = summarize(rows)
    write_rows(rows, out_dir / "metrics" / "per_image_metrics.csv")
    return rows, summary


METRIC_KEYS = ["full_mse", "full_psnr", "full_ssim", "raw_full_mse", "raw_full_psnr", "raw_full_ssim",
               "hole_mse", "hole_psnr", "full_lpips"]


def _agg(rows: list[dict]) -> dict:
    out = {"n": len(rows)}
    for k in METRIC_KEYS:
        vals = np.array([r[k] for r in rows if k in r and math.isfinite(r[k])], dtype=np.float64)
        if len(vals):
            out[k] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0}
    return out


def summarize(rows: list[dict]) -> dict:
    methods = sorted({r["method"] for r in rows})
    summary = {"overall": {}, "by_mask_family": {}, "by_hole_bucket": {}}
    for m in methods:
        mr = [r for r in rows if r["method"] == m]
        summary["overall"][m] = _agg(mr)
        summary["by_mask_family"][m] = {f: _agg([r for r in mr if r["mask_family"] == f])
                                        for f in sorted({r["mask_family"] for r in mr})}
        summary["by_hole_bucket"][m] = {b: _agg([r for r in mr if r["hole_bucket"] == b])
                                        for b in [bucket_name(lo) for lo, _ in HOLE_BUCKETS]
                                        if any(r["hole_bucket"] == b for r in mr)}
    return summary


def write_rows(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r.keys()))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
