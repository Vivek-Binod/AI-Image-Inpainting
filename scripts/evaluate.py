"""Evaluate trained checkpoints and non-learned baselines on the held-out test split.

Example:
    python scripts/evaluate.py --checkpoints checkpoints/unet/best.pt checkpoints/gated_unet/best.pt

All methods are evaluated on the same test images with the same seeded masks.
Outputs (in --out-dir, default results/):
    metrics/per_image_metrics.csv, metrics/summary.json, metrics/results_table.md,
    reconstructed/<method>/*.png (first --save-examples test images)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.inpainting_dataset import InpaintingDataset, read_split  # noqa: E402
from src.evaluation.evaluator import evaluate  # noqa: E402
from src.masks import mask_config_from_dict  # noqa: E402
from src.training.trainer import load_model_from_checkpoint  # noqa: E402
from src.utils.common import get_device, load_config, resolve_path, save_json, seed_everything  # noqa: E402

LABELS = {"unet": "U-Net (conv)", "gated_unet": "Gated-Conv U-Net", "mean_fill": "Mean-colour fill",
          "telea": "OpenCV Telea"}


def results_table(summary: dict) -> str:
    cols = [("full_psnr", "PSNR (dB) ↑"), ("full_ssim", "SSIM ↑"), ("full_mse", "MSE ↓"),
            ("hole_psnr", "Hole PSNR (dB) ↑"), ("hole_mse", "Hole MSE ↓"), ("raw_full_psnr", "Raw PSNR (dB) ↑"),
            ("raw_full_ssim", "Raw SSIM ↑"), ("full_lpips", "LPIPS ↓")]
    any_method = next(iter(summary["overall"].values()))
    cols = [c for c in cols if c[0] in any_method]
    lines = ["| Method | N | " + " | ".join(c[1] for c in cols) + " |",
             "|---|---|" + "---|" * len(cols)]
    for m, agg in summary["overall"].items():
        cells = []
        for k, _ in cols:
            fmt = "{:.5f} ± {:.5f}" if "mse" in k else "{:.4f} ± {:.4f}" if k in ("full_ssim", "raw_full_ssim", "full_lpips") else "{:.2f} ± {:.2f}"
            cells.append(fmt.format(agg[k]["mean"], agg[k]["std"]))
        lines.append(f"| {LABELS.get(m, m)} | {agg['n']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", nargs="*", default=[], help="trained model checkpoints (.pt)")
    ap.add_argument("--config", default="configs/default.yaml", help="config providing data/mask settings")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--baselines", nargs="*", default=["mean_fill", "telea"])
    ap.add_argument("--no-lpips", action="store_true")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--save-examples", type=int, default=24)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    device = get_device(args.device)

    models, ckpt_info, image_size = {}, {}, None
    for path in args.checkpoints:
        model, ckpt = load_model_from_checkpoint(path, device)
        name = ckpt["model_name"]
        if name in models:
            sys.exit(f"Two checkpoints for the same model '{name}' - evaluate them separately.")
        models[name] = model
        ckpt_info[name] = {"path": str(path), "epoch": ckpt["epoch"], "best_val": ckpt["best_val"],
                           "saved_at": ckpt.get("saved_at")}
        if image_size not in (None, ckpt["image_size"]):
            sys.exit("Checkpoints were trained at different image sizes.")
        image_size = ckpt["image_size"]
        # Use the mask settings the model was trained with.
        cfg["mask"] = ckpt["config"].get("mask", cfg["mask"])
    image_size = image_size or cfg["data"]["image_size"]

    files = read_split(resolve_path(cfg["data"]["splits_dir"]) / f"{args.split}.txt")
    if args.max_images:
        files = files[: args.max_images]
    offset = cfg["eval"]["test_mask_seed_offset"] if args.split == "test" else 1_000_000
    ds = InpaintingDataset(resolve_path(cfg["data"]["root"]), files, image_size,
                           mask_config_from_dict(cfg["mask"]), augment=False, deterministic=True,
                           seed=cfg["seed"] + offset)
    out_dir = resolve_path(args.out_dir)
    print(f"Evaluating {list(models) + args.baselines} on {len(ds)} {args.split} images ({image_size}px) on {device}")
    _, summary = evaluate(ds, models, device, out_dir, baselines=args.baselines,
                          use_lpips=not args.no_lpips, save_examples=args.save_examples)
    summary["protocol"] = {"split": args.split, "num_images": len(ds), "image_size": image_size,
                           "mask_seed_base": cfg["seed"] + offset, "mask_config": cfg["mask"],
                           "checkpoints": ckpt_info, "device": str(device)}
    metrics_dir = out_dir / "metrics"
    save_json(summary, metrics_dir / "summary.json")
    table = results_table(summary)
    (metrics_dir / "results_table.md").write_text(table + "\n")
    print(table)
    print(json.dumps(summary["protocol"], indent=2))


if __name__ == "__main__":
    main()
