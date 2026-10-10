"""Inpaint your own image with a trained checkpoint.

Examples:
    # user-supplied mask (white = missing, black = known; same size as the image)
    python scripts/inpaint.py --image photo.jpg --mask mask.png --checkpoint checkpoints/gated_unet/best.pt

    # generated mask: center | rect | free_form | mixed
    python scripts/inpaint.py --image photo.jpg --mask-type free_form --seed 3 --checkpoint checkpoints/gated_unet/best.pt

Outputs in --out-dir (default results/inference/<image stem>/):
    original.png, mask.png, masked.png, raw_output.png, inpainted.png, comparison.png
If the input image itself is the ground truth (generated mask), PSNR/SSIM are also printed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.datasets.inpainting_dataset import load_image  # noqa: E402
from src.evaluation.evaluator import save_png  # noqa: E402
from src.evaluation.metrics import masked_psnr, psnr, ssim  # noqa: E402
from src.inference.inpainter import Inpainter, load_mask, make_mask  # noqa: E402
from src.utils.common import resolve_path  # noqa: E402
from src.visualization.plots import comparison_figure  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True)
    ap.add_argument("--checkpoint", required=True)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--mask", help="mask image: white = missing")
    g.add_argument("--mask-type", default="center", choices=["center", "rect", "free_form", "mixed"])
    ap.add_argument("--seed", type=int, default=0, help="seed for generated masks")
    ap.add_argument("--center-frac", type=float, default=0.4, help="side of the centre hole as a fraction")
    ap.add_argument("--max-size", type=int, default=None, help="optionally downscale large inputs to this side")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    try:
        image = load_image(args.image)
        if args.max_size and max(image.shape[:2]) > args.max_size:
            from PIL import Image
            h, w = image.shape[:2]
            s = args.max_size / max(h, w)
            im = Image.fromarray((image * 255).round().astype(np.uint8)).resize((round(w * s), round(h * s)), Image.BICUBIC)
            image = np.asarray(im, dtype=np.float32) / 255.0
        hw = image.shape[:2]
        mask = load_mask(args.mask, hw) if args.mask else make_mask(args.mask_type, hw, args.seed, args.center_frac)
        inpainter = Inpainter(args.checkpoint, args.device)
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(f"Error: {exc}")

    out = inpainter(image, mask)
    out_dir = resolve_path(args.out_dir or f"results/inference/{Path(args.image).stem}")
    save_png(image, out_dir / "original.png")
    save_png(mask, out_dir / "mask.png")
    save_png(out["masked"] + mask[..., None] * 0.8, out_dir / "masked.png")
    save_png(out["raw"], out_dir / "raw_output.png")
    save_png(out["composite"], out_dir / "inpainted.png")
    comparison_figure(image, mask, out["masked"], out["composite"], out_dir / "comparison.png",
                      title=f"{inpainter.ckpt['model_name']} (epoch {inpainter.ckpt['epoch']})")
    print(f"Model: {inpainter.ckpt['model_name']} | trained epochs: {inpainter.ckpt['epoch']} | "
          f"model resolution: {inpainter.size}px | input {hw[1]}x{hw[0]} | hole {mask.mean():.1%}")
    if not args.mask:
        print(f"Against the original image: PSNR {psnr(out['composite'], image):.2f} dB, "
              f"SSIM {ssim(out['composite'], image):.4f}, hole PSNR {masked_psnr(out['raw'], image, mask):.2f} dB")
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
