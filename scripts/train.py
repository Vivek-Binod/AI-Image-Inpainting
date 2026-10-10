"""Train an inpainting model.

Examples:
    python scripts/train.py --config configs/unet.yaml
    python scripts/train.py --config configs/gated_unet.yaml --set train.epochs=40
    python scripts/train.py --config configs/unet.yaml --resume checkpoints/unet/last.pt

Outputs (in --run-dir, default checkpoints/<model name>):
    best.pt, last.pt, training_log.csv, history.json, config.json, summary.json, samples/epoch_XXX.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.trainer import Trainer  # noqa: E402
from src.utils.common import load_config, resolve_path, set_overrides  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="YAML config file")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")
    ap.add_argument("--run-dir", default=None, help="output directory (default checkpoints/<model>)")
    ap.add_argument("--resume", default=None, help="checkpoint to resume from")
    ap.add_argument("--overwrite", action="store_true", help="allow reusing a run dir that already has checkpoints")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    args = ap.parse_args()

    cfg = set_overrides(load_config(args.config), args.set)
    run_dir = resolve_path(args.run_dir or f"checkpoints/{cfg['model']['name']}")
    existing = list(run_dir.glob("*.pt")) if run_dir.exists() else []
    if existing and not args.resume and not args.overwrite:
        sys.exit(f"{run_dir} already contains checkpoints ({', '.join(p.name for p in existing)}). "
                 "Use --resume to continue, --overwrite to start over, or choose another --run-dir.")
    if args.overwrite and not args.resume and (run_dir / "training_log.csv").exists():
        (run_dir / "training_log.csv").unlink()  # start a fresh log for a fresh run

    summary = Trainer(cfg, run_dir, device=args.device, resume=args.resume).fit()
    print("Training summary:", summary)


if __name__ == "__main__":
    main()
