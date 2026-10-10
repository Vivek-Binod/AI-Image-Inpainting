"""Assemble a Hugging Face Docker Space in build/hf-space (originals are only read).

The folder gets: deploy/huggingface/* (Dockerfile, README, requirements), backend/, src/,
the built frontend (frontend/dist), results/metrics/summary.json, and weights-only copies
of the two best checkpoints (optimizer/scheduler state dropped; the weights are identical).

    .venv/bin/python scripts/prepare_hf_space.py            # then upload build/hf-space
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = {"unet": "checkpoints/unet_4gb/best.pt", "gated_unet": "checkpoints/gated_unet_4gb/best.pt"}
KEEP = ("model_name", "base_channels", "image_size", "model_state", "epoch", "best_val", "config", "saved_at")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="build/hf-space")
    out = ROOT / ap.parse_args().out

    dist = ROOT / "frontend/dist/index.html"
    if not dist.is_file():
        sys.exit("frontend/dist is missing - run ./start.sh once (or npm run build in frontend/) first.")

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    for f in (ROOT / "deploy/huggingface").iterdir():
        shutil.copy2(f, out / f.name)
    shutil.copytree(ROOT / "backend", out / "backend", ignore=ignore)
    shutil.copytree(ROOT / "src", out / "src", ignore=ignore)
    shutil.copytree(ROOT / "frontend/dist", out / "frontend/dist")
    (out / "results/metrics").mkdir(parents=True)
    shutil.copy2(ROOT / "results/metrics/summary.json", out / "results/metrics/summary.json")

    for name, rel in CHECKPOINTS.items():
        ckpt = torch.load(ROOT / rel, map_location="cpu", weights_only=False)
        slim = {k: ckpt[k] for k in KEEP if k in ckpt}
        dest = out / rel
        dest.parent.mkdir(parents=True)
        torch.save(slim, dest)
        print(f"{name}: {rel} {(ROOT / rel).stat().st_size / 1e6:.0f} MB -> {dest.stat().st_size / 1e6:.0f} MB (weights only)")

    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"Space folder ready: {out} ({size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
