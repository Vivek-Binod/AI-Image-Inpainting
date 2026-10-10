"""End-to-end smoke test (Mode A). Checks that the pipeline *works*; the numbers it
produces are meaningless and must not be reported as model performance.

It needs no dataset: it draws small synthetic images (gradients and shapes),
then runs data loading -> mask generation -> forward pass -> loss -> backward ->
optimizer step -> checkpoint save -> checkpoint reload -> evaluation -> inference.

    python scripts/smoke_test.py [--model unet|gated_unet] [--device cpu]
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.datasets.inpainting_dataset import InpaintingDataset, load_image, make_splits  # noqa: E402
from src.evaluation.evaluator import evaluate  # noqa: E402
from src.inference.inpainter import Inpainter, make_mask  # noqa: E402
from src.masks import mask_config_from_dict  # noqa: E402
from src.training.trainer import Trainer, load_model_from_checkpoint  # noqa: E402
from src.utils.common import PROJECT_ROOT, get_device, load_config, set_overrides  # noqa: E402


def synthetic_image(rng: np.random.Generator, size: int = 96) -> np.ndarray:
    """A random colour gradient with a few filled shapes (uint8, RGB)."""
    a, b = rng.integers(0, 256, 3), rng.integers(0, 256, 3)
    t = np.linspace(0, 1, size)[None, :, None]
    img = (a * (1 - t) + b * t).repeat(size, 0).astype(np.uint8)
    for _ in range(int(rng.integers(2, 5))):
        color = tuple(int(c) for c in rng.integers(0, 256, 3))
        if rng.random() < 0.5:
            cv2.circle(img, tuple(int(v) for v in rng.integers(0, size, 2)), int(rng.integers(5, size // 3)), color, -1)
        else:
            p1, p2 = rng.integers(0, size, 2), rng.integers(0, size, 2)
            cv2.rectangle(img, tuple(int(v) for v in p1), tuple(int(v) for v in p2), color, -1)
    return img


def write_synthetic(folder: Path, n: int, seed: int) -> list[str]:
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    names = []
    for i in range(n):
        name = f"synthetic_{i:03d}.png"
        cv2.imwrite(str(folder / name), cv2.cvtColor(synthetic_image(rng), cv2.COLOR_RGB2BGR))
        names.append(f"{folder.name}/{name}")
    return names


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="unet", choices=["unet", "gated_unet"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--keep", action="store_true", help="keep the temporary directory")
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="inpaint_smoke_"))
    try:
        files = write_synthetic(tmp / "images", 96, seed=0)
        splits = make_splits([f for f in files], val_count=16, test_count=16, seed=0)
        (tmp / "splits").mkdir()
        for k, v in splits.items():
            (tmp / "splits" / f"{k}.txt").write_text("\n".join(v))
        print(f"[1/6] synthetic data: {', '.join(f'{k}={len(v)}' for k, v in splits.items())}")

        cfg = set_overrides(load_config(PROJECT_ROOT / "configs/smoke.yaml"),
                            [f"data.root={tmp}", f"data.splits_dir={tmp / 'splits'}", f"model.name={args.model}"])
        trainer = Trainer(cfg, tmp / "run", device=args.device)
        before = [p.detach().clone() for p in trainer.model.parameters()]
        summary = trainer.fit()
        changed = any(not torch.equal(a, b.detach()) for a, b in zip(before, trainer.model.parameters()))
        assert changed, "optimizer did not update any parameter"
        assert summary["epochs_completed"] == cfg["train"]["epochs"]
        assert (tmp / "run/last.pt").exists() and (tmp / "run/best.pt").exists()
        print(f"[2/6] training ran {summary['epochs_completed']} epochs; parameters updated; checkpoints saved")

        device = get_device(args.device)
        model, ckpt = load_model_from_checkpoint(tmp / "run/best.pt", device)
        for k, v in model.state_dict().items():
            assert torch.equal(v.cpu(), ckpt["model_state"][k].cpu())
        print(f"[3/6] checkpoint reloaded (epoch {ckpt['epoch']})")

        ds = InpaintingDataset(tmp, splits["test"], cfg["data"]["image_size"], mask_config_from_dict(cfg["mask"]),
                               deterministic=True, seed=123)
        _, s = evaluate(ds, {args.model: model}, device, tmp / "eval", use_lpips=False, save_examples=2,
                        batch_size=8)
        assert all(np.isfinite(s["overall"][m]["full_psnr"]["mean"]) for m in s["overall"])
        print(f"[4/6] evaluation ran for {list(s['overall'])}")

        inp = Inpainter(tmp / "run/best.pt", args.device)
        img = load_image(tmp / files[0])
        mask = make_mask("free_form", img.shape[:2], seed=1)
        out = inp(img, mask)
        known = mask < 0.5
        assert np.allclose(out["composite"][known], img[known]), "known pixels changed in the composite"
        assert np.isfinite(out["raw"]).all()
        print("[5/6] inference ran; known pixels preserved in the composite")

        sample_dir = PROJECT_ROOT / "data/sample"
        if not any(sample_dir.glob("*.png")):
            write_synthetic(sample_dir, 4, seed=99)
        print(f"[6/6] synthetic sample images available in {sample_dir}")
        print("SMOKE TEST PASSED (pipeline correctness only - not a performance result)")
    finally:
        if args.keep:
            print(f"Temporary files kept in {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
