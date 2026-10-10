"""Shared fixtures for the backend tests.

Run from the project root:  .venv/bin/python -m pytest tests -q
Tests that need the trained checkpoints or the Places365 data are skipped (not
faked) when those files are missing.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.config import Settings  # noqa: E402

CHECKPOINTS = {
    "unet": ROOT / "checkpoints/unet_4gb/best.pt",
    "gated_unet": ROOT / "checkpoints/gated_unet_4gb/best.pt",
}
HAVE_CHECKPOINTS = all(p.is_file() for p in CHECKPOINTS.values())
needs_checkpoints = pytest.mark.skipif(not HAVE_CHECKPOINTS, reason="trained checkpoints not present")


def real_test_image() -> np.ndarray | None:
    """First image of the held-out Places365 test split (a real photograph), if available."""
    split = ROOT / "data/splits/test.txt"
    if not split.is_file():
        return None
    first = split.read_text().split()[0]
    path = ROOT / "data/raw" / first
    if not path.is_file():
        return None
    return np.asarray(Image.open(path).convert("RGB"))


def synthetic_image(h: int, w: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    img = np.stack([x / w * 255, y / h * 255, (x + y) / (h + w) * 255], -1)
    img += rng.normal(0, 12, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


@pytest.fixture(scope="session")
def photo() -> np.ndarray:
    """A real photo when the dataset is present, otherwise a synthetic one; made non-square."""
    img = real_test_image()
    if img is None:
        img = synthetic_image(256, 256)
    # Non-square crop (256 x 200) so aspect-ratio handling is exercised.
    return np.ascontiguousarray(img[:200, :])


def png_bytes(arr: np.ndarray, mode: str | None = None) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr, mode).save(buf, format="PNG")
    return buf.getvalue()


def jpeg_bytes(arr: np.ndarray, exif_orientation: int | None = None, quality: int = 95) -> bytes:
    buf = io.BytesIO()
    img = Image.fromarray(arr)
    if exif_orientation:
        exif = Image.Exif()
        exif[0x0112] = exif_orientation
        img.save(buf, format="JPEG", quality=quality, exif=exif)
    else:
        img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def decode_png(data: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


def rect_mask(h: int, w: int, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    m[y0:y1, x0:x1] = 1
    return m


@pytest.fixture(scope="session")
def settings() -> Settings:
    s = Settings.from_env()
    s.checkpoints = dict(CHECKPOINTS)
    s.frontend_dist = None
    s.device = "auto"
    return s
