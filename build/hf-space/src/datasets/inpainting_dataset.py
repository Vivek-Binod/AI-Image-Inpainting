"""Image loading, preprocessing, train/val/test splitting and the inpainting Dataset.

Image representation: RGB, float32, values in [0, 1], tensors shaped (C, H, W).
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

from src.masks.generator import MaskConfig, generate_mask

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_image(path: str | Path, size: int | None = None) -> np.ndarray:
    """Load an image as RGB float32 in [0, 1] with shape (H, W, 3).

    Greyscale, palette and RGBA images are converted to RGB. If ``size`` is
    given, the image is centre-cropped to a square and resized to size x size.
    Raises ``FileNotFoundError`` or ``ValueError`` with a readable message on
    invalid input.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image format '{path.suffix}' for {path}. "
                         f"Supported: {sorted(IMAGE_EXTENSIONS)}")
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            if size is not None:
                img = center_crop_resize(img, size)
            arr = np.asarray(img, dtype=np.float32) / 255.0
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Could not decode image {path}: {exc}") from exc
    return arr


def center_crop_resize(img: Image.Image, size: int) -> Image.Image:
    """Centre-crop to the largest square, then resize to (size, size) with bicubic filtering."""
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    img = img.crop((left, top, left + s, top + s))
    if s != size:
        img = img.resize((size, size), Image.BICUBIC)
    return img


def is_valid_image(path: str | Path) -> bool:
    """Return True if PIL can fully decode the file."""
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            img.convert("RGB").load()
        return True
    except Exception:
        return False


def make_splits(files: list[str], val_count: int, test_count: int, seed: int) -> dict[str, list[str]]:
    """Shuffle (deterministically) and split a file list into disjoint train/val/test lists."""
    if val_count + test_count >= len(files):
        raise ValueError(f"val ({val_count}) + test ({test_count}) must be < number of files ({len(files)})")
    files = sorted(files)
    random.Random(seed).shuffle(files)
    test = files[:test_count]
    val = files[test_count:test_count + val_count]
    train = files[test_count + val_count:]
    return {"train": train, "val": val, "test": test}


def read_split(split_file: str | Path) -> list[str]:
    path = Path(split_file)
    if not path.is_file():
        raise FileNotFoundError(f"Split file not found: {path}. Run scripts/prepare_data.py first.")
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def image_to_tensor(arr: np.ndarray) -> torch.Tensor:
    """(H, W, C) float array -> (C, H, W) float tensor."""
    return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1))).float()


def tensor_to_image(t: torch.Tensor) -> np.ndarray:
    """(C, H, W) tensor in [0, 1] -> (H, W, C) float numpy array clipped to [0, 1]."""
    return t.detach().float().cpu().clamp(0, 1).numpy().transpose(1, 2, 0)


class InpaintingDataset(Dataset):
    """Returns dictionaries with keys:

    - ``image``:  ground truth, (3, H, W) in [0, 1]
    - ``mask``:   (1, H, W), 1 = missing
    - ``masked``: corrupted image I * (1 - M), (3, H, W)
    - ``index``, ``mask_family``, ``path``

    With ``deterministic=True`` (validation/test) the mask for item ``i`` is
    generated from ``seed + i``, so every run and every method sees the same
    masks. With ``deterministic=False`` (training) masks are drawn from the
    torch RNG, which DataLoader seeds per worker.
    """

    def __init__(self, root: str | Path, files: list[str], image_size: int = 128,
                 mask_cfg: MaskConfig | None = None, augment: bool = False,
                 deterministic: bool = False, seed: int = 0):
        self.root = Path(root)
        self.files = list(files)
        self.image_size = image_size
        self.mask_cfg = mask_cfg or MaskConfig()
        self.augment = augment
        self.deterministic = deterministic
        self.seed = seed
        if not self.files:
            raise ValueError("InpaintingDataset received an empty file list.")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        path = self.root / self.files[idx]
        img = load_image(path, self.image_size)
        if self.augment and torch.rand(1).item() < 0.5:
            img = img[:, ::-1, :]  # horizontal flip

        if self.deterministic:
            rng = np.random.default_rng(self.seed + idx)
        else:
            rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,)).item()))
        mask, family = generate_mask(self.image_size, self.image_size, rng, self.mask_cfg)

        image_t = image_to_tensor(img)
        mask_t = torch.from_numpy(mask)[None]
        return {
            "image": image_t,
            "mask": mask_t,
            "masked": image_t * (1.0 - mask_t),
            "index": idx,
            "mask_family": family,
            "path": str(self.files[idx]),
        }
