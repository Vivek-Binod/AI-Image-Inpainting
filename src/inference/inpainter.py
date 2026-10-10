"""Inference on user images with a trained checkpoint.

Images of any size are supported: the image and mask are resized to the
resolution the model was trained at (e.g. 128x128), the model predicts the
hole, the prediction is resized back to the original resolution, and the final
composite takes known pixels from the *original* full-resolution image. Fine
detail inside the hole is therefore limited by the training resolution.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.masks.generator import MaskConfig, center_square_mask, generate_mask
from src.training.trainer import load_model_from_checkpoint
from src.utils.common import get_device


def load_mask(path: str | Path, expected_hw: tuple[int, int] | None = None) -> np.ndarray:
    """Load a mask image: white (>127) = missing (1), black = known (0)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Mask not found: {path}")
    try:
        with Image.open(path) as m:
            arr = np.asarray(m.convert("L"), dtype=np.float32)
    except Exception as exc:
        raise ValueError(f"Could not decode mask {path}: {exc}") from exc
    mask = (arr > 127).astype(np.float32)
    if expected_hw is not None and mask.shape != tuple(expected_hw):
        raise ValueError(f"Mask size {mask.shape[::-1]} (W x H) does not match image size "
                         f"{tuple(expected_hw)[::-1]} (W x H).")
    if mask.sum() == 0:
        raise ValueError("Mask contains no missing (white) pixels - nothing to inpaint.")
    return mask


def _resize(arr: np.ndarray, hw: tuple[int, int], is_mask: bool = False) -> np.ndarray:
    h, w = hw
    if is_mask:
        # Conservative downscale: a low-resolution pixel is "missing" if any part of
        # it was missing, so hole content never leaks into the network input.
        img = Image.fromarray((arr * 255).astype(np.uint8), "L").resize((w, h), Image.BOX)
        return (np.asarray(img) > 0).astype(np.float32)
    img = Image.fromarray((np.clip(arr, 0, 1) * 255).round().astype(np.uint8)).resize((w, h), Image.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0


def make_mask(kind: str, hw: tuple[int, int], seed: int = 0, frac: float = 0.4) -> np.ndarray:
    """Create a mask for an image of size ``hw``: 'center', 'rect', 'free_form' or 'mixed'."""
    h, w = hw
    if kind == "center":
        return center_square_mask(h, w, frac)
    mask, _ = generate_mask(h, w, np.random.default_rng(seed), MaskConfig(mask_type=kind))
    return mask


class Inpainter:
    def __init__(self, checkpoint: str | Path, device: str = "auto"):
        self.device = get_device(device)
        self.model, self.ckpt = load_model_from_checkpoint(checkpoint, self.device)
        self.size = int(self.ckpt.get("image_size", 128))

    @torch.no_grad()
    def __call__(self, image: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
        """image: (H, W, 3) float in [0, 1]; mask: (H, W) with 1 = missing.

        Returns a dict with 'masked', 'raw' (network output at original size) and 'composite'.
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected an RGB image of shape (H, W, 3), got {image.shape}")
        if mask.shape != image.shape[:2]:
            raise ValueError(f"Mask shape {mask.shape} does not match image shape {image.shape[:2]}")
        mask = (mask > 0.5).astype(np.float32)
        H, W = mask.shape
        s = self.size
        img_s = _resize(image, (s, s)) if (H, W) != (s, s) else image.astype(np.float32)
        mask_s = _resize(mask, (s, s), is_mask=True) if (H, W) != (s, s) else mask
        img_t = torch.from_numpy(img_s.transpose(2, 0, 1))[None].to(self.device)
        mask_t = torch.from_numpy(mask_s)[None, None].to(self.device)
        pred = self.model(img_t * (1 - mask_t), mask_t)[0].float().cpu().numpy().transpose(1, 2, 0)
        raw = _resize(pred, (H, W)) if (H, W) != (s, s) else pred
        m3 = mask[..., None]
        return {"masked": image * (1 - m3), "raw": raw, "composite": m3 * raw + (1 - m3) * image}
