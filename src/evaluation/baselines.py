"""Non-learned inpainting baselines. Inputs: corrupted image (H, W, 3) in [0, 1]
and mask (H, W) with 1 = missing. Outputs fill only the hole; known pixels are
returned unchanged, so for these methods the raw output equals the composite."""
from __future__ import annotations

import cv2
import numpy as np


def mean_fill(masked: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fill every missing pixel with the mean colour of the known pixels."""
    known = mask < 0.5
    color = masked[known].mean(axis=0) if known.any() else np.full(3, 0.5, np.float32)
    out = masked.copy()
    out[~known] = color
    return out


def telea_inpaint(masked: np.ndarray, mask: np.ndarray, radius: int = 3) -> np.ndarray:
    """OpenCV implementation of Telea's fast-marching inpainting (Telea, 2004)."""
    img8 = (np.clip(masked, 0, 1) * 255).round().astype(np.uint8)
    m8 = (mask > 0.5).astype(np.uint8) * 255
    out8 = cv2.inpaint(img8, m8, radius, cv2.INPAINT_TELEA)
    out = out8.astype(np.float32) / 255.0
    known = mask < 0.5
    out[known] = masked[known]  # keep known pixels bit-exact (undo uint8 rounding)
    return out


BASELINES = {"mean_fill": mean_fill, "telea": telea_inpaint}
