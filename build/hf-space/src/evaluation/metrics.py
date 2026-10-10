"""Reconstruction metrics. All functions take float images in [0, 1] shaped (H, W, 3)
and, where relevant, a mask shaped (H, W) with 1 = missing pixel.

MSE   = mean((I - I_hat)^2)
PSNR  = 10 * log10(MAX^2 / MSE), MAX = 1.0 for [0, 1] images (infinite if MSE = 0)
SSIM  = structural similarity (Wang et al., 2004), computed with scikit-image on
        the full RGB image (per channel, averaged) with data_range = 1.0 and
        scikit-image's default 7x7 uniform window.
Masked MSE/PSNR restrict the mean to missing pixels (all 3 channels).
"""
from __future__ import annotations

import math

import numpy as np
from skimage.metrics import structural_similarity

DATA_RANGE = 1.0


def _check(a: np.ndarray, b: np.ndarray) -> None:
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    if a.min() < -1e-6 or a.max() > 1 + 1e-6 or b.min() < -1e-6 or b.max() > 1 + 1e-6:
        raise ValueError("Images must be in [0, 1].")


def mse(a: np.ndarray, b: np.ndarray) -> float:
    _check(a, b)
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def psnr_from_mse(m: float, data_range: float = DATA_RANGE) -> float:
    return math.inf if m <= 0 else 10.0 * math.log10(data_range ** 2 / m)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    return psnr_from_mse(mse(a, b))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    _check(a, b)
    return float(structural_similarity(a.astype(np.float64), b.astype(np.float64),
                                       data_range=DATA_RANGE, channel_axis=2))


def masked_mse(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """MSE over pixels with mask == 1 (averaged over all channels of those pixels)."""
    _check(a, b)
    if mask.shape != a.shape[:2]:
        raise ValueError(f"Mask shape {mask.shape} does not match image {a.shape[:2]}")
    sel = mask > 0.5
    if not sel.any():
        raise ValueError("Mask has no missing pixels; masked metrics are undefined.")
    diff = (a.astype(np.float64) - b.astype(np.float64))[sel]
    return float(np.mean(diff ** 2))


def masked_psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    return psnr_from_mse(masked_mse(a, b, mask))


class LPIPSMetric:
    """Optional LPIPS (Zhang et al., 2018) using the AlexNet variant of the `lpips` package."""

    def __init__(self, device: str = "cpu"):
        import lpips  # noqa: WPS433 (optional dependency)
        import torch

        self._torch = torch
        self.device = device
        self.net = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    def __call__(self, a: np.ndarray, b: np.ndarray) -> float:
        t = self._torch
        to_t = lambda x: t.from_numpy(x.transpose(2, 0, 1)[None].astype(np.float32)).to(self.device) * 2 - 1
        with t.no_grad():
            return float(self.net(to_t(a), to_t(b)).item())
