"""Unit tests for evaluation metrics (src/evaluation/metrics.py)."""
from __future__ import annotations

import math
import numpy as np
import pytest

from src.evaluation.metrics import (
    masked_mse,
    masked_psnr,
    mse,
    psnr,
    psnr_from_mse,
    ssim,
)


def test_mse_identical_images():
    img = np.full((32, 32, 3), 0.5, dtype=np.float32)
    assert mse(img, img) == 0.0


def test_psnr_identical_images():
    img = np.full((32, 32, 3), 0.5, dtype=np.float32)
    assert psnr(img, img) == math.inf


def test_psnr_known_difference():
    # If error is uniformly 0.1, MSE = 0.01, PSNR = 10 * log10(1.0 / 0.01) = 20.0 dB
    a = np.full((16, 16, 3), 0.5, dtype=np.float32)
    b = np.full((16, 16, 3), 0.6, dtype=np.float32)
    computed_psnr = psnr(a, b)
    assert pytest.approx(computed_psnr, rel=1e-3) == 20.0


def test_ssim_range_and_identity():
    a = np.random.default_rng(42).uniform(0.0, 1.0, (32, 32, 3)).astype(np.float32)
    # Identity comparison should be 1.0
    assert pytest.approx(ssim(a, a), abs=1e-4) == 1.0

    b = np.random.default_rng(99).uniform(0.0, 1.0, (32, 32, 3)).astype(np.float32)
    val = ssim(a, b)
    assert -1.0 <= val <= 1.0


def test_masked_mse_and_psnr():
    a = np.zeros((20, 20, 3), dtype=np.float32)
    b = np.zeros((20, 20, 3), dtype=np.float32)

    # Set difference in bottom-right 10x10 corner
    b[10:, 10:] = 0.5

    # Mask only selects that corner
    mask = np.zeros((20, 20), dtype=np.float32)
    mask[10:, 10:] = 1.0

    # MSE on the mask should be 0.25 (0.5^2)
    assert pytest.approx(masked_mse(a, b, mask), rel=1e-4) == 0.25

    # PSNR = 10 * log10(1.0 / 0.25) = 10 * log10(4) ~ 6.02 dB
    assert pytest.approx(masked_psnr(a, b, mask), rel=1e-3) == 6.0206


def test_masked_metrics_empty_mask():
    a = np.zeros((10, 10, 3), dtype=np.float32)
    b = np.zeros((10, 10, 3), dtype=np.float32)
    empty_mask = np.zeros((10, 10), dtype=np.float32)
    with pytest.raises(ValueError, match="Mask has no missing pixels"):
        masked_mse(a, b, empty_mask)


def test_metrics_shape_mismatch():
    a = np.zeros((10, 10, 3), dtype=np.float32)
    b = np.zeros((10, 12, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="Shape mismatch"):
        mse(a, b)


def test_metrics_out_of_bounds():
    a = np.full((10, 10, 3), 1.5, dtype=np.float32)
    b = np.zeros((10, 10, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="Images must be in"):
        mse(a, b)
