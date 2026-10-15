"""Unit tests for mask generators (src/masks/generator.py)."""
from __future__ import annotations

import numpy as np
import pytest

from src.masks.generator import (
    MaskConfig,
    apply_mask,
    center_square_mask,
    free_form_mask,
    generate_mask,
    rect_mask,
)


def test_rect_mask_shape_and_values():
    rng = np.random.default_rng(42)
    cfg = MaskConfig(mask_type="rect", rect_min_frac=0.2, rect_max_frac=0.4, max_rects=2)
    mask = rect_mask(64, 64, rng, cfg)

    assert mask.shape == (64, 64)
    assert mask.dtype == np.float32
    assert set(np.unique(mask)).issubset({0.0, 1.0})
    assert mask.sum() > 0


def test_free_form_mask_shape_and_values():
    rng = np.random.default_rng(42)
    cfg = MaskConfig(mask_type="free_form", max_strokes=3)
    mask = free_form_mask(80, 80, rng, cfg)

    assert mask.shape == (80, 80)
    assert mask.dtype == np.float32
    assert set(np.unique(mask)).issubset({0.0, 1.0})
    assert mask.sum() > 0


@pytest.mark.parametrize("mtype", ["rect", "free_form", "mixed"])
def test_generate_mask_types(mtype: str):
    rng = np.random.default_rng(123)
    cfg = MaskConfig(mask_type=mtype)
    mask, family = generate_mask(48, 48, rng, cfg)

    assert mask.shape == (48, 48)
    assert mask.dtype == np.float32
    assert mask.sum() >= 1.0
    if mtype == "mixed":
        assert family in ("rect", "free_form", "both")
    else:
        assert family == mtype


def test_generate_mask_invalid_type():
    rng = np.random.default_rng(0)
    cfg = MaskConfig()
    with pytest.raises(ValueError, match="Unknown mask type"):
        generate_mask(32, 32, rng, cfg, mask_type="invalid_type")


def test_center_square_mask():
    h, w = 100, 100
    frac = 0.4
    mask = center_square_mask(h, w, frac=frac)

    assert mask.shape == (h, w)
    assert mask.dtype == np.float32
    # Center pixel must be 1.0 (masked)
    assert mask[50, 50] == 1.0
    # Outer corners must be 0.0 (known)
    assert mask[0, 0] == 0.0
    assert mask[0, 99] == 0.0
    assert mask[99, 0] == 0.0
    assert mask[99, 99] == 0.0
    # Expected area is roughly (h * frac) * (w * frac) = 40 * 40 = 1600
    assert mask.sum() == 1600.0


def test_apply_mask():
    h, w = 32, 32
    image = np.ones((h, w, 3), dtype=np.float32)
    mask = np.zeros((h, w), dtype=np.float32)
    mask[10:20, 10:20] = 1.0

    corrupted = apply_mask(image, mask)
    assert corrupted.shape == (h, w, 3)
    # Masked region should be zeroed
    assert np.all(corrupted[10:20, 10:20] == 0.0)
    # Unmasked region should be untouched
    assert np.all(corrupted[0:10, 0:10] == 1.0)


def test_apply_mask_shape_mismatch():
    image = np.ones((32, 32, 3), dtype=np.float32)
    mismatched_mask = np.zeros((24, 24), dtype=np.float32)
    with pytest.raises(ValueError, match="sizes differ"):
        apply_mask(image, mismatched_mask)
