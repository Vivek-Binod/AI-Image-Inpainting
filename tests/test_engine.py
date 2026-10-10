"""Checkpoint loading/compatibility and real inference with the trained models."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import torch

from backend.app.engine import Engine, ModelUnavailable
from tests.conftest import CHECKPOINTS, needs_checkpoints, rect_mask

pytestmark = needs_checkpoints


@pytest.fixture(scope="module")
def engine(settings):
    return Engine(settings)


def test_both_checkpoints_load_and_match_their_architecture(engine):
    assert engine.available == ["unet", "gated_unet"]
    for name, n_params in (("unet", 4_307_555), ("gated_unet", 8_615_011)):
        m = engine.models[name]
        assert m.available and m.error is None
        assert m.info["parameters"] == n_params
        assert m.info["epoch"] == 30
        assert m.info["training_resolution"] == 128
        assert m.model.block_type == ("gated" if name == "gated_unet" else "conv")
        assert not m.model.training  # eval mode


def test_weights_are_the_trained_ones_not_random(engine):
    ckpt = torch.load(CHECKPOINTS["unet"], map_location="cpu", weights_only=False)
    state = engine.models["unet"].model.state_dict()
    for k, v in ckpt["model_state"].items():
        assert torch.equal(state[k].cpu(), v.cpu()), k


def test_missing_checkpoint_is_reported_not_replaced(settings, tmp_path):
    s = dataclasses.replace(settings, checkpoints={"unet": tmp_path / "nope.pt", "gated_unet": CHECKPOINTS["gated_unet"]})
    eng = Engine(s)
    assert eng.available == ["gated_unet"]
    assert "not found" in eng.models["unet"].error.lower()
    assert eng.default_model == "gated_unet"
    with pytest.raises(ModelUnavailable, match="unavailable"):
        eng.inpaint(np.zeros((64, 64, 3), np.uint8), rect_mask(64, 64, 5, 20, 5, 20), "unet", 128)


def test_wrong_or_corrupt_checkpoint_is_rejected(settings, tmp_path):
    bad = tmp_path / "corrupt.pt"
    bad.write_bytes(b"not a checkpoint")
    s = dataclasses.replace(settings, checkpoints={"unet": CHECKPOINTS["gated_unet"], "gated_unet": bad})
    eng = Engine(s)
    assert eng.available == []
    assert "not 'unet'" in eng.models["unet"].error
    assert eng.models["gated_unet"].error
    assert eng.default_model is None


@pytest.mark.parametrize("model", ["unet", "gated_unet"])
@pytest.mark.parametrize("hw", [(200, 256), (128, 128), (517, 301), (37, 1000)])
def test_output_size_and_known_pixels(engine, photo, model, hw):
    from PIL import Image
    H, W = hw
    img = np.asarray(Image.fromarray(photo).resize((W, H)))
    mask = rect_mask(H, W, H // 4, H // 2, W // 3, W // 2)
    res = engine.inpaint(img, mask, model, 128)
    assert res.result.shape == (H, W, 3) and res.result.dtype == np.uint8
    known = mask == 0
    assert np.array_equal(res.result[known], img[known])          # bit-exact outside the hole
    assert not np.array_equal(res.result[~known], img[~known])    # the hole was actually filled
    assert res.work_hw == (max(8, round(H * min(1, 128 / max(H, W)))), max(8, round(W * min(1, 128 / max(H, W)))))


@pytest.mark.parametrize("model", ["unet", "gated_unet"])
def test_empty_and_small_masks(engine, photo, model):
    H, W = photo.shape[:2]
    res = engine.inpaint(photo, np.zeros((H, W), np.uint8), model, 128)
    assert np.array_equal(res.result, photo)
    small = rect_mask(H, W, 100, 103, 120, 124)
    res = engine.inpaint(photo, small, model, 128)
    assert np.array_equal(res.result[small == 0], photo[small == 0])
    assert int((res.result != photo).any(-1).sum()) <= small.sum()


@pytest.mark.parametrize("model", ["unet", "gated_unet"])
def test_hole_content_never_reaches_the_network(engine, photo, model):
    """Polarity check: pixels under the mask (1) are hidden from the model, so changing
    them cannot change the result. With an inverted mask this would fail."""
    H, W = photo.shape[:2]
    mask = rect_mask(H, W, 40, 120, 60, 180)
    other = photo.copy()
    other[mask == 1] = np.random.default_rng(0).integers(0, 256, (int(mask.sum()), 3), dtype=np.uint8)
    a = engine.inpaint(photo, mask, model, 128).result
    b = engine.inpaint(other, mask, model, 128).result
    assert np.array_equal(a, b)


@pytest.mark.parametrize("model", ["unet", "gated_unet"])
def test_prediction_is_close_to_truth_on_a_real_photo(engine, photo, model):
    """The trained model must beat a trivial mean fill inside the hole on a held-out photo."""
    H, W = photo.shape[:2]
    mask = rect_mask(H, W, 60, 110, 90, 160)
    out = engine.inpaint(photo, mask, model, 128).result.astype(np.float64)
    truth = photo.astype(np.float64)
    hole = mask == 1
    mse_model = ((out[hole] - truth[hole]) ** 2).mean()
    mean_fill = truth[~hole].mean(0)
    mse_mean = ((mean_fill - truth[hole]) ** 2).mean()
    assert mse_model < mse_mean


def test_cpu_inference_matches_shapes(settings, photo):
    s = dataclasses.replace(settings, device="cpu")
    eng = Engine(s)
    mask = rect_mask(*photo.shape[:2], 50, 90, 50, 90)
    res = eng.inpaint(photo, mask, "gated_unet", 128)
    assert res.result.shape == photo.shape
    assert np.array_equal(res.result[mask == 0], photo[mask == 0])


def test_existing_cli_inpainter_still_loads():
    """The original inference code (src/inference) keeps working with the same checkpoints."""
    from src.inference.inpainter import Inpainter
    inp = Inpainter(Path(CHECKPOINTS["unet"]), "cpu")
    img = np.random.default_rng(0).random((128, 128, 3)).astype(np.float32)
    mask = rect_mask(128, 128, 30, 60, 30, 60).astype(np.float32)
    out = inp(img, mask)
    assert out["composite"].shape == (128, 128, 3)
    assert np.array_equal(out["composite"][mask == 0], img[mask == 0])
