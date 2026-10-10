"""HTTP API: endpoints, validation, PNG output and failure handling."""
from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from fastapi.testclient import TestClient

from backend.app.main import create_app
from tests.conftest import CHECKPOINTS, decode_png, jpeg_bytes, needs_checkpoints, png_bytes, rect_mask, synthetic_image


@pytest.fixture(scope="module")
def client(settings):
    with TestClient(create_app(settings)) as c:
        yield c


def post(client, image: bytes, mask: bytes | None, **form):
    files = {"image": ("photo.png", image, "image/png")}
    if mask is not None:
        files["mask"] = ("mask.png", mask, "image/png")
    return client.post("/inpaint", files=files, data={k: str(v) for k, v in form.items()})


@needs_checkpoints
def test_health_and_info(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    assert set(r.json()["models_loaded"]) == {"unet", "gated_unet"}
    info = client.get("/model/info").json()
    assert info["default_model"] == "gated_unet" and info["default_work_size"] == 128
    names = {m["name"]: m for m in info["models"]}
    assert names["unet"]["checkpoint"] == "checkpoints/unet_4gb/best.pt"
    assert names["gated_unet"]["available"] and names["gated_unet"]["parameters"] == 8_615_011


@needs_checkpoints
@pytest.mark.parametrize("model", ["unet", "gated_unet"])
def test_inpaint_real_photo_returns_exact_png(client, photo, model):
    H, W = photo.shape[:2]
    mask = rect_mask(H, W, 50, 120, 70, 170)
    r = post(client, png_bytes(photo), png_bytes(mask * 255), model=model)
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "image/png"
    assert r.headers["x-model"] == model
    assert r.headers["x-image-size"] == f"{W}x{H}"
    assert r.headers["x-work-size"] == "128x100"
    assert abs(float(r.headers["x-hole-fraction"]) - mask.mean()) < 1e-6
    out = decode_png(r.content)
    assert out.shape == photo.shape
    assert np.array_equal(out[mask == 0], photo[mask == 0])
    assert not np.array_equal(out[mask == 1], photo[mask == 1])


@needs_checkpoints
def test_jpeg_with_exif_rotation_and_rgba_inputs(client):
    stored = synthetic_image(60, 100)  # displayed rotated: 60 wide x 100 high
    mask = rect_mask(100, 60, 30, 50, 10, 40)
    r = client.post("/inpaint", files={"image": ("p.jpg", jpeg_bytes(stored, 6), "image/jpeg"),
                                       "mask": ("m.png", png_bytes(mask * 255), "image/png")})
    assert r.status_code == 200, r.text
    assert decode_png(r.content).shape == (100, 60, 3)
    assert r.headers["x-image-size"] == "60x100"

    rgba = np.dstack([synthetic_image(64, 64), np.full((64, 64), 255, np.uint8)])
    r = post(client, png_bytes(rgba, "RGBA"), png_bytes(rect_mask(64, 64, 10, 30, 10, 30) * 255))
    assert r.status_code == 200 and r.headers["x-had-alpha"] == "1"
    assert np.array_equal(decode_png(r.content)[40:], rgba[40:, :, :3])


@needs_checkpoints
def test_default_model_and_work_size_override(client, photo):
    H, W = photo.shape[:2]
    m = png_bytes(rect_mask(H, W, 50, 90, 50, 90) * 255)
    r = post(client, png_bytes(photo), m)
    assert r.status_code == 200 and r.headers["x-model"] == "gated_unet"
    r = post(client, png_bytes(photo), m, work_size=256)
    assert r.status_code == 200 and r.headers["x-work-size"] == "256x200"


@pytest.mark.parametrize("case,status,fragment", [
    ("no_mask", 422, ""),
    ("empty_mask", 422, "mask is empty"),
    ("full_mask", 422, "whole image"),
    ("size_mismatch", 422, "must have the same size"),
    ("garbage_image", 422, "not a recognised image"),
    ("unknown_model", 422, "Unknown model"),
    ("bad_work_size", 422, "work_size"),
])
@needs_checkpoints
def test_validation_errors(client, case, status, fragment):
    img = png_bytes(synthetic_image(64, 96))
    mask = png_bytes(rect_mask(64, 96, 10, 20, 10, 20) * 255)
    kwargs = {}
    if case == "no_mask":
        mask = None
    elif case == "empty_mask":
        mask = png_bytes(np.zeros((64, 96), np.uint8))
    elif case == "full_mask":
        mask = png_bytes(np.full((64, 96), 255, np.uint8))
    elif case == "size_mismatch":
        mask = png_bytes(np.full((64, 64), 255, np.uint8))
    elif case == "garbage_image":
        img = b"hello"
    elif case == "unknown_model":
        kwargs["model"] = "resnet"
    elif case == "bad_work_size":
        kwargs["work_size"] = 4096
    r = post(client, img, mask, **kwargs)
    assert r.status_code == status, r.text
    assert fragment in str(r.json()["detail"])


@needs_checkpoints
def test_upload_size_limit(settings, photo):
    s = dataclasses.replace(settings, max_upload_bytes=1000)
    with TestClient(create_app(s)) as c:
        r = post(c, png_bytes(photo), png_bytes(rect_mask(*photo.shape[:2], 1, 5, 1, 5) * 255))
    assert r.status_code == 413 and "larger than" in r.json()["detail"]


def test_missing_checkpoints_give_clear_errors(settings, tmp_path):
    s = dataclasses.replace(settings, checkpoints={"unet": tmp_path / "a.pt", "gated_unet": tmp_path / "b.pt"})
    with TestClient(create_app(s)) as c:
        h = c.get("/health")
        assert h.status_code == 503 and h.json()["status"] == "unavailable"
        info = c.get("/model/info").json()
        assert all(not m["available"] and "not found" in m["error"].lower() for m in info["models"])
        r = post(c, png_bytes(synthetic_image(64, 64)), png_bytes(rect_mask(64, 64, 5, 30, 5, 30) * 255))
        assert r.status_code == 503 and "No trained checkpoint" in r.json()["detail"]


@needs_checkpoints
def test_one_missing_checkpoint_keeps_the_other_usable(settings, tmp_path):
    s = dataclasses.replace(settings, checkpoints={"unet": tmp_path / "a.pt", "gated_unet": CHECKPOINTS["gated_unet"]})
    with TestClient(create_app(s)) as c:
        img = png_bytes(synthetic_image(64, 64))
        mask = png_bytes(rect_mask(64, 64, 5, 30, 5, 30) * 255)
        r = post(c, img, mask, model="unet")
        assert r.status_code == 503 and "U-Net checkpoint is unavailable" in r.json()["detail"]
        assert post(c, img, mask, model="gated_unet").status_code == 200
