"""Preprocessing, mask polarity/alignment and the exact-composite invariant (no model needed)."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from backend.app.imaging import (
    InputError, composite, decode_image, decode_mask, encode_png, model_size, prediction_to_original,
    to_model_input,
)
from tests.conftest import decode_png, jpeg_bytes, png_bytes, rect_mask, synthetic_image

MAXPX = 40_000_000


# ----- decoding ---------------------------------------------------------------

def test_decode_png_roundtrip_is_exact():
    img = synthetic_image(37, 53)
    out = decode_image(png_bytes(img), MAXPX)
    assert out.rgb.dtype == np.uint8 and out.rgb.shape == (37, 53, 3)
    assert np.array_equal(out.rgb, img)
    assert out.format == "PNG" and not out.had_alpha


def test_decode_jpeg_and_grey_and_palette():
    img = synthetic_image(40, 60)
    assert decode_image(jpeg_bytes(img), MAXPX).rgb.shape == (40, 60, 3)
    grey = decode_image(png_bytes(img[..., 0], "L"), MAXPX).rgb
    assert grey.shape == (40, 60, 3) and np.array_equal(grey[..., 0], img[..., 0])
    pal = Image.fromarray(img).convert("P")
    import io
    buf = io.BytesIO()
    pal.save(buf, format="PNG")
    assert decode_image(buf.getvalue(), MAXPX).rgb.shape == (40, 60, 3)


def test_decode_16bit_greyscale():
    import io
    arr = (np.arange(40 * 30, dtype=np.uint16).reshape(30, 40) * 50)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    out = decode_image(buf.getvalue(), MAXPX).rgb
    assert out.shape == (30, 40, 3) and out.max() == 255 and out.min() == 0


def test_alpha_is_flattened_on_white():
    rgba = np.zeros((20, 20, 4), np.uint8)
    rgba[..., 0] = 200
    rgba[:10, :, 3] = 255   # top half opaque red-ish, bottom half fully transparent
    out = decode_image(png_bytes(rgba, "RGBA"), MAXPX)
    assert out.had_alpha
    assert (out.rgb[:10] == [200, 0, 0]).all()
    assert (out.rgb[10:] == 255).all()


def test_exif_orientation_is_applied():
    # Orientation 6 = the stored pixels must be rotated 90 deg clockwise for display.
    img = synthetic_image(30, 50)  # stored 50 wide x 30 high
    out = decode_image(jpeg_bytes(img, exif_orientation=6), MAXPX).rgb
    assert out.shape == (50, 30, 3)  # displayed 30 wide x 50 high, as browsers show it


@pytest.mark.parametrize("data,msg", [
    (b"", "empty"),
    (b"definitely not an image", "not a recognised image"),
])
def test_decode_rejects_garbage(data, msg):
    with pytest.raises(InputError, match=msg):
        decode_image(data, MAXPX)


def test_decode_rejects_too_many_pixels_and_tiny_images():
    with pytest.raises(InputError, match="limit"):
        decode_image(png_bytes(synthetic_image(100, 100)), max_pixels=5000)
    with pytest.raises(InputError, match="at least"):
        decode_image(png_bytes(synthetic_image(8, 100)), MAXPX)


# ----- mask polarity and alignment ------------------------------------------

def test_mask_polarity_white_is_missing():
    m = np.zeros((10, 12), np.uint8)
    m[2:5, 3:7] = 255
    out = decode_mask(png_bytes(m, "L"), (10, 12), MAXPX)
    assert out.dtype == np.uint8 and set(np.unique(out)) == {0, 1}
    assert np.array_equal(out, (m > 0).astype(np.uint8))


def test_mask_threshold_and_transparent_strokes():
    m = np.full((6, 6), 127, np.uint8)
    m[0, 0] = 128
    out = decode_mask(png_bytes(m, "L"), (6, 6), MAXPX)
    assert out.sum() == 1 and out[0, 0] == 1
    rgba = np.zeros((6, 6, 4), np.uint8)
    rgba[1:3, 1:3] = [255, 255, 255, 255]  # white strokes on a transparent canvas
    out = decode_mask(png_bytes(rgba, "RGBA"), (6, 6), MAXPX)
    assert np.array_equal(out, rect_mask(6, 6, 1, 3, 1, 3))


def test_mask_size_mismatch_is_rejected():
    with pytest.raises(InputError, match="must have the same size"):
        decode_mask(png_bytes(np.zeros((10, 20), np.uint8), "L"), (10, 21 * 2), MAXPX)


def test_lower_resolution_mask_is_upsampled_in_place():
    small = rect_mask(50, 80, 10, 20, 40, 60) * 255
    out = decode_mask(png_bytes(small.astype(np.uint8), "L"), (100, 160), MAXPX)
    assert np.array_equal(out, rect_mask(100, 160, 20, 40, 80, 120))


@pytest.mark.parametrize("hw,work,expected", [
    ((200, 256), 128, (100, 128)),
    ((256, 256), 128, (128, 128)),
    ((3024, 4032), 128, (96, 128)),
    ((100, 60), 128, (100, 60)),       # never upscaled
    ((4000, 1000), 256, (256, 64)),
])
def test_model_size_preserves_aspect(hw, work, expected):
    assert model_size(hw, work) == expected


def test_model_input_is_padded_to_multiple_of_8_and_mask_aligned():
    H, W = 300, 401
    img = synthetic_image(H, W)
    mask = rect_mask(H, W, 60, 150, 100, 200)
    x = to_model_input(img, mask, 128, resample="dilate")
    h, w = x.hw
    assert (h, w) == model_size((H, W), 128) == (96, 128)
    assert x.image.shape[0] % 8 == 0 and x.image.shape[1] % 8 == 0
    assert x.image.shape[:2] == x.mask.shape
    assert set(np.unique(x.mask)) <= {0.0, 1.0}
    # The low-resolution hole covers the scaled rectangle plus the bicubic support
    # (2 low-res px) around it, and nothing further away.
    sy, sx = H / h, W / w
    ys, xs = np.nonzero(x.mask[:h, :w])
    for lo, hi, s, idx in ((60, 150, sy, ys), (100, 200, sx, xs)):
        assert lo / s - 3.5 <= idx.min() <= lo / s - 1.5      # covers the hole + bicubic support
        assert hi / s + 0.5 <= idx.max() <= hi / s + 2.5
    # Hole pixels are zeroed in the network input.
    assert (x.image[x.mask == 1] == 0).all()
    # Hole far from the padding -> padding is known.
    assert x.mask[h:].sum() == 0 and x.mask[:, w:].sum() == 0


def test_hole_coverage_matches_brute_force():
    from backend.app.imaging import _hole_coverage
    rng = np.random.default_rng(3)
    mask = (rng.random((61, 83)) > 0.97).astype(np.uint8)
    h, w = 23, 31
    got = _hole_coverage(mask, (h, w), radius=2.0)
    sy, sx = 61 / h, 83 / w
    for i in range(h):
        for j in range(w):
            cy, cx = (i + 0.5) * sy, (j + 0.5) * sx
            y0, y1 = max(0, int(np.floor(cy - 2 * sy))), min(61, int(np.ceil(cy + 2 * sy)))
            x0, x1 = max(0, int(np.floor(cx - 2 * sx))), min(83, int(np.ceil(cx + 2 * sx)))
            assert got[i, j] == bool(mask[y0:y1, x0:x1].any()), (i, j)


def test_hole_content_cannot_leak_into_known_model_pixels():
    H, W = 301, 457
    img = synthetic_image(H, W)
    mask = rect_mask(H, W, 100, 180, 150, 260)
    other = img.copy()
    other[mask == 1] = 255 - other[mask == 1]
    for mode in ("normalized", "dilate"):
        a, b = to_model_input(img, mask, 128, mode), to_model_input(other, mask, 128, mode)
        assert np.array_equal(a.mask, b.mask)
        assert np.array_equal(a.image, b.image)


def test_normalized_resampling_hole_is_the_exact_footprint():
    """With a hole aligned to the downscale grid, the low-res hole is exactly the scaled hole
    (not enlarged) and known values match a plain resize of an image without the hole."""
    H, W = 256, 256
    img = synthetic_image(H, W)
    mask = rect_mask(H, W, 64, 128, 96, 160)
    x = to_model_input(img, mask, 128)
    assert np.array_equal(x.mask, rect_mask(128, 128, 32, 64, 48, 80).astype(np.float32))
    plain = np.asarray(Image.fromarray(img).resize((128, 128), Image.BICUBIC), dtype=np.float32) / 255
    far = np.ones((128, 128), bool)
    far[28:68, 44:84] = False  # away from the hole the two resamplings agree
    assert np.abs(x.image[far] - plain[far]).max() < 2 / 255  # plain resize is rounded to uint8


def test_prediction_is_cropped_and_resized_back():
    pred = np.random.default_rng(0).random((104, 128, 3)).astype(np.float32)
    out = prediction_to_original(pred, (100, 128), (200, 256))
    assert out.shape == (200, 256, 3) and out.dtype == np.uint8


# ----- the exact composite invariant ------------------------------------------

def test_composite_preserves_known_pixels_exactly():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, (64, 80, 3), dtype=np.uint8)
    pred = rng.integers(0, 256, (64, 80, 3), dtype=np.uint8)
    mask = rect_mask(64, 80, 10, 30, 20, 50)
    out = composite(img, pred, mask)
    known = mask == 0
    assert np.array_equal(out[known], img[known])
    assert np.array_equal(out[~known], pred[~known])


def test_composite_empty_mask_returns_original():
    img = synthetic_image(33, 47)
    pred = 255 - img
    out = composite(img, pred, np.zeros((33, 47), np.uint8))
    assert np.array_equal(out, img)


def test_png_encoding_keeps_composite_exact():
    img = synthetic_image(50, 70)
    mask = rect_mask(50, 70, 5, 8, 5, 9)  # a small region
    out = composite(img, np.zeros_like(img), mask)
    back = decode_png(encode_png(out))
    assert np.array_equal(back, out)
    assert np.array_equal(back[mask == 0], img[mask == 0])
