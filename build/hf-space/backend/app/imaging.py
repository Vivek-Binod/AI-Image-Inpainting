"""Decoding uploads, model-resolution preprocessing and exact compositing.

Conventions (shared with ``src/``): images are RGB, masks are binary with
1 = missing pixel (to reconstruct) and 0 = known pixel (preserved).

Coordinate frames handled here:

* original  - the uploaded image after EXIF orientation is applied (this is
              also what browsers display, so it is the frame the user draws in);
* mask      - the uploaded mask; it must be the original size or the same
              aspect ratio at a lower resolution (it is then upsampled with
              nearest-neighbour, which keeps it binary and aligned);
* model     - the original resized with its aspect ratio preserved so that the
              longer side equals ``work_size`` and padded (reflection) to a
              multiple of 8, as the U-Net requires. Known pixels are resampled
              from known pixels only (see ``to_model_input``).
"""
from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

# Formats PIL decodes that browsers can also display, so the user can draw on them.
SUPPORTED_FORMATS = {"PNG", "JPEG", "WEBP", "BMP", "GIF", "TIFF", "MPO"}


class InputError(ValueError):
    """An invalid upload; the message is shown to the user."""


@dataclass
class DecodedImage:
    rgb: np.ndarray          # (H, W, 3) uint8, EXIF orientation applied, alpha flattened on white
    format: str
    had_alpha: bool


def _open(data: bytes, what: str, max_pixels: int) -> Image.Image:
    if not data:
        raise InputError(f"The {what} file is empty.")
    try:
        img = Image.open(io.BytesIO(data))
    except UnidentifiedImageError:
        raise InputError(f"The {what} file is not a recognised image. Use PNG, JPEG, WebP or BMP.") from None
    w, h = img.size
    if w * h > max_pixels:
        raise InputError(f"The {what} is {w}x{h} ({w * h / 1e6:.1f} MP); the limit is {max_pixels / 1e6:.0f} MP.")
    return img


def decode_image(data: bytes, max_pixels: int, min_side: int = 16) -> DecodedImage:
    img = _open(data, "image", max_pixels)
    fmt = img.format or "unknown"
    if fmt not in SUPPORTED_FORMATS:
        raise InputError(f"Unsupported image format '{fmt}'. Use PNG, JPEG, WebP or BMP.")
    try:
        img = ImageOps.exif_transpose(img)  # browsers display images in EXIF orientation
        img.load()
    except (OSError, SyntaxError) as exc:
        raise InputError(f"The image could not be decoded ({exc}).") from None
    if min(img.size) < min_side:
        raise InputError(f"The image is {img.size[0]}x{img.size[1]}; both sides must be at least {min_side} px.")

    if img.mode in ("I;16", "I;16B", "I;16L", "I"):  # 16/32-bit greyscale: scale to 8 bit
        arr = np.asarray(img, dtype=np.float64)
        img = Image.fromarray((arr / max(arr.max(), 1) * 255).round().astype(np.uint8), "L")
    had_alpha = img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info)
    if had_alpha:
        # Transparent pixels are shown on white in the web UI, so flatten on white here too.
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, rgba)
    rgb = np.array(img.convert("RGB"), dtype=np.uint8)
    return DecodedImage(rgb=rgb, format=fmt, had_alpha=had_alpha)


def decode_mask(data: bytes, image_hw: tuple[int, int], max_pixels: int) -> np.ndarray:
    """Return a (H, W) uint8 mask in {0, 1} aligned with the image (1 = missing).

    A pixel is missing when its grey value is > 127 (transparent pixels count as
    black), so both opaque black/white masks and white strokes on a transparent
    canvas work.
    """
    img = _open(data, "mask", max_pixels)
    try:
        img.load()
    except OSError as exc:
        raise InputError(f"The mask could not be decoded ({exc}).") from None
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        img = Image.alpha_composite(Image.new("RGBA", rgba.size, (0, 0, 0, 255)), rgba)
    mask = (np.asarray(img.convert("L")) > 127).astype(np.uint8)

    H, W = image_hw
    mh, mw = mask.shape
    if (mh, mw) != (H, W):
        # Allow a lower-resolution mask of the same aspect ratio (the browser caps
        # its drawing canvas for very large photos); reject anything else, because
        # stretching a mismatched mask would silently misalign the hole.
        same_aspect = abs(H * mw / W - mh) <= 1.0 and abs(W * mh / H - mw) <= 1.0
        if not same_aspect or mw > W or mh > H:
            raise InputError(f"The mask is {mw}x{mh} but the image is {W}x{H} (after EXIF rotation). "
                             "They must have the same size or the same aspect ratio.")
        mask = np.asarray(Image.fromarray(mask * 255).resize((W, H), Image.NEAREST)) > 127
        mask = mask.astype(np.uint8)
    return mask


@dataclass
class ModelInput:
    image: np.ndarray        # (h_pad, w_pad, 3) float32 in [0, 1]
    mask: np.ndarray         # (h_pad, w_pad) float32 in {0, 1}
    hw: tuple[int, int]      # model-resolution size before padding


def model_size(hw: tuple[int, int], work_size: int) -> tuple[int, int]:
    """Aspect-preserving size whose longer side is ``work_size`` (never upscales)."""
    H, W = hw
    scale = min(1.0, work_size / max(H, W))
    return max(8, round(H * scale)), max(8, round(W * scale))


def _hole_coverage(mask: np.ndarray, hw: tuple[int, int], radius: float) -> np.ndarray:
    """Low-resolution mask: a pixel is missing if ANY full-resolution hole pixel lies within
    ``radius`` low-resolution pixels of its centre (computed exactly with an integral image)."""
    H, W = mask.shape
    h, w = hw
    sy, sx = H / h, W / w

    def ranges(n_out: int, scale: float, n_in: int):
        c = (np.arange(n_out) + 0.5) * scale
        lo = np.clip(np.floor(c - radius * scale), 0, n_in).astype(np.int64)
        hi = np.clip(np.ceil(c + radius * scale), 0, n_in).astype(np.int64)
        return lo, hi

    y0, y1 = ranges(h, sy, H)
    x0, x1 = ranges(w, sx, W)
    integral = np.zeros((H + 1, W + 1), np.int64)
    integral[1:, 1:] = mask.astype(np.int64).cumsum(0).cumsum(1)
    total = (integral[y1][:, x1] - integral[y0][:, x1] - integral[y1][:, x0] + integral[y0][:, x0])
    return total > 0


RESAMPLE_MODES = ("normalized", "dilate")


def _resize_f(arr: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """Bicubic resize of a float (H, W) array."""
    h, w = hw
    return np.asarray(Image.fromarray(arr.astype(np.float32), "F").resize((w, h), Image.BICUBIC), dtype=np.float32)


def to_model_input(rgb: np.ndarray, mask: np.ndarray, work_size: int, resample: str = "normalized") -> ModelInput:
    """Prepare the network input. Hole content never reaches the network in either mode:

    * ``normalized`` (default): known pixels are downscaled with normalised bicubic
      filtering (sum of w * I * (1 - M) / sum of w * (1 - M)), so every low-resolution value
      is built from known pixels only; a low-resolution pixel is missing if its own
      footprint contains any hole pixel. The hole is not enlarged beyond that.
    * ``dilate``: plain bicubic downscaling, with every low-resolution pixel whose bicubic
      window (2 px) touches the hole marked missing. Simpler, but enlarges the hole.
    """
    if resample not in RESAMPLE_MODES:
        raise ValueError(f"resample must be one of {RESAMPLE_MODES}")
    H, W = mask.shape
    h, w = model_size((H, W), work_size)
    if (h, w) == (H, W):
        img = rgb.astype(np.float32) / 255.0
        m = mask > 0
    elif resample == "dilate":
        img = np.asarray(Image.fromarray(rgb).resize((w, h), Image.BICUBIC), dtype=np.float32) / 255.0
        # PIL's bicubic downscaling reads input pixels within 2 output pixels of each centre.
        m = _hole_coverage(mask, (h, w), radius=2.0)
    else:
        known = (mask == 0).astype(np.float32)
        den = _resize_f(known, (h, w))
        img = np.stack([_resize_f(rgb[..., c].astype(np.float32) / 255.0 * known, (h, w)) for c in range(3)], -1)
        img = np.clip(img / np.maximum(den, 1e-6)[..., None], 0.0, 1.0)
        m = _hole_coverage(mask, (h, w), radius=0.5) | (den < 0.5)
    m = m.astype(np.float32)
    img = img * (1.0 - m[..., None])
    ph, pw = (-h) % 8, (-w) % 8
    if ph or pw:
        # Reflection padding keeps image statistics natural at the border. The mask is
        # reflected too, so reflected hole pixels stay holes; the padding is cropped after.
        mode = "reflect" if h > ph and w > pw else "edge"
        img = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode=mode)
        m = np.pad(m, ((0, ph), (0, pw)), mode=mode)
    return ModelInput(image=img, mask=m, hw=(h, w))


def prediction_to_original(pred: np.ndarray, hw_model: tuple[int, int], hw_orig: tuple[int, int]) -> np.ndarray:
    """Crop the padding and resize a (h_pad, w_pad, 3) float prediction to the original size (uint8)."""
    h, w = hw_model
    pred8 = (np.clip(pred[:h, :w], 0, 1) * 255).round().astype(np.uint8)
    H, W = hw_orig
    if (h, w) == (H, W):
        return pred8
    return np.asarray(Image.fromarray(pred8).resize((W, H), Image.BICUBIC), dtype=np.uint8)


def composite(original: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """result = I * (1 - M) + P * M, in uint8 so known pixels are bit-identical to the original."""
    if original.shape != prediction.shape or original.shape[:2] != mask.shape:
        raise ValueError(f"Shape mismatch: image {original.shape}, prediction {prediction.shape}, mask {mask.shape}")
    return np.where(mask[..., None].astype(bool), prediction, original).astype(np.uint8)


def encode_png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG", compress_level=6)
    return buf.getvalue()
