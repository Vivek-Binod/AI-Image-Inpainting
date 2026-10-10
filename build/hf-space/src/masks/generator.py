"""Binary mask generation for image inpainting.

Mask convention used everywhere in this project:
    1 = missing pixel (hole to be filled)
    0 = known pixel

Masks are returned as float32 numpy arrays of shape (H, W) with values in {0, 1}.
Every generator takes an explicit ``numpy.random.Generator`` so that masks are
reproducible: the evaluation code seeds one generator per test image, which
gives every method exactly the same masks.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

MASK_TYPES = ("rect", "free_form", "mixed")


@dataclass
class MaskConfig:
    """Parameters controlling mask generation (all sizes are fractions of the image side)."""

    mask_type: str = "mixed"            # one of MASK_TYPES
    rect_min_frac: float = 0.15         # min rectangle side as fraction of image side
    rect_max_frac: float = 0.5          # max rectangle side as fraction of image side
    max_rects: int = 3                  # rectangles per mask drawn from [1, max_rects]
    stroke_max_vertices: int = 8        # vertices per brush stroke
    stroke_max_width_frac: float = 0.12 # max brush width as fraction of image side
    stroke_max_len_frac: float = 0.3    # max segment length as fraction of image side
    max_strokes: int = 4                # strokes per mask drawn from [1, max_strokes]
    # probability of each family when mask_type == "mixed"
    mixed_probs: dict = field(default_factory=lambda: {"rect": 0.35, "free_form": 0.45, "both": 0.2})


def rect_mask(h: int, w: int, rng: np.random.Generator, cfg: MaskConfig,
              num_rects: int | None = None) -> np.ndarray:
    """One or more axis-aligned rectangular holes at random positions."""
    mask = np.zeros((h, w), np.float32)
    n = num_rects if num_rects is not None else int(rng.integers(1, cfg.max_rects + 1))
    for _ in range(n):
        rh = int(rng.uniform(cfg.rect_min_frac, cfg.rect_max_frac) * h)
        rw = int(rng.uniform(cfg.rect_min_frac, cfg.rect_max_frac) * w)
        rh, rw = max(rh, 1), max(rw, 1)
        top = int(rng.integers(0, h - rh + 1))
        left = int(rng.integers(0, w - rw + 1))
        mask[top:top + rh, left:left + rw] = 1.0
    return mask


def free_form_mask(h: int, w: int, rng: np.random.Generator, cfg: MaskConfig) -> np.ndarray:
    """Irregular brush-stroke holes.

    Each stroke is a random walk: start at a random point, then repeatedly pick
    a random direction and length and draw a thick line segment with round
    joints. This follows the spirit of the free-form mask generation of Yu et
    al. (2019) but is a simplified re-implementation.
    """
    mask = np.zeros((h, w), np.uint8)
    side = min(h, w)
    max_width = max(int(cfg.stroke_max_width_frac * side), 2)
    max_len = max(int(cfg.stroke_max_len_frac * side), 2)
    for _ in range(int(rng.integers(1, cfg.max_strokes + 1))):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        width = int(rng.integers(max(max_width // 3, 1), max_width + 1))
        for _ in range(int(rng.integers(1, cfg.stroke_max_vertices + 1))):
            angle = rng.uniform(0, 2 * np.pi)
            length = rng.uniform(max_len / 4, max_len)
            nx = int(np.clip(x + length * np.cos(angle), 0, w - 1))
            ny = int(np.clip(y + length * np.sin(angle), 0, h - 1))
            cv2.line(mask, (x, y), (nx, ny), 1, width)
            cv2.circle(mask, (nx, ny), width // 2, 1, -1)
            x, y = nx, ny
    return mask.astype(np.float32)


def generate_mask(h: int, w: int, rng: np.random.Generator, cfg: MaskConfig | None = None,
                  mask_type: str | None = None) -> tuple[np.ndarray, str]:
    """Generate a mask and return ``(mask, family)``.

    ``family`` is the type actually drawn ("rect", "free_form" or "both"),
    which matters when ``mask_type == "mixed"``.
    """
    cfg = cfg or MaskConfig()
    mtype = mask_type or cfg.mask_type
    if mtype not in MASK_TYPES:
        raise ValueError(f"Unknown mask type '{mtype}'. Expected one of {MASK_TYPES}.")
    if mtype == "mixed":
        names = list(cfg.mixed_probs.keys())
        probs = np.array([cfg.mixed_probs[k] for k in names], dtype=np.float64)
        family = names[int(rng.choice(len(names), p=probs / probs.sum()))]
    else:
        family = mtype

    if family == "rect":
        mask = rect_mask(h, w, rng, cfg)
    elif family == "free_form":
        mask = free_form_mask(h, w, rng, cfg)
    elif family == "both":
        mask = np.maximum(rect_mask(h, w, rng, cfg, num_rects=1), free_form_mask(h, w, rng, cfg))
    else:
        raise ValueError(f"Unknown mask family '{family}'.")

    # Guarantee at least one missing pixel so masked-region metrics are defined.
    if mask.sum() == 0:
        mask[h // 2, w // 2] = 1.0
    return mask, family


def center_square_mask(h: int, w: int, frac: float = 0.4) -> np.ndarray:
    """Deterministic centred square hole (useful for the CLI and demos)."""
    mask = np.zeros((h, w), np.float32)
    sh, sw = int(h * frac), int(w * frac)
    top, left = (h - sh) // 2, (w - sw) // 2
    mask[top:top + sh, left:left + sw] = 1.0
    return mask


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return the corrupted image I * (1 - M). ``image`` is (H, W, C), ``mask`` is (H, W)."""
    if image.shape[:2] != mask.shape:
        raise ValueError(f"Image {image.shape[:2]} and mask {mask.shape} sizes differ.")
    return image * (1.0 - mask[..., None])
