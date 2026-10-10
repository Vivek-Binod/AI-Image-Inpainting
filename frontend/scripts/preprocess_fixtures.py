"""Reference outputs of backend/app/imaging.py for src/lib/preprocess.test.ts.

    ../.venv/bin/python scripts/preprocess_fixtures.py   (from frontend/)
"""
import base64
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from backend.app.imaging import composite, model_size, prediction_to_original, to_model_input  # noqa: E402

CASES = [  # (W, H, work_size)
    (61, 47, 32),     # downscale both axes, reflect padding
    (200, 150, 64),   # downscale, smooth image
    (37, 21, 128),    # no resize, padding only
    (300, 20, 64),    # very wide: height clamped to 8
    (100, 30, 64),    # downscale, padding on both axes
    (64, 200, 128),   # height-limited, width needs no resampling
]


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def main() -> None:
    rng = np.random.default_rng(7)
    out = []
    for W, H, work in CASES:
        yy, xx = np.mgrid[0:H, 0:W]
        smooth = np.stack([127 + 120 * np.sin(xx / 7 + c) * np.cos(yy / 5 - c) for c in range(3)], -1)
        rgb = np.clip(smooth + rng.normal(0, 25, smooth.shape), 0, 255).round().astype(np.uint8)
        mask = np.zeros((H, W), np.uint8)
        for _ in range(3):
            y0, x0 = rng.integers(0, H), rng.integers(0, W)
            mask[y0:y0 + rng.integers(1, max(2, H // 3)), x0:x0 + rng.integers(1, max(2, W // 3))] = 1
        x = to_model_input(rgb, mask, work)
        pred = rng.random(x.image.shape, dtype=np.float32)
        full = prediction_to_original(pred, x.hw, mask.shape)
        out.append({
            "W": W, "H": H, "work": work, "hw": list(model_size((H, W), work)),
            "rgb": b64(rgb), "mask": b64(mask),
            "image": b64(x.image.transpose(2, 0, 1).astype(np.float32)), "modelMask": b64(x.mask.astype(np.float32)),
            "pred": b64(pred.transpose(2, 0, 1)), "full": b64(full), "result": b64(composite(rgb, full, mask)),
        })
    dest = Path(__file__).resolve().parents[1] / "src/lib/preprocess.fixtures.json"
    dest.write_text(json.dumps(out))
    print(f"wrote {dest} ({dest.stat().st_size / 1e3:.0f} kB, {len(out)} cases)")


if __name__ == "__main__":
    main()
