"""Check a downloaded result against the original upload and the mask the browser sent.

    .venv/bin/python frontend/e2e/verify.py <original> <mask.png> <result.png> <expect.json>

expect.json: {"holes": [[x0,y0,x1,y1], ...], "known": [[x0,y0,x1,y1], ...]} as fractions
of the displayed image: every "holes" box must be inside the mask (white) and every
"known" box outside it (black). Prints a JSON report; exit code 1 on any failure.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from backend.app.imaging import decode_image, decode_mask  # noqa: E402

orig_p, mask_p, res_p, exp_p = map(Path, sys.argv[1:5])
orig = decode_image(orig_p.read_bytes(), 10**9).rgb            # what the server sees (EXIF applied)
H, W = orig.shape[:2]
mask = decode_mask(mask_p.read_bytes(), (H, W), 10**9).astype(bool)
raw_mask = np.asarray(Image.open(mask_p).convert("L"))
result = np.asarray(Image.open(res_p).convert("RGB"))
exp = json.loads(exp_p.read_text())


def box(b):
    x0, y0, x1, y1 = b
    return slice(int(round(y0 * H)), int(round(y1 * H))), slice(int(round(x0 * W)), int(round(x1 * W)))


checks = {
    "result_size_matches_display_size": result.shape == orig.shape,
    "mask_is_binary": set(np.unique(raw_mask).tolist()) <= {0, 255},
    "mask_nonempty": bool(mask.any()),
    "known_pixels_bit_exact": bool(np.array_equal(result[~mask], orig[~mask])),
    "hole_pixels_changed": float((result[mask] != orig[mask]).any(-1).mean()) > 0.5,
}
for i, b in enumerate(exp.get("holes", [])):
    checks[f"hole_box_{i}_inside_mask"] = bool(mask[box(b)].all())
for i, b in enumerate(exp.get("known", [])):
    checks[f"known_box_{i}_outside_mask"] = not bool(mask[box(b)].any())
report = {"size": [W, H], "mask_size": list(raw_mask.shape[::-1]), "hole_fraction": float(mask.mean()),
          "checks": checks, "ok": all(checks.values())}
print(json.dumps(report))
sys.exit(0 if report["ok"] else 1)
