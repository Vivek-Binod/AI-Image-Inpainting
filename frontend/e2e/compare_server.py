"""Compare an in-browser result with the server code path on the same image and mask.

    .venv/bin/python frontend/e2e/compare_server.py <original> <mask.png> <result.png> <model> <work_size>

Runs backend.app.engine (PyTorch, CPU) and prints JSON with the largest per-channel
difference inside the hole and the fraction of hole values differing by more than 2/255.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from backend.app.config import Settings  # noqa: E402
from backend.app.engine import Engine  # noqa: E402
from backend.app.imaging import decode_image, decode_mask  # noqa: E402

orig_p, mask_p, res_p = map(Path, sys.argv[1:4])
model, work = sys.argv[4], int(sys.argv[5])
rgb = decode_image(orig_p.read_bytes(), 10**9).rgb
mask = decode_mask(mask_p.read_bytes(), rgb.shape[:2], 10**9)
browser = np.asarray(Image.open(res_p).convert("RGB")).astype(int)
settings = Settings.from_env()
settings.device = "cpu"
server = Engine(settings).inpaint(rgb, mask, model, work).result.astype(int)
hole = mask.astype(bool)
d = np.abs(browser - server)[hole]
print(json.dumps({"max_hole_diff": int(d.max()), "frac_over_2": float((d > 2).mean()),
                  "known_identical": bool((browser[~hole] == server[~hole]).all())}))
