"""Export the two best checkpoints to ONNX for in-browser inference (static website build).

Writes <out>/<name>.onnx (dynamic height/width, both multiples of 8) and <out>/models.json,
the same model description the backend serves at /model/info. Each export is checked
against PyTorch with onnxruntime on several input sizes before it is kept.

    .venv/bin/python scripts/export_onnx.py [--out build/onnx]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402

from backend.app.config import Settings  # noqa: E402
from backend.app.engine import Engine  # noqa: E402

CHECK_SIZES = ((128, 128), (96, 128), (128, 88), (256, 192), (64, 64))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="build/onnx")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    settings = Settings.from_env()
    settings.device = "cpu"
    engine = Engine(settings)
    if engine.available != list(settings.checkpoints):
        sys.exit(f"Not every checkpoint loaded: {engine.describe()['models']}")

    rng = np.random.default_rng(0)
    for name in engine.available:
        model = engine.models[name].model.eval()
        path = out / f"{name}.onnx"
        x = torch.rand(1, 3, 128, 128)
        m = (torch.rand(1, 1, 128, 128) > 0.7).float()
        torch.onnx.export(model, (x * (1 - m), m), str(path), dynamo=False, opset_version=args.opset,
                          input_names=["masked", "mask"], output_names=["prediction"],
                          dynamic_axes={"masked": {2: "height", 3: "width"}, "mask": {2: "height", 3: "width"},
                                        "prediction": {2: "height", 3: "width"}})
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        worst = 0.0
        for h, w in CHECK_SIZES:
            img = rng.random((1, 3, h, w), dtype=np.float32)
            msk = (rng.random((1, 1, h, w)) > 0.6).astype(np.float32)
            img *= 1 - msk
            with torch.no_grad():
                ref = model(torch.from_numpy(img), torch.from_numpy(msk)).numpy()
            got = sess.run(None, {"masked": img, "mask": msk})[0]
            worst = max(worst, float(np.abs(ref - got).max()))
        if worst > 1e-4:
            path.unlink()
            sys.exit(f"{name}: ONNX output differs from PyTorch by {worst:.2e}; export discarded.")
        print(f"{name}: {path.relative_to(ROOT)} ({path.stat().st_size / 1e6:.1f} MB), "
              f"max |onnx - torch| = {worst:.1e} over sizes {CHECK_SIZES}")

    info = engine.describe()
    # In the browser the network runs single-threaded in WebAssembly, so larger work sizes are slow.
    info.update(device="browser", cuda_device=None, max_work_size=256)
    for m in info["models"]:
        m["checkpoint"] = f"models/{m['name']}.onnx (from {m['checkpoint']})"
    (out / "models.json").write_text(json.dumps(info, indent=2))
    print(f"wrote {(out / 'models.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
