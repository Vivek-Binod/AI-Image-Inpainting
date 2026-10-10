"""Download (optional), extract, validate and split the Places365 val_256 images.

Usage:
    python scripts/prepare_data.py --download            # download + extract + split
    python scripts/prepare_data.py                       # extract existing tar + split
    python scripts/prepare_data.py --image-dir some/dir  # use any folder of images instead

Writes data/splits/{train,val,test}.txt (paths relative to the data root) and
data/splits/dataset_stats.json.
"""
from __future__ import annotations

import argparse
import sys
import tarfile
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from src.datasets.inpainting_dataset import IMAGE_EXTENSIONS, is_valid_image, make_splits  # noqa: E402
from src.utils.common import PROJECT_ROOT, save_json  # noqa: E402

PLACES_URL = "https://data.csail.mit.edu/places/places365/val_256.tar"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--download", action="store_true", help="download val_256.tar (~525 MB) from MIT CSAIL")
    ap.add_argument("--raw-dir", default="data/raw", help="where the tar is / is extracted")
    ap.add_argument("--image-dir", default=None, help="use an existing folder of images instead of Places365")
    ap.add_argument("--splits-dir", default="data/splits")
    ap.add_argument("--val-count", type=int, default=3000)
    ap.add_argument("--test-count", type=int, default=3500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-validation", action="store_true", help="skip decoding every image")
    args = ap.parse_args()

    raw = PROJECT_ROOT / args.raw_dir
    if args.image_dir:
        image_dir = Path(args.image_dir).resolve()
    else:
        tar_path = raw / "val_256.tar"
        image_dir = raw / "val_256"
        if args.download and not tar_path.exists():
            raw.mkdir(parents=True, exist_ok=True)
            print(f"Downloading {PLACES_URL} -> {tar_path}")
            urllib.request.urlretrieve(PLACES_URL, tar_path)
        if not image_dir.is_dir():
            if not tar_path.exists():
                sys.exit(f"Neither {image_dir} nor {tar_path} exists. Run with --download or see data/README.md.")
            print(f"Extracting {tar_path} ...")
            with tarfile.open(tar_path) as tf:
                tf.extractall(raw, filter="data")
    if not image_dir.is_dir():
        sys.exit(f"Image directory not found: {image_dir}")

    files = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    print(f"Found {len(files)} candidate image files in {image_dir}")
    if args.skip_validation:
        valid = files
    else:
        with ThreadPoolExecutor(8) as ex:
            ok = list(ex.map(is_valid_image, files))
        valid = [f for f, good in zip(files, ok) if good]
        bad = [f for f, good in zip(files, ok) if not good]
        print(f"Valid: {len(valid)} | invalid/corrupt (excluded): {len(bad)}")
        for b in bad[:10]:
            print("  invalid:", b)

    data_root = image_dir.parent
    rel = [str(f.relative_to(data_root)) for f in valid]
    splits = make_splits(rel, args.val_count, args.test_count, args.seed)
    out = PROJECT_ROOT / args.splits_dir
    out.mkdir(parents=True, exist_ok=True)
    for name, lst in splits.items():
        (out / f"{name}.txt").write_text("\n".join(lst) + "\n")
        print(f"{name:5s}: {len(lst)} images -> {out / (name + '.txt')}")

    sizes, modes = Counter(), Counter()
    for f in valid[:: max(len(valid) // 2000, 1)]:  # sample for statistics
        with Image.open(f) as im:
            sizes[im.size] += 1
            modes[im.mode] += 1
    stats = {"image_dir": str(image_dir.relative_to(PROJECT_ROOT)) if image_dir.is_relative_to(PROJECT_ROOT) else str(image_dir),
             "data_root_for_splits": str(data_root.relative_to(PROJECT_ROOT)) if data_root.is_relative_to(PROJECT_ROOT) else str(data_root),
             "candidates": len(files), "valid": len(valid), "split_seed": args.seed,
             "splits": {k: len(v) for k, v in splits.items()},
             "sampled_sizes": {f"{w}x{h}": c for (w, h), c in sizes.most_common()},
             "sampled_modes": dict(modes)}
    save_json(stats, out / "dataset_stats.json")
    print("Stats:", stats)


if __name__ == "__main__":
    main()
