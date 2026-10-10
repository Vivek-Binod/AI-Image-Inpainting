"""Create e2e fixture images from a real held-out test photo (Places365 test split).

    .venv/bin/python frontend/e2e/fixtures.py <out_dir>

photo.png    1024 x 768 PNG (non-square; displayed downscaled in the browser)
rotated.jpg  JPEG stored 768 x 600 with EXIF orientation 6, i.e. displayed 600 x 768
"""
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
split = ROOT / "data/splits/test.txt"
src = ROOT / "data/raw" / split.read_text().split()[1]
img = Image.open(src).convert("RGB")                        # 256 x 256 real photo
img.crop((0, 32, 256, 224)).resize((1024, 768), Image.BICUBIC).save(out / "photo.png")
stored = img.resize((768, 600), Image.BICUBIC)
exif = Image.Exif()
exif[0x0112] = 6
stored.save(out / "rotated.jpg", quality=95, exif=exif)
print(src.name)
