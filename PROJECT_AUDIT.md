# Project audit — AI Image Inpainting

Audit date: 2026-10-10. Everything below was checked by running it in this
environment (WSL2 Ubuntu 22.04, Python 3.10 `.venv`, PyTorch 2.6.0+cu124,
NVIDIA RTX 3050 Laptop GPU 4 GB). Nothing was retrained for the website.

## 1. What existed before the website work

| Area | Location | Status (verified) |
|---|---|---|
| U-Net and gated-convolution U-Net | `src/models/unet.py`, `src/models/__init__.py` (`build_model`) | Works. 4,307,555 / 8,615,011 parameters, base width 32, ELU, dilated bottleneck, sigmoid output. |
| Data pipeline | `src/datasets/inpainting_dataset.py`, `scripts/prepare_data.py` | Places365 `val_256` (36,500 images) in `data/raw/val_256/`; splits in `data/splits/{train,val,test}.txt` = 30,000 / 3,000 / 3,500 (seed 42). Images centre-cropped + bicubic to 128 px. |
| Masks | `src/masks/generator.py` | Rectangles, free-form strokes, mixed; 1 = missing. Test masks are seeded (42 + 2,000,000 + i). |
| Loss / training | `src/losses/`, `src/training/trainer.py`, `scripts/train.py` | 6·L1 hole + 1·L1 valid + 0.05·VGG-16 perceptual; Adam, cosine LR, AMP. |
| Evaluation + baselines | `src/evaluation/`, `scripts/evaluate.py` | PSNR, SSIM, hole PSNR, optional LPIPS; mean fill and OpenCV Telea. |
| CLI inference | `src/inference/inpainter.py`, `scripts/inpaint.py` | Works on any image size (see finding F2 below). |
| Smoke tests | `scripts/smoke_test.py` | **Passed** for `unet` and `gated_unet` on CUDA (re-run during this audit). |
| `tests/` | — | Was empty. |
| Frontend / backend / website | — | Did not exist. |

### Checkpoints (reused, not retrained)

| Model | Checkpoint | Epochs | Training resolution | Best val. loss | Saved |
|---|---|---|---|---|---|
| U-Net | `checkpoints/unet_4gb/best.pt` | 30 / 30 (completed) | 128 | 0.59744 | 2026-10-09 19:08 |
| Gated U-Net | `checkpoints/gated_unet_4gb/best.pt` | 30 / 30 (completed) | 128 | 0.59692 | 2026-10-10 07:42 |

Both load with strict state-dict matching into the current architecture and
produce finite outputs. `checkpoints/pilot_*` are short pipeline-check runs and
are not used. Configs: `configs/unet_4gb.yaml`, `configs/gated_unet_4gb.yaml`
(batch 16; otherwise `configs/default.yaml`).

### Published test results (verified from `results/metrics/summary.json`, 3,500 test images, 128 px)

| Method | PSNR (dB) | SSIM | Hole PSNR (dB) |
|---|---|---|---|
| Gated U-Net | 27.72 | 0.9059 | 19.16 |
| U-Net | 27.69 | 0.9057 | 19.14 |
| Telea | 25.98 | 0.8913 | 17.43 |
| Mean fill | 21.68 | 0.8399 | 13.13 |

One training run (seed 42) per model; the U-Net vs gated difference is small.
The report is `results/final_submission/AI_Image_Inpainting_Mini_Project_Report.{pdf,docx,md}`.

## 2. Findings

* **F1 — No website existed.** Built (section 3).
* **F2 — CLI inference distorts non-square photos.** `src/inference/inpainter.py`
  resizes any image to a 128×128 square (aspect ratio not preserved) and back.
  Training used square crops, so non-square photos were squashed. The website
  path preserves the aspect ratio instead. The CLI was left unchanged (its
  outputs are cited in the report); the measured effect is in section 5.
* **F3 — Hole content can leak into the network input when downscaling.**
  In the CLI path, the bicubic downscale mixes hole pixels into neighbouring
  "known" low-resolution pixels (and the box-filtered mask is not strictly
  conservative). For object removal this means colours of the removed object can
  bleed into the fill. Found by an automated test; the website path resamples
  known pixels from known pixels only (normalised bicubic) and never lets hole
  content reach the network — tested for both models
  (`tests/test_engine.py::test_hole_content_never_reaches_the_network`).
* **F4 — Known pixels.** The training/eval composite is done in float; the
  website composites in uint8 at the original resolution, so pixels outside the
  mask are bit-identical in the output PNG (tested, including after PNG encoding
  and through the browser download).
* **F5 — Environment.** The `npm` on the WSL PATH is the Windows one
  (`/mnt/c/Program Files/nodejs`), unusable here. A Linux Node 24.21.0 was
  unpacked to `~/.local/opt/` (checksum verified); no system packages installed.
* **F6 — Stray file.** `tatus --short` in the project root contains the help
  screen of `less` (it looks like the output of a mistyped `git status --short`).
  It was not created by this work and was left untouched; it is safe to delete.

## 3. What was added

| Path | Purpose |
|---|---|
| `backend/app/main.py` | FastAPI app: `GET /health`, `GET /model/info`, `POST /inpaint` (multipart image + mask → PNG); serves the built site at `/`. |
| `backend/app/engine.py` | Loads both checkpoints once with the project's `load_model_from_checkpoint`; reports missing/invalid checkpoints as unavailable; runs inference under a lock. |
| `backend/app/imaging.py` | Upload decoding (EXIF orientation, alpha, 16-bit, palette, size limits), mask decoding/alignment, aspect-preserving model input, exact uint8 composite, PNG encoding. |
| `backend/app/config.py` | Environment-variable settings (checkpoint paths, device, processing resolution, limits). |
| `frontend/` | React 19 + TypeScript + Vite 8 + Tailwind 4 single-page app (upload, mask canvas with brush/eraser/rectangle, brush size, undo/redo, clear, overlay toggle, masked-input preview, model selector, processing resolution, Inpaint, loading/success/error states, before/after slider and side-by-side, PNG download, reset). |
| `frontend/src/lib/geometry.ts` | Pointer → displayed → mask → image → model coordinate mapping. |
| `frontend/src/lib/mask.ts` | Mask operations, binarisation (overlay = exactly the mask sent), PNG export (white = missing). |
| `tests/` | 60 backend tests (pytest). |
| `frontend/src/lib/*.test.ts` | 16 frontend unit tests (Vitest). |
| `frontend/e2e/` | 5 Playwright browser tests against the real backend and models. |
| `scripts/web_eval.py` | Controlled evaluation of the website inference path (section 5). |
| `start.sh` | Builds the frontend if needed and starts the site on http://localhost:8000. |
| `requirements.txt`, `README.md` | Added web/test dependencies and run instructions. |

No existing source file, config, checkpoint, dataset file or metric file was
modified or deleted.

## 4. Verification performed

| Test | Result |
|---|---|
| `scripts/smoke_test.py --model unet / gated_unet --device cuda` | passed |
| `pytest tests` (60 tests: decoding, EXIF, alpha, 16-bit, mask polarity & thresholds, mask alignment/upsampling, aspect-preserving model size, padding, no hole leakage, exact composite, empty & small masks, PNG round-trip, checkpoint loading, trained-weights identity, missing/corrupt/wrong checkpoints, output sizes for 4 shapes × 2 models, CPU inference, beats mean fill on a real test photo, API endpoints, validation errors, size limit, 503 when checkpoints missing) | **60 passed** |
| `npm test` (Vitest: coordinate mapping, canvas sizing, model size parity with the backend, mask binarisation and polarity, rectangle snapping) | **16 passed** |
| `npx playwright test` (Chromium): full journey with both models — upload 1024×768 photo, rectangle + brush + eraser, undo/redo/clear, overlay toggle, preview, Inpaint, compare, download; the downloaded PNG is checked against the original and the exact mask the browser sent (known pixels bit-exact, drawn regions inside the mask, erased/untouched regions outside); EXIF-rotated JPEG alignment; unreadable upload; server error display (the only mocked response); mobile (Pixel 7 viewport, real touch strokes, no horizontal scroll) | **5 passed** |
| Live server (`./start.sh`, http://localhost:8000): `curl` POST /inpaint and a headless-browser upload→inpaint against the running instance | 200, real model output (logged in `results/logs/website.log`) |

## 5. Phase 7 — baseline and controlled experiments

Run: `.venv/bin/python scripts/web_eval.py` on all 3,500 held-out test images with
the same seeded masks as the published evaluation (log `results/logs/web_eval.log`,
outputs `results/web_eval/{per_image.csv,summary.json,summary.md}`). Means ± 95 % CI.
LPIPS: `lpips` 0.1.4, AlexNet, lower is better. Nothing here was used for training
or model selection.

**1. Baseline reproduced.** Running the website code on the report's protocol
(128-px images and masks, `r128`) reproduces the published per-image hole PSNR to
a mean absolute difference of 0.002 dB (max 0.17 dB, from 8-bit rounding of the
prediction): U-Net 19.14 dB, Gated U-Net 19.16 dB.

**2. LPIPS (first time it was measured).** Test-set LPIPS at 128 px: U-Net
**0.0883**, Gated U-Net **0.0896**. Hole PSNR slightly favours the gated model
and LPIPS slightly favours the U-Net, so neither model is clearly better.

**3. Hole size and mask shape** (Gated U-Net, 128 px; U-Net within 0.05 dB):

| Hole size | n | Hole PSNR (dB) | LPIPS |
|---|---|---|---|
| 0–10 % | 896 | 21.17 | 0.026 |
| 10–20 % | 1395 | 18.95 | 0.076 |
| 20–30 % | 957 | 18.06 | 0.136 |
| 30–40 % | 231 | 17.36 | 0.212 |
| 40–100 % | 21 | 16.93 | 0.264 |

By mask family: free-form 20.63 dB / 0.048, mixed 18.69 dB / 0.113, rectangles
17.61 dB / 0.127. LPIPS grows about tenfold from the smallest to the largest holes,
consistent with the blur seen in large holes.

**4. Training vs inference preprocessing** (native 256-px test images, masks
upsampled ×2, results measured at 256 px):

| Condition | U-Net hole PSNR / LPIPS | Gated hole PSNR / LPIPS |
|---|---|---|
| network at 128, normalised resampling (**website default**) | 18.42 / 0.1142 | 18.44 / 0.1162 |
| network at 128, dilate-the-hole resampling | 17.18 / 0.1220 | 17.20 / 0.1244 |
| network at 256 (untested resolution) | 18.39 / 0.1100 | 18.51 / 0.1104 |
| OpenCV Telea at 256 | 16.85 / 0.1295 | — |

* Normalised resampling beats the simpler leak-proof alternative by
  +1.24 ± 0.04 dB hole PSNR and −0.008 LPIPS for both models (paired). It is
  therefore the website default.
* Running the network at 256 instead of 128 changes hole PSNR by −0.03 ± 0.03 dB
  (U-Net) / +0.07 ± 0.03 dB (Gated) and lowers LPIPS by 0.004–0.006 (paired). The
  effect is small. The default stays at the training resolution (128); 256 is
  selectable in the UI. This was measured only on 256-px images.
* The models stay ahead of Telea at 256 px on all four metrics.

**5. Aspect ratio (finding F2)** (non-square 256×176 crops, n = 3,467 with a
non-empty hole). The original CLI path squashes the crop to 128×128; the website
path keeps the aspect ratio (128×88). Website − CLI: −0.069 ± 0.019 dB (U-Net)
and −0.057 ± 0.020 dB (Gated), with LPIPS +0.003. So the website is very
slightly *worse* on these metrics. A likely but untested explanation: the CLI
downscale lets a little true hole content into the network input (finding F3),
which helps on benchmarks but would leak the removed object in real use. Keeping
the aspect ratio did not give a measurable quality gain; the website keeps it
because it does not leak hole content and does not distort the image.

**Not done**, in the order listed by the plan:

* object-shaped masks: no object-mask source is available in this dataset;
* contextual attention or another long-range module, higher-resolution
  training, and style/adversarial losses: each needs new multi-hour training runs.

No improvement over the published models is claimed. The only change with a
measured gain is the inference preprocessing in item 4, and that gain is
relative to another option written during this work, not to the report.

## 6. Remaining limitations and next steps

* The models were trained at 128×128 on scene images. Fills are blurry, especially
  in large holes and on large photos; upscaling the 128-px prediction does not
  add detail. The output is a plausible estimate, not the hidden content.
* One training run per model; the U-Net / Gated U-Net difference is within noise.
* Not done (each needs new training runs of several hours on this 4 GB GPU, and
  should be run one at a time against the preserved baseline):
  object-shaped masks (needs a mask source such as segmentation masks),
  contextual attention or another long-range module, higher-resolution
  (256 px) training, and style/adversarial losses. None of these is claimed.
* No public deployment: Docker is not available here and nothing was hosted.
  The site runs locally (section 7). For a server, run `./start.sh` with
  `HOST=0.0.0.0` behind a reverse proxy with HTTPS.

## 7. How to run

```bash
cd /root/AI-Image-Inpainting
./start.sh                       # http://localhost:8000
.venv/bin/python -m pytest tests -q
cd frontend && npm test && npm run build && \
  BROWSER_LIBS=~/.local/opt/browser-libs/root/usr/lib/x86_64-linux-gnu npx playwright test
```
