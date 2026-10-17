# AI Image Inpainting

Fills masked regions of RGB images with U-Net inpainting networks (ordinary
convolutions vs. gated convolutions), compared against mean-fill and OpenCV
Telea baselines on Places365 images. Design notes: `docs/project_plan.md`.
Project status and audit: `PROJECT_AUDIT.md`.

## Website (upload → paint → inpaint → download)

```bash
cd AI-Image-Inpainting
./start.sh                 # builds the frontend if needed, serves site + API
# open http://localhost:8000   (from Windows too: WSL2 forwards localhost)
```

`./start.sh --dev` runs the API on :8000 and the Vite dev server with hot reload
on http://localhost:5173. `PORT=9000 HOST=0.0.0.0 ./start.sh` changes the
port / listens on the LAN. Stop with Ctrl+C.

In the page: upload a photo (drag & drop, paste or browse), paint the region to
remove with the brush / rectangle (eraser, undo/redo, clear and an overlay toggle
are in the toolbar), check the *Masked input preview*, pick **U-Net** or **Gated
U-Net**, click **Inpaint**, compare with the slider or side by side and click
**Download PNG**.

What the backend does (`backend/app/`):

* loads `checkpoints/unet_4gb/best.pt` and `checkpoints/gated_unet_4gb/best.pt`
  once at startup (strict state-dict loading; a missing or incompatible
  checkpoint makes that model *unavailable* with the reason - random weights
  are never used);
* applies EXIF orientation (as browsers do), flattens transparency on white;
* resizes the photo with its aspect ratio preserved so the longer side is the
  processing resolution (default **128 px, the training resolution**), padding
  to a multiple of 8. Known pixels are resampled from known pixels only, so the
  content under the mask never reaches the network;
* runs the network, upsamples the prediction to the original size and composites
  `result = I·(1−M) + P·M` in 8-bit at the original resolution, so every pixel
  outside the mask is bit-identical to the upload; returns a lossless PNG.

Mask convention everywhere: white / 1 = missing (reconstruct), black / 0 = known.

**Limitation:** the networks were trained on 128×128 images. The hole is filled
at that resolution and upscaled, so filled regions are blurry on large photos and
in large holes; upscaling does not restore fine detail. The result is a plausible
estimate, not the hidden content.

API (also at http://localhost:8000/docs):

```bash
curl localhost:8000/health
curl localhost:8000/model/info
curl -F image=@photo.jpg -F mask=@mask.png -F model=gated_unet -o result.png localhost:8000/inpaint
#    optional: -F work_size=256  (64..512, longer side the network runs at)
```

Backend settings are environment variables documented in `backend/app/config.py`
(checkpoint paths, device, default model, size limits).

### Requirements for the website

* Python packages: `requirements.txt` (adds FastAPI, uvicorn, python-multipart,
  pytest, httpx to the training stack).
* A **Linux** Node.js ≥ 20 for building the frontend (the Windows `npm` on the
  WSL PATH does not work). Used here: Node 24.21.0 unpacked to
  `~/.local/opt/node-v24.21.0-linux-x64` (`start.sh` finds it automatically):

  ```bash
  mkdir -p ~/.local/opt && cd ~/.local/opt
  curl -fsSLO https://nodejs.org/dist/v24.21.0/node-v24.21.0-linux-x64.tar.xz
  tar -xJf node-v24.21.0-linux-x64.tar.xz
  ```

### Tests

```bash
.venv/bin/python -m pytest tests -q                      # backend: preprocessing, masks, models, API
cd frontend && npm test                                   # frontend unit tests (coordinate mapping, mask encoding)
npm run build && npx playwright test                      # browser end-to-end against the real backend + models
```

The Playwright tests need Chromium (`npx playwright install chromium`). On this
WSL image Chromium also needed `libgbm1` and `libwayland-server0`; they were
unpacked locally (no system install) with
`apt-get download libgbm1 libwayland-server0` + `dpkg-deb -x <deb> ~/.local/opt/browser-libs/root`
and are passed with
`BROWSER_LIBS=~/.local/opt/browser-libs/root/usr/lib/x86_64-linux-gnu npx playwright test`.

### Website-path evaluation

`scripts/web_eval.py` evaluates the website's inference path on the held-out
test split with the same seeded masks as `scripts/evaluate.py` (PSNR, SSIM,
hole PSNR, LPIPS); results in `results/web_eval/`. LPIPS needs
`pip install lpips==0.1.4` (downloads AlexNet weights, 233 MB, on first use).

## Setup (WSL Ubuntu, Python 3.10, NVIDIA GPU)

```bash
cd AI-Image-Inpainting
python3 -m venv .venv            # skip if .venv already exists
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Smoke test (synthetic data, no downloads)

```bash
.venv/bin/python scripts/smoke_test.py --model unet       --device cuda
.venv/bin/python scripts/smoke_test.py --model gated_unet --device cuda
```

It checks data loading, masking, forward/backward, optimizer step, checkpoint
save/reload, evaluation and inference using `configs/smoke.yaml`. Use
`--device cpu` without a GPU. Its numbers are not performance results.

## Full pipeline

```bash
.venv/bin/python scripts/prepare_data.py --download            # Places365 val_256 -> data/splits/
.venv/bin/python scripts/train.py --config configs/unet.yaml
.venv/bin/python scripts/train.py --config configs/gated_unet.yaml
.venv/bin/python scripts/evaluate.py --checkpoints checkpoints/unet/best.pt checkpoints/gated_unet/best.pt
```

Training with the default config uses the VGG-16 perceptual loss, which
downloads ImageNet VGG-16 weights (~530 MB) on first use.

## 4 GB GPUs (RTX 3050 Laptop)

`configs/unet_4gb.yaml` and `configs/gated_unet_4gb.yaml` are identical to
`unet.yaml` / `gated_unet.yaml` (128 px, width 32, AMP, VGG perceptual loss,
30 epochs) except `train.batch_size: 16`. At batch 32 the U-Net needs ~3.4 GiB
and the Gated U-Net ~4.1 GiB of VRAM (more than is free under WSL); at batch 16
they need ~1.8 / ~2.3 GiB. Measured speed: ~5.8 min/epoch (U-Net) and
~7.3 min/epoch (Gated U-Net).

```bash
.venv/bin/python scripts/train.py --config configs/unet_4gb.yaml       --run-dir checkpoints/unet_4gb
.venv/bin/python scripts/train.py --config configs/gated_unet_4gb.yaml --run-dir checkpoints/gated_unet_4gb
# resume an interrupted run (continues from the last completed epoch):
.venv/bin/python scripts/train.py --config configs/gated_unet_4gb.yaml --run-dir checkpoints/gated_unet_4gb \
    --resume checkpoints/gated_unet_4gb/last.pt
# validation-split evaluation (keep results/metrics/ for the final test run):
.venv/bin/python scripts/evaluate.py --checkpoints checkpoints/unet_4gb/best.pt --split val --no-lpips \
    --out-dir results/val_eval/unet_4gb
```

`checkpoints/pilot_*` are short real-data pilot runs (a few hundred steps) used
only to check the pipeline; they are not results.

## Sharing the website on Hugging Face Spaces

**Live:** https://huggingface.co/spaces/vivikuts/ai-inpainting (direct: https://vivikuts-ai-inpainting.static.hf.space)

### Static Space, models run in the browser (free; this is what is deployed)

`scripts/prepare_hf_static.sh` exports both checkpoints to ONNX (`scripts/export_onnx.py`, which
checks each export against PyTorch), builds the frontend with `VITE_ENGINE=browser`
(`npm run build:static`) and assembles `build/hf-static`. In that build the networks run in the
visitor's browser with onnxruntime-web in a Web Worker; `frontend/src/lib/preprocess.ts` is a
port of `backend/app/imaging.py` (Pillow-exact bicubic, hole-safe downscaling, exact
compositing), tested against fixtures from the Python code. Photos never leave the device.
The first Inpaint downloads the selected model (17 or 35 MB); work size is capped at 256 px.

```bash
./scripts/prepare_hf_static.sh
.venv/bin/hf auth login                     # a "Write" token from huggingface.co/settings/tokens
.venv/bin/hf repos create <user>/ai-inpainting --type space --sdk static --public   # first time only
.venv/bin/hf upload <user>/ai-inpainting build/hf-static . --repo-type space
```

Tests: `npx playwright test -c playwright.static.config.ts` (from `frontend/`) serves
`dist-static` locally; with `STATIC_URL=https://vivikuts-ai-inpainting.static.hf.space/` it runs
against the deployed site. Both check the downloaded result against the Python server path.

### Docker Space with the FastAPI backend (needs Hugging Face PRO)

Hugging Face only hosts Docker Spaces on its free CPU hardware with a PRO subscription.
`deploy/huggingface/` (Dockerfile, README, CPU requirements) and
`scripts/prepare_hf_space.py` (assembles `build/hf-space` with weights-only checkpoints) are
kept for that case; upload as above with `--sdk docker` and `build/hf-space`.


