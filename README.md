# AI Image Inpainting

Fills masked regions of RGB images with U-Net inpainting networks (ordinary
convolutions vs. gated convolutions), compared against mean-fill and OpenCV
Telea baselines on Places365 images. Design notes: `docs/project_plan.md`.

## Setup (WSL Ubuntu, Python 3.10, NVIDIA GPU)

```bash
cd /root/AI-Image-Inpainting
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
