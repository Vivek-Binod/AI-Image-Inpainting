# Project Plan and Requirements Summary

*Recorded at the start of the project (2026-10-09), before implementation.*

## Assignment materials

No assignment brief, reference link, or report template was supplied. The
placeholders in the project brief were left unfilled, and nothing relevant was
found in the working directory. The design below therefore rests on the
following **assumptions**:

- Any deep learning framework is acceptable, so PyTorch is used.
- No dataset is mandated. Any legally usable public image dataset is acceptable.
- No report format is mandated, so a conventional undergraduate report
  structure in Markdown is used, with IEEE-style references.
- Student, institution, course, and instructor details are unknown and are left
  as placeholders.

## Technical objective

Given an RGB image `I` and a binary mask `M` (1 = missing, 0 = known), predict
`I_pred` from the corrupted image `I ⊙ (1 − M)` and `M`. Then form the composite
`I_final = M ⊙ I_pred + (1 − M) ⊙ I`, which keeps every known pixel unchanged.

## Environment (inspected)

| Item | Value |
|---|---|
| OS | Ubuntu 22.04.3 LTS on WSL2 (Linux 5.15) |
| Python | 3.10.12 (system); project venv in `.venv/` |
| CPU / RAM | 12 logical cores / 11 GiB |
| GPU | NVIDIA GeForce RTX 3050 Laptop, 4 GiB VRAM |
| Storage | ~926 GB free |
| Git | 2.34.1; `/root` is not a repository |
| Framework | PyTorch was not installed. It is installed into the project venv (CUDA 12.4 wheels). |

## Design decisions

- **Dataset:** Places365-Standard validation images at 256×256 (36,500 scene
  photos), downloaded from the official MIT CSAIL server. They are resized to
  128×128 to fit a 4 GiB GPU. The images are split with a fixed seed into
  train, validation, and test sets.
- **Masks:** random rectangles, free-form brush strokes, and a mixture of both.
  The mask generator is seeded, so test masks are identical for every method.
- **Models:**
  1. A U-Net encoder-decoder with ordinary convolutions (baseline network).
  2. A U-Net built from gated convolutions (Yu et al., 2019), with a dilated
     gated-convolution bottleneck that enlarges the receptive field.
  Both models receive the same 4-channel input: the masked RGB image plus the
  mask.
- **Loss:** a weighted L1 loss on missing pixels, an L1 loss on known pixels,
  and a VGG-16 perceptual loss. Both models use the same objective.
- **Non-learned baselines:** mean-colour fill and OpenCV Telea inpainting.
- **Metrics:** MSE, PSNR, and SSIM over the full composite image; MSE and PSNR
  over the missing region only; LPIPS if available. Results are broken down by
  mask type and hole size.

## Evaluation protocol

The test split stays untouched until final evaluation. The best checkpoint is
chosen by validation loss. Every method is evaluated on the same test images
with the same seeded masks.

## Risks

- Limited VRAM. Mitigation: 128 px images, moderate channel widths, and mixed
  precision.
- Short training budget. Results will not be state of the art, and this is
  disclosed.
- Places365 images are very diverse, so large holes will produce blurry fills.

## Compute estimate

Each model needs roughly 1–2 minutes per epoch over about 30k images on the
RTX 3050, so a few tens of epochs per model fit within about an hour.
