---
title: AI Image Inpainting
emoji: 🖌️
colorFrom: indigo
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
---

# AI Image Inpainting

Upload a photo, paint over what you want removed, and a U-Net or Gated U-Net fills it in.

The models were trained on 128-px Places365 images, so fills are a plausible guess and look
soft on large photos or big holes; upscaling does not add detail. Uploaded images are
processed in memory and not stored. This Space runs on CPU.
