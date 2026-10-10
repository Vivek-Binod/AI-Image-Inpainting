---
title: AI Image Inpainting
emoji: 🖌️
colorFrom: indigo
colorTo: pink
sdk: static
app_file: index.html
pinned: false
---

# AI Image Inpainting

Upload a photo, paint over what you want removed, and a U-Net or Gated U-Net fills it in.
The networks run entirely in your browser (onnxruntime-web), so photos never leave your device;
the first Inpaint downloads the selected model (17 or 35 MB).

The models were trained on 128-px Places365 images, so fills are a plausible guess and look
soft on large photos or big holes; upscaling does not add detail.
