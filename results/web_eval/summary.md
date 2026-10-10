| method / condition | n | PSNR (dB) | SSIM | hole PSNR (dB) | LPIPS ↓ |
|---|---|---|---|---|---|
| unet/r128 | 3500 | 27.69 ± 0.21 | 0.9057 | 19.14 ± 0.15 | 0.0883 ± 0.0023 |
| unet/n256_w128 | 3500 | 26.97 ± 0.20 | 0.9019 | 18.42 ± 0.15 | 0.1142 ± 0.0024 |
| unet/n256_w128_dilate | 3500 | 25.74 ± 0.19 | 0.8912 | 17.18 ± 0.14 | 0.1220 ± 0.0025 |
| unet/n256_w256 | 3500 | 26.95 ± 0.20 | 0.9039 | 18.39 ± 0.14 | 0.1100 ± 0.0024 |
| unet/ns_web | 3467 | 26.25 ± 0.23 | 0.8806 | 18.09 ± 0.15 | 0.1314 ± 0.0031 |
| unet/ns_cli | 3467 | 26.32 ± 0.23 | 0.8817 | 18.16 ± 0.16 | 0.1288 ± 0.0031 |
| gated_unet/r128 | 3500 | 27.71 ± 0.21 | 0.9059 | 19.16 ± 0.16 | 0.0896 ± 0.0023 |
| gated_unet/n256_w128 | 3500 | 26.99 ± 0.20 | 0.9020 | 18.44 ± 0.15 | 0.1162 ± 0.0025 |
| gated_unet/n256_w128_dilate | 3500 | 25.76 ± 0.20 | 0.8913 | 17.20 ± 0.15 | 0.1244 ± 0.0026 |
| gated_unet/n256_w256 | 3500 | 27.06 ± 0.20 | 0.9050 | 18.51 ± 0.15 | 0.1104 ± 0.0025 |
| gated_unet/ns_web | 3467 | 26.28 ± 0.23 | 0.8807 | 18.11 ± 0.16 | 0.1336 ± 0.0031 |
| gated_unet/ns_cli | 3467 | 26.34 ± 0.23 | 0.8818 | 18.17 ± 0.16 | 0.1309 ± 0.0031 |
| telea/telea256 | 3500 | 25.41 ± 0.20 | 0.8951 | 16.85 ± 0.14 | 0.1295 ± 0.0026 |

Paired differences (B − A, mean ± 95% CI):

- unet: n256_w128_dilate -> n256_w128 (n=3500): hole PSNR +1.238 ± 0.042 dB, PSNR +1.238 ± 0.042 dB, SSIM +0.0106, LPIPS -0.0077 ± 0.0003
- unet: n256_w128 -> n256_w256 (n=3500): hole PSNR -0.028 ± 0.031 dB, PSNR -0.028 ± 0.031 dB, SSIM +0.0020, LPIPS -0.0043 ± 0.0002
- unet: ns_cli -> ns_web (n=3467): hole PSNR -0.069 ± 0.019 dB, PSNR -0.069 ± 0.019 dB, SSIM -0.0011, LPIPS +0.0026 ± 0.0002
- gated_unet: n256_w128_dilate -> n256_w128 (n=3500): hole PSNR +1.236 ± 0.042 dB, PSNR +1.236 ± 0.042 dB, SSIM +0.0107, LPIPS -0.0081 ± 0.0003
- gated_unet: n256_w128 -> n256_w256 (n=3500): hole PSNR +0.070 ± 0.028 dB, PSNR +0.070 ± 0.028 dB, SSIM +0.0030, LPIPS -0.0058 ± 0.0002
- gated_unet: ns_cli -> ns_web (n=3467): hole PSNR -0.057 ± 0.020 dB, PSNR -0.057 ± 0.020 dB, SSIM -0.0011, LPIPS +0.0027 ± 0.0002
- unet -> gated_unet (n256_w128) (n=3500): hole PSNR +0.019 ± 0.013 dB, PSNR +0.019 ± 0.013 dB, SSIM +0.0002, LPIPS +0.0020 ± 0.0001

Reproduction of published test hole PSNR (r128 vs results/metrics):

- unet: mean |diff| 0.0018 dB, max 0.1295 dB (n=3500)
- gated_unet: mean |diff| 0.0020 dB, max 0.1665 dB (n=3500)
