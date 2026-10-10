"""U-Net style encoder-decoder inpainting networks.

Two variants share exactly the same topology and differ only in the
convolution block:

* ``conv``  - ordinary convolution followed by ELU.
* ``gated`` - gated convolution (Yu et al., "Free-Form Image Inpainting with
  Gated Convolution", ICCV 2019): two parallel convolutions produce features
  ``F`` and gates ``G`` and the output is ``ELU(F) * sigmoid(G)``. The learned
  soft gate lets each spatial location and channel decide how much to trust
  its input, which is useful because hole pixels carry no information.

Input:  4 channels = corrupted RGB image (holes set to 0) + binary mask (1 = hole).
Output: 3-channel RGB prediction in [0, 1] (sigmoid), same spatial size as input.

Topology (base width c, input H x W, H and W divisible by 8):

    enc0  : 2 blocks, c,   H     (skip)
    enc1  : stride-2 block + block, 2c, H/2   (skip)
    enc2  : stride-2 block + block, 4c, H/4   (skip)
    enc3  : stride-2 block + block, 8c, H/8
    bottleneck: dilated blocks (dilation 2, 4, 8) + block, 8c, H/8
    dec2  : upsample x2, concat enc2, 2 blocks, 4c
    dec1  : upsample x2, concat enc1, 2 blocks, 2c
    dec0  : upsample x2, concat enc0, 2 blocks, c
    head  : 1x1 conv -> 3 channels -> sigmoid

The dilated bottleneck enlarges the receptive field so that pixels deep inside
a large hole can see context from the surrounding image.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv2d + ELU."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dilation: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=dilation, dilation=dilation)
        self.act = nn.ELU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class GatedConvBlock(nn.Module):
    """Gated convolution: ELU(conv_f(x)) * sigmoid(conv_g(x))."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dilation: int = 1):
        super().__init__()
        # A single conv produces both feature and gate maps (2 * out_ch channels);
        # this is mathematically identical to two separate convolutions.
        self.conv = nn.Conv2d(in_ch, 2 * out_ch, 3, stride=stride, padding=dilation, dilation=dilation)
        # Not inplace: ``feat`` is a view returned by chunk(), which autograd forbids modifying in place.
        self.act = nn.ELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat, gate = self.conv(x).chunk(2, dim=1)
        return self.act(feat) * torch.sigmoid(gate)


BLOCKS = {"conv": ConvBlock, "gated": GatedConvBlock}


class InpaintUNet(nn.Module):
    def __init__(self, block: str = "conv", base_channels: int = 32, in_channels: int = 4,
                 out_channels: int = 3):
        super().__init__()
        if block not in BLOCKS:
            raise ValueError(f"Unknown block type '{block}'. Expected one of {list(BLOCKS)}.")
        B = BLOCKS[block]
        c = base_channels
        self.block_type = block

        self.enc0 = nn.Sequential(B(in_channels, c), B(c, c))
        self.enc1 = nn.Sequential(B(c, 2 * c, stride=2), B(2 * c, 2 * c))
        self.enc2 = nn.Sequential(B(2 * c, 4 * c, stride=2), B(4 * c, 4 * c))
        self.enc3 = nn.Sequential(B(4 * c, 8 * c, stride=2), B(8 * c, 8 * c))
        self.bottleneck = nn.Sequential(
            B(8 * c, 8 * c, dilation=2),
            B(8 * c, 8 * c, dilation=4),
            B(8 * c, 8 * c, dilation=8),
            B(8 * c, 8 * c),
        )
        self.dec2 = nn.Sequential(B(8 * c + 4 * c, 4 * c), B(4 * c, 4 * c))
        self.dec1 = nn.Sequential(B(4 * c + 2 * c, 2 * c), B(2 * c, 2 * c))
        self.dec0 = nn.Sequential(B(2 * c + c, c), B(c, c))
        self.head = nn.Conv2d(c, out_channels, 1)

    @staticmethod
    def _up(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Nearest-neighbour upsampling followed by convolution (in the decoder
        # block) avoids the checkerboard artefacts of transposed convolutions.
        x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        return torch.cat([x, skip], dim=1)

    def forward(self, masked: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if masked.dim() != 4 or masked.shape[1] != 3:
            raise ValueError(f"Expected masked image of shape (N, 3, H, W), got {tuple(masked.shape)}")
        if mask.shape != (masked.shape[0], 1, *masked.shape[2:]):
            raise ValueError(f"Expected mask of shape (N, 1, H, W) matching image, got {tuple(mask.shape)}")
        if masked.shape[2] % 8 or masked.shape[3] % 8:
            raise ValueError(f"Height and width must be divisible by 8, got {tuple(masked.shape[2:])}")

        x = torch.cat([masked, mask], dim=1)
        e0 = self.enc0(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b = self.bottleneck(e3)
        d2 = self.dec2(self._up(b, e2))
        d1 = self.dec1(self._up(d2, e1))
        d0 = self.dec0(self._up(d1, e0))
        return torch.sigmoid(self.head(d0))


def composite(pred: torch.Tensor, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """I_final = M * I_pred + (1 - M) * I  (keeps known pixels exactly)."""
    return mask * pred + (1.0 - mask) * image


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
