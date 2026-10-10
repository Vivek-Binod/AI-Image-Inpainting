"""Training objective.

    L = w_hole * L_hole + w_valid * L_valid + w_perc * L_perc

L_hole  = sum(M * |I - I_pred|) / (3 * sum(M) + eps)               mean L1 over missing pixels
L_valid = sum((1-M) * |I - I_pred|) / (3 * sum(1-M) + eps)         mean L1 over known pixels
L_perc  = sum_l mean|phi_l(I_pred) - phi_l(I)| + mean|phi_l(I_comp) - phi_l(I)|

where M is broadcast over the 3 RGB channels (hence the factor 3 in the
denominator: numerator and denominator both count channel-pixels), phi_l are
VGG-16 activations (relu1_2, relu2_2, relu3_3), and I_comp is the composite.
The hole/valid split and the 6 : 1 weighting follow Liu et al. (2018); the
perceptual term follows Johnson et al. (2016) and Liu et al. (2018). The style
and total-variation terms of Liu et al. are deliberately omitted to keep the
objective simple.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8


def masked_l1(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Mean absolute error over pixels where ``weight`` (N,1,H,W) is 1, averaged over channels."""
    weight = weight.expand_as(pred)
    return (weight * (pred - target).abs()).sum() / (weight.sum() + EPS)


class VGGFeatures(nn.Module):
    """Frozen VGG-16 feature extractor returning relu1_2, relu2_2, relu3_3 activations."""

    LAYER_ENDS = (4, 9, 16)  # indices (exclusive) into torchvision vgg16().features

    def __init__(self, pretrained: bool = True):
        super().__init__()
        from torchvision.models import VGG16_Weights, vgg16

        weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        features = vgg16(weights=weights).features[: self.LAYER_ENDS[-1]].eval()
        for p in features.parameters():
            p.requires_grad_(False)
        self.features = features
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def train(self, mode: bool = True):  # keep frozen network in eval mode
        return super().train(False)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = (x - self.mean) / self.std
        outs, start = [], 0
        for end in self.LAYER_ENDS:
            x = self.features[start:end](x)
            outs.append(x)
            start = end
        return outs


class InpaintingLoss(nn.Module):
    def __init__(self, w_hole: float = 6.0, w_valid: float = 1.0, w_perceptual: float = 0.05,
                 pretrained_vgg: bool = True):
        super().__init__()
        self.w_hole, self.w_valid, self.w_perceptual = w_hole, w_valid, w_perceptual
        self.vgg = VGGFeatures(pretrained_vgg) if w_perceptual > 0 else None

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        pred = pred.float()
        hole = masked_l1(pred, target, mask)
        valid = masked_l1(pred, target, 1.0 - mask)
        total = self.w_hole * hole + self.w_valid * valid
        out = {"hole_l1": hole.detach(), "valid_l1": valid.detach()}

        if self.vgg is not None:
            comp = mask * pred + (1.0 - mask) * target
            f_gt = [f.detach() for f in self.vgg(target)]
            f_pred, f_comp = self.vgg(pred), self.vgg(comp)
            perc = sum(F.l1_loss(a, g) + F.l1_loss(b, g) for a, b, g in zip(f_pred, f_comp, f_gt))
            total = total + self.w_perceptual * perc
            out["perceptual"] = perc.detach()

        out["total"] = total
        return out
