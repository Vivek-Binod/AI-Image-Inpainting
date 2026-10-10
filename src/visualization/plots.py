"""Figures generated from real data and logged results."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def comparison_figure(original: np.ndarray, mask: np.ndarray, masked: np.ndarray, result: np.ndarray,
                      path: str | Path, title: str | None = None) -> None:
    """Original | mask | corrupted | inpainted side by side."""
    shown = masked + mask[..., None] * 0.8
    panels = [(original, "Original"), (mask, "Mask (white = missing)"), (np.clip(shown, 0, 1), "Corrupted input"),
              (result, "Inpainted")]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))
    for ax, (img, label) in zip(axes, panels):
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None, vmin=0, vmax=1)
        ax.set_title(label, fontsize=10)
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def training_curves(histories: dict[str, list[dict]], path: str | Path) -> None:
    """Training/validation total loss and validation hole PSNR per epoch for each model."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    for (name, hist), c in zip(histories.items(), colors):
        ep = [h["epoch"] for h in hist]
        axes[0].plot(ep, [h["train_total"] for h in hist], color=c, ls="--", label=f"{name} train")
        axes[0].plot(ep, [h["val_total"] for h in hist], color=c, label=f"{name} val")
        axes[1].plot(ep, [h["val_hole_psnr"] for h in hist], color=c, label=name)
    axes[0].set(xlabel="Epoch", ylabel="Total loss", title="Training and validation loss")
    axes[1].set(xlabel="Epoch", ylabel="PSNR (dB)", title="Validation PSNR inside the hole (raw output)")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
