"""Generate the report figures from the project's saved outputs (read-only).

Run with the project venv:  .venv/bin/python results/final_submission/build/make_figures.py
Writes PNGs (300 dpi) to results/final_submission/figures/.

Sources: checkpoints/*/history.json, results/metrics (test), results/val_eval (validation),
results/reconstructed (test examples), results/inference (validation examples), and the
project's own mask generator (src/masks/generator.py) for the mask illustrations.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.datasets.inpainting_dataset import load_image  # noqa: E402
from src.masks.generator import MaskConfig, free_form_mask, generate_mask, rect_mask  # noqa: E402

FIG = ROOT / "results/final_submission/figures"
FIG.mkdir(parents=True, exist_ok=True)

# Fixed method -> colour mapping (dataviz default categorical order), used in every figure.
COL = {"gated_unet": "#2a78d6", "unet": "#eb6834", "telea": "#1baf7a", "mean_fill": "#eda100"}
LAB = {"gated_unet": "Gated U-Net", "unet": "U-Net", "telea": "OpenCV Telea", "mean_fill": "Mean fill"}
ORDER = ["gated_unet", "unet", "telea", "mean_fill"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e4e3df"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 300,
})


def save(fig, name):
    fig.savefig(FIG / name, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)
    print("wrote", FIG / name)


def img(path):
    return np.asarray(Image.open(path).convert("RGB"))


# ----------------------------------------------------------------------------------------------
# Figure 1: pipeline
# ----------------------------------------------------------------------------------------------
def box(ax, x, y, w, h, text, fc="#f3f6fb", ec="#2a78d6", fs=8, pad=0.03, gap=0.06):
    """Rounded box: bold first line anchored at the top, remaining lines below it."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
                                fc=fc, ec=ec, lw=1.0))
    lines = text.split("\n")
    ax.text(x + w / 2, y + h - pad, lines[0], ha="center", va="top", fontsize=fs + 0.5, weight="bold", color=INK)
    ax.text(x + w / 2, y + h - pad - gap, "\n".join(lines[1:]), ha="center", va="top", fontsize=fs,
            color=INK2, linespacing=1.3)


def arrow(ax, p, q, color=MUTED, style="-|>", ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=10, color=color, lw=1.1,
                                 linestyle=ls, connectionstyle=f"arc3,rad={rad}"))


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    W, H = 0.215, 0.2
    xs = [0.005, 0.26, 0.515, 0.77]
    top, bot = 0.73, 0.40
    box(ax, xs[0], top, W, H, "1. Dataset\nPlaces365 val_256\n36,500 RGB images\n256 × 256 px")
    box(ax, xs[1], top, W, H, "2. Split (seed 42)\ntrain 30,000\nvalidation 3,000\ntest 3,500")
    box(ax, xs[2], top, W, H, "3. Preprocessing\ncentre crop + resize\nto 128 × 128, RGB\nvalues in [0, 1]")
    box(ax, xs[3], top, W, H, "4. Masks\nrect / free-form / both\n1 missing / 0 known\nseeded for val/test")
    box(ax, xs[3], bot, W, H, "5. Network input\nI ⊙ (1 − M)  (3 ch)\n+ mask M  (1 ch)\n= 4 × 128 × 128")
    box(ax, xs[2], bot, W, H, "6. Network\nU-Net (conv) or\nGated U-Net\noutput Î in [0, 1]", fc="#eef4fc")
    box(ax, xs[1], bot, W, H, "7. Composite\nI_final = M ⊙ Î\n+ (1 − M) ⊙ I\nknown pixels kept")
    box(ax, xs[0], bot, W, H, "8. Evaluation\nPSNR, SSIM (full)\nhole PSNR; compared\nwith baselines")
    box(ax, xs[1], 0.02, xs[3] + W - xs[1], 0.25,
        "Training (train split only)\nloss = 6 L1(hole) + 1 L1(known) + 0.05 VGG-16 perceptual\n"
        "Adam, lr 0.0002, cosine schedule, batch 16, fp16 mixed precision\n"
        "30 epochs; checkpoint kept by lowest validation loss",
        fc="#fff7ec", ec="#eb6834")
    for i in range(3):
        arrow(ax, (xs[i] + W, top + H / 2), (xs[i + 1], top + H / 2))
    arrow(ax, (xs[3] + W / 2, top), (xs[3] + W / 2, bot + H))
    for i in (3, 2, 1):
        arrow(ax, (xs[i], bot + H / 2), (xs[i - 1] + W, bot + H / 2))
    arrow(ax, (xs[2] + W / 2, 0.27), (xs[2] + W / 2, bot), color="#eb6834", style="<|-|>", ls="--")
    ax.text(xs[0] + W / 2, 0.30, "Baselines (no learning):\nmean-colour fill,\nOpenCV Telea", ha="center",
            va="top", fontsize=8, color=INK2)
    arrow(ax, (xs[0] + W / 2, 0.315), (xs[0] + W / 2, bot), color="#1baf7a")
    save(fig, "fig02_pipeline.png")


# ----------------------------------------------------------------------------------------------
# Figure 2: architecture (from src/models/unet.py, base width c = 32)
# ----------------------------------------------------------------------------------------------
def fig_architecture():
    fig = plt.figure(figsize=(7.2, 4.4))
    ax = fig.add_axes([0, 0.24, 1, 0.76])
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1)
    ax.axis("off")
    enc = [("enc0", "2 blocks", 32, 128), ("enc1", "s2 + block", 64, 64), ("enc2", "s2 + block", 128, 32),
           ("enc3", "s2 + block", 256, 16)]
    dec = [("dec0", "2 blocks", 32, 128), ("dec1", "2 blocks", 64, 64), ("dec2", "2 blocks", 128, 32)]
    w = 0.13
    ys = [0.72, 0.53, 0.34, 0.15]
    ex, dx = 0.04, 0.70
    for (n, b, c, s), y in zip(enc, ys):
        box(ax, ex, y, w, 0.14, f"{n}\n{b}\n{c} ch, {s}×{s}", fs=7.5, pad=0.02, gap=0.04)
    for (n, b, c, s), y in zip(dec, ys[:3]):
        box(ax, dx, y, w, 0.14, f"{n}\nup×2, concat\n{b}, {c} ch", fs=7.5, pad=0.02, gap=0.04)
    box(ax, 0.36, 0.015, 0.24, 0.12, "Bottleneck\n256 ch, 16×16; dilation 2, 4, 8, 1", fs=7.5, pad=0.02, gap=0.04)
    box(ax, 0.87, 0.72, 0.12, 0.14, "head\n1×1 conv\n3 ch, sigmoid", fs=7.5, fc="#eef4fc", pad=0.02, gap=0.04)
    ax.text(ex + w / 2, 0.99, "input: 4 × 128 × 128\n(masked RGB + mask)", ha="center", va="top", fontsize=7.5,
            color=INK2)
    for i in range(3):
        arrow(ax, (ex + w / 2, ys[i]), (ex + w / 2, ys[i + 1] + 0.14))
    arrow(ax, (ex + w / 2, ys[3]), (0.36, 0.075), rad=0.15)
    arrow(ax, (0.60, 0.075), (dx + w / 2, ys[2]), rad=0.15)
    for i in (2, 1):
        arrow(ax, (dx + w / 2, ys[i] + 0.14), (dx + w / 2, ys[i - 1]))
    arrow(ax, (dx + w, 0.79), (0.87, 0.79))
    for y in ys[:3]:
        arrow(ax, (ex + w, y + 0.07), (dx, y + 0.07), color="#2a78d6", ls="--")
    ax.text(0.42, ys[0] + 0.085, "skip connections (concatenation)", ha="center", fontsize=7.5, color="#2a78d6")
    ax.text(0.93, 0.68, "output Î:\n3 × 128 × 128", ha="center", va="top", fontsize=7.5, color=INK2)

    bx = fig.add_axes([0, 0, 1, 0.21])
    bx.set_xlim(0, 1)
    bx.set_ylim(0, 1)
    bx.axis("off")
    box(bx, 0.02, 0.06, 0.44, 0.88, "Conv block (U-Net)\nx → Conv 3×3 (C_out) → ELU\nwhole network: 4,307,555 parameters",
        fs=8, fc="#fff3ec", ec="#eb6834", pad=0.1, gap=0.25)
    box(bx, 0.52, 0.08, 0.46, 0.84,
        "Gated conv block (Gated U-Net)\nx → Conv 3×3 (2·C_out) → split F, G;  y = ELU(F) ⊙ σ(G)\n"
        "whole network: 8,615,011 parameters", fs=8, fc="#eef4fc", ec="#2a78d6", pad=0.1, gap=0.25)
    save(fig, "fig03_architecture.png")


# ----------------------------------------------------------------------------------------------
# Figure 3: masks (generator examples on a validation image + test hole-ratio distribution)
# ----------------------------------------------------------------------------------------------
def fig_masks():
    val0 = (ROOT / "data/splits/val.txt").read_text().split()[0]
    image = load_image(ROOT / "data/raw" / val0, 128)
    cfg = MaskConfig()
    examples = [("rect", rect_mask(128, 128, np.random.default_rng(3), cfg)),
                ("rect", rect_mask(128, 128, np.random.default_rng(11), cfg)),
                ("free_form", free_form_mask(128, 128, np.random.default_rng(5), cfg)),
                ("free_form", free_form_mask(128, 128, np.random.default_rng(8), cfg))]
    # "both" examples: first two seeds for which the real mixed generator draws the "both" family
    both, seed = [], 0
    while len(both) < 2:
        m, fam = generate_mask(128, 128, np.random.default_rng(seed), cfg)
        if fam == "both":
            both.append(m)
        seed += 1
    examples += [("both", both[0]), ("both", both[1])]
    names = {"rect": "Rectangles", "free_form": "Free-form strokes", "both": "Rect + strokes"}

    fig = plt.figure(figsize=(7.2, 4.9))
    for k, (fam, m) in enumerate(examples):
        a = fig.add_axes([0.005 + k * 0.166, 0.63, 0.155, 0.30])
        shown = image * (1 - m[..., None]) + m[..., None] * 0.8
        a.imshow(shown)
        a.set_xticks([])
        a.set_yticks([])
        a.grid(False)
        for s in a.spines.values():
            s.set_visible(False)
        a.set_title(f"{names[fam]}\nhole {m.mean():.0%}", fontsize=7.5, color=INK2)
    fig.text(0.005, 0.985, "(a) Masks from the project generator on a validation image (grey = missing)",
             fontsize=9, weight="bold")

    ax = fig.add_axes([0.08, 0.08, 0.88, 0.45])
    rows = [r for r in csv.DictReader(open(ROOT / "results/metrics/per_image_metrics.csv")) if r["method"] == "unet"]
    ratios = np.array([float(r["hole_ratio"]) for r in rows]) * 100
    ax.hist(ratios, bins=np.arange(0, 56, 2), color="#86b6ef", edgecolor="white", linewidth=0.8)
    for b in (10, 20, 30, 40):
        ax.axvline(b, color=MUTED, lw=0.8, ls="--")
    ax.set_xlabel("Hole ratio (% of image pixels missing)")
    ax.set_ylabel("Test images")
    ax.set_xlim(0, 54)
    ax.text(0.99, 0.95, f"n = {len(ratios):,}\nmean {ratios.mean():.1f}%\nmax {ratios.max():.1f}%",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK2)
    fig.text(0.005, 0.585, "(b) Distribution of hole ratio over the 3,500 test masks (dashed: analysis buckets)",
             fontsize=9, weight="bold")
    save(fig, "fig01_masks.png")


# ----------------------------------------------------------------------------------------------
# Figure 4: original / mask / corrupted / reconstructions (validation images, 256 px inference)
# ----------------------------------------------------------------------------------------------
def fig_inference():
    cases = ["Places365_val_00031625_free_form", "Places365_val_00027525_rect", "Places365_val_00023471_center"]
    cols = [("original.png", "Original", "gated_unet_4gb"), ("mask.png", "Mask (white = hole)", "gated_unet_4gb"),
            ("masked.png", "Corrupted", "gated_unet_4gb"), ("inpainted.png", "U-Net", "unet_4gb"),
            ("inpainted.png", "Gated U-Net", "gated_unet_4gb")]
    fig, axes = plt.subplots(len(cases), len(cols), figsize=(7.2, 4.5))
    for i, c in enumerate(cases):
        for j, (f, t, run) in enumerate(cols):
            a = axes[i, j]
            a.imshow(img(ROOT / "results/inference" / run / c / f))
            a.set_xticks([])
            a.set_yticks([])
            a.grid(False)
            for s in a.spines.values():
                s.set_visible(False)
            if i == 0:
                a.set_title(t, fontsize=8.5)
    fig.subplots_adjust(wspace=0.03, hspace=0.04)
    save(fig, "fig07_inference_examples.png")


# ----------------------------------------------------------------------------------------------
# Figure 5: method comparison on the first six test images (no selection)
# ----------------------------------------------------------------------------------------------
def fig_test_comparison(n=6):
    rows = list(csv.DictReader(open(ROOT / "results/metrics/per_image_metrics.csv")))
    hp = {(int(r["index"]), r["method"]): float(r["hole_psnr"]) for r in rows}
    hr = {int(r["index"]): float(r["hole_ratio"]) for r in rows}
    rec = ROOT / "results/reconstructed"
    cols = [("inputs", "original", "Original"), ("inputs", "masked", "Corrupted")] + \
           [(m, "composite", LAB[m]) for m in ("mean_fill", "telea", "unet", "gated_unet")]
    fig, axes = plt.subplots(n, len(cols), figsize=(7.2, 7.9))
    for i in range(n):
        for j, (d, kind, title) in enumerate(cols):
            a = axes[i, j]
            a.imshow(img(rec / d / f"{i:04d}_{kind}.png"))
            a.set_xticks([])
            a.set_yticks([])
            a.grid(False)
            for s in a.spines.values():
                s.set_visible(False)
            if i == 0:
                a.set_title(title, fontsize=8.5)
            if d in COL:
                a.set_xlabel(f"{hp[(i, d)]:.1f} dB", fontsize=7.5, color=INK2, labelpad=1.5)
            if j == 1:
                a.set_xlabel(f"hole {hr[i]:.0%}", fontsize=7.5, color=INK2, labelpad=1.5)
    fig.subplots_adjust(wspace=0.04, hspace=0.20)
    save(fig, "fig06_test_comparison.png")


# ----------------------------------------------------------------------------------------------
# Figure 6: training curves
# ----------------------------------------------------------------------------------------------
def fig_curves():
    hist = {m: json.loads((ROOT / f"checkpoints/{m}_4gb/history.json").read_text()) for m in ("unet", "gated_unet")}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for m in ("gated_unet", "unet"):
        h = hist[m]
        ep = [r["epoch"] for r in h]
        axes[0].plot(ep, [r["train_total"] for r in h], color=COL[m], lw=1.6, ls="--", label=f"{LAB[m]} train")
        axes[0].plot(ep, [r["val_total"] for r in h], color=COL[m], lw=2, label=f"{LAB[m]} validation")
        axes[1].plot(ep, [r["val_hole_psnr"] for r in h], color=COL[m], lw=2, marker="o", ms=3, label=LAB[m])
    axes[0].set(xlabel="Epoch", ylabel="Total loss", title="(a) Training and validation loss")
    axes[1].set(xlabel="Epoch", ylabel="PSNR (dB)", title="(b) Validation hole PSNR (pooled, raw output)")
    axes[0].legend(frameon=False, fontsize=7.5)
    axes[1].legend(frameon=False, loc="lower right")
    for a in axes:
        a.set_xlim(0.5, 30.5)
    fig.tight_layout(w_pad=2)
    save(fig, "fig04_training_curves.png")


# ----------------------------------------------------------------------------------------------
# Figure 7: test metrics with 95% CI of the mean
# ----------------------------------------------------------------------------------------------
def fig_test_bars():
    s = json.loads((ROOT / "results/metrics/summary.json").read_text())["overall"]
    keys = [("full_psnr", "PSNR (dB)", "{:.2f}"), ("full_ssim", "SSIM", "{:.4f}"), ("hole_psnr", "Hole PSNR (dB)", "{:.2f}")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9))
    for a, (k, t, fmt) in zip(axes, keys):
        means = [s[m][k]["mean"] for m in ORDER]
        ci = [1.96 * s[m][k]["std"] / np.sqrt(s[m]["n"]) for m in ORDER]
        x = np.arange(len(ORDER))
        a.bar(x, means, width=0.62, color=[COL[m] for m in ORDER], edgecolor="white", linewidth=2)
        a.errorbar(x, means, yerr=ci, fmt="none", ecolor=INK2, capsize=2.5, lw=0.9)
        for xi, v, e in zip(x, means, ci):
            a.text(xi, v + e + (0.004 if k == "full_ssim" else 0.25), fmt.format(v), ha="center", va="bottom",
                   fontsize=7, color=INK)
        a.set_xticks(x)
        a.set_xticklabels([LAB[m].replace("OpenCV ", "").replace(" ", "\n", 1) for m in ORDER], fontsize=7.5)
        a.set_title(t)
        a.grid(axis="x", visible=False)
        if k == "full_ssim":
            a.set_ylim(0, 1.05)
        else:
            a.set_ylim(0, max(means) * 1.15)
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig05_test_metrics.png")


# ----------------------------------------------------------------------------------------------
# Figure 8: hole PSNR vs hole size (test)
# ----------------------------------------------------------------------------------------------
def fig_buckets():
    s = json.loads((ROOT / "results/metrics/summary.json").read_text())["by_hole_bucket"]
    buckets = list(s["unet"].keys())
    x = np.arange(len(buckets))
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    for m in ORDER:
        y = [s[m][b]["hole_psnr"]["mean"] for b in buckets]
        ax.plot(x, y, color=COL[m], lw=2, marker="o", ms=5, mec="white", mew=1.2, label=LAB[m])
        if m in ("telea", "mean_fill"):
            ax.text(x[-1] + 0.08, y[-1], LAB[m], va="center", fontsize=7.5, color=INK2)
        elif m == "unet":  # the two networks' lines coincide; one shared label
            ax.text(x[-1] + 0.08, y[-1] + 0.15, "U-Net and\nGated U-Net\n(overlap)", va="center", fontsize=7.5,
                    color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\n(n = {s['unet'][b]['n']:,})" for b in buckets])
    ax.set_xlim(-0.2, len(buckets) - 0.2)
    ax.set_xlabel("Hole ratio bucket (test images per bucket)")
    ax.set_ylabel("Mean hole PSNR (dB)")
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    save(fig, "fig08_hole_size.png")


if __name__ == "__main__":
    fig_pipeline()
    fig_architecture()
    fig_masks()
    fig_inference()
    fig_test_comparison()
    fig_curves()
    fig_test_bars()
    fig_buckets()
