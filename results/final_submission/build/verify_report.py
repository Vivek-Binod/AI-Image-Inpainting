"""Independent verification of the final report against the raw saved project files.

Recomputes every reported statistic from results/*/per_image_metrics.csv and checkpoints/*/history.json
(it does NOT read facts.json or summary.json), then checks the final Markdown report.

    .venv/bin/python results/final_submission/build/verify_report.py
"""
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
FS = ROOT / "results/final_submission"
MD = (FS / "AI_Image_Inpainting_Mini_Project_Report.md").read_text().replace(" ", " ")
LAB = {"gated_unet": "Gated U-Net", "unet": "U-Net", "telea": "OpenCV Telea", "mean_fill": "Mean fill"}
ORDER = ["gated_unet", "unet", "telea", "mean_fill"]
fails, passes = [], 0


def check(cond, msg):
    global passes
    if cond:
        passes += 1
    else:
        fails.append(msg)


def load(path):
    rows = list(csv.DictReader(open(path)))
    by = {}
    for r in rows:
        by.setdefault(r["method"], []).append(r)
    return rows, by


def ms(rs, k):
    v = np.array([float(r[k]) for r in rs])
    return v.mean(), v.std(ddof=1)


plain = MD.replace("**", "")
data = {"val": load(ROOT / "results/val_eval/both_4gb/metrics/per_image_metrics.csv"),
        "test": load(ROOT / "results/metrics/per_image_metrics.csv")}

# 1. Overall tables (Tables 8 and 9), recomputed from per-image rows
for split, (rows, by) in data.items():
    check(len(rows) == {"val": 4 * 3000, "test": 4 * 3500}[split], f"{split}: unexpected row count {len(rows)}")
    check(not any("lpips" in k for k in rows[0]), f"{split}: LPIPS column present but report says not computed")
    for m in ORDER:
        p, s, h = ms(by[m], "full_psnr"), ms(by[m], "full_ssim"), ms(by[m], "hole_psnr")
        row = f"| {LAB[m]} | {p[0]:.2f} ± {p[1]:.2f} | {s[0]:.4f} ± {s[1]:.4f} | {h[0]:.2f} ± {h[1]:.2f} |"
        check(row in plain, f"{split} table row missing/mismatched: {row}")

# 2. Values given by the user for the test table
user = {"gated_unet": (27.72, 0.9059, 19.16), "unet": (27.69, 0.9057, 19.14), "telea": (25.98, 0.8913, 17.43),
        "mean_fill": (21.68, 0.8399, 13.13)}
_, tb = data["test"]
for m, (p, s, h) in user.items():
    check(round(ms(tb[m], "full_psnr")[0], 2) == p and round(ms(tb[m], "full_ssim")[0], 4) == s
          and round(ms(tb[m], "hole_psnr")[0], 2) == h, f"user-supplied test value mismatch for {m}")

# 3. Validation-vs-test table and the "< 0.1 dB" claim
_, vb = data["val"]
for m in ("gated_unet", "unet"):
    for split, b, n in (("Validation", vb, "3,000"), ("Test", tb, "3,500")):
        row = (f"| {LAB[m]} | {split} | {n} | {ms(b[m], 'full_psnr')[0]:.2f} | {ms(b[m], 'full_ssim')[0]:.4f} | "
               f"{ms(b[m], 'hole_psnr')[0]:.2f} |")
        check(row in plain, f"val-vs-test row mismatch: {row}")
    diff = max(abs(ms(vb[m], k)[0] - ms(tb[m], k)[0]) for k in ("full_psnr", "hole_psnr"))
    check(diff < 0.1, f"claim 'largest val/test difference below 0.1 dB' false for {m}: {diff:.3f}")

# 4. Gated U-Net best on all three metrics (claimed for test; also holds for val per text)
for split, b in (("val", vb), ("test", tb)):
    for k in ("full_psnr", "full_ssim", "hole_psnr"):
        best = max(ORDER, key=lambda m: ms(b[m], k)[0])
        check(best == "gated_unet", f"{split}: gated U-Net is not best on {k} (best: {best})")


# 5. Paired differences (Table 11 and text)
def paired(b, a, c, k):
    x = {int(r["index"]): float(r[k]) for r in b[a]}
    y = {int(r["index"]): float(r[k]) for r in b[c]}
    d = np.array([x[i] - y[i] for i in sorted(x)])
    se = d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), d.mean() - 1.96 * se, d.mean() + 1.96 * se, (d > 0).mean()


def sgn(v, f):
    s = format(v, f)
    return s.replace("-", "−") if s.startswith("-") else s


for a, c in (("gated_unet", "unet"), ("gated_unet", "telea"), ("unet", "telea"), ("telea", "mean_fill")):
    mh, lo, hi, fb = paired(tb, a, c, "hole_psnr")
    sm = paired(tb, a, c, "full_ssim")[0]
    row = (f"| {LAB[a]} − {LAB[c]} | {sgn(mh, '+.3f')} | [{sgn(lo, '+.3f')}, {sgn(hi, '+.3f')}] | "
           f"{sgn(sm, '+.4f')} | {fb * 100:.1f} % |")
    check(row in plain, f"paired row mismatch: {row}")
mh, lo, hi, fb = paired(tb, "gated_unet", "unet", "hole_psnr")
check(f"{sgn(mh, '+.3f')} dB, with a 95 % confidence interval of [{sgn(lo, '+.3f')}, {sgn(hi, '+.3f')}] dB" in plain,
      "paired gated-unet sentence mismatch")
check(f"{sgn(paired(vb, 'gated_unet', 'unet', 'hole_psnr')[0], '+.3f')} dB)" in plain, "validation paired value mismatch")
check(0.015 < mh < 0.03, "'about 0.02 dB' claim out of range")
for a, c in (("unet", "telea"), ("gated_unet", "telea")):
    check(1.65 < paired(tb, a, c, "hole_psnr")[0] < 1.8, f"'about 1.7 dB' claim out of range for {a}")
check(5.5 < paired(tb, "unet", "mean_fill", "hole_psnr")[0] < 6.5, "'about 6 dB on mean fill' claim out of range")


# 6. Breakdown tables (12 and 13) and claims about them
def group_means(b, key):
    out = {}
    for m in ORDER:
        for r in b[m]:
            out.setdefault(r[key], {}).setdefault(m, []).append(float(r["hole_psnr"]))
    return {g: {m: (np.mean(v), len(v)) for m, v in d.items()} for g, d in out.items()}


fam = group_means(tb, "mask_family")
names = {"rect": "Rectangles", "free_form": "Free-form strokes", "both": "Both"}
for g, d in fam.items():
    row = f"| {names[g]} | {d['unet'][1]:,} | " + " | ".join(f"{d[m][0]:.2f}" for m in ORDER) + " |"
    check(row in plain, f"mask-family row mismatch: {row}")
for m in ORDER:
    check(min(fam, key=lambda g: fam[g][m][0]) == "rect", f"'rectangles hardest for every method' false for {m}")
buck = group_means(tb, "hole_bucket")
order_b = ["0-10%", "10-20%", "20-30%", "30-40%", "40-100%"]
for g in order_b:
    d = buck[g]
    row = f"| {g} | {d['unet'][1]:,} | " + " | ".join(f"{d[m][0]:.2f}" for m in ORDER) + " |"
    check(row in plain, f"hole-bucket row mismatch: {row}")
for m in ("gated_unet", "unet", "telea"):
    seq = [buck[g][m][0] for g in order_b]
    check(all(x > y for x, y in zip(seq, seq[1:])), f"'hole PSNR falls as hole grows' not monotonic for {m}")
for g in order_b:
    for m in ("gated_unet", "unet"):
        check(buck[g][m][0] > buck[g]["telea"][0], f"networks not above Telea in bucket {g}")
for g in fam:
    for m in ("gated_unet", "unet"):
        check(fam[g][m][0] > fam[g]["telea"][0], f"networks not above Telea for family {g}")
check(f"from {buck['0-10%']['gated_unet'][0]:.2f} dB for holes below 10 % of the image to "
      f"{buck['40-100%']['gated_unet'][0]:.2f} dB" in plain, "bucket sentence mismatch")
ratios = np.array([float(r["hole_ratio"]) for r in tb["unet"]])
check(f"mean of {ratios.mean() * 100:.1f} %, a median of {np.median(ratios) * 100:.1f} % and a maximum of "
      f"{ratios.max() * 100:.1f} %" in plain, "hole ratio sentence mismatch")

# 7. Figure 6 per-image numbers (first six test images)
# (figure is generated from the same CSV; spot-check that indices 0-5 exist with the right paths)
paths = [r["path"] for r in tb["unet"] if int(r["index"]) < 6]
split_test = (ROOT / "data/splits/test.txt").read_text().split()
check(paths == split_test[:6], "Figure 6 images are not the first six test images")

# 8. Training logs (Table 7, Figure 4 claims)
for name, lab in (("unet_4gb", "U-Net"), ("gated_unet_4gb", "Gated U-Net")):
    h = json.loads((ROOT / f"checkpoints/{name}/history.json").read_text())
    check(len(h) == 30, f"{name}: not 30 epochs")
    best = min(h, key=lambda r: r["val_total"])
    check(best["epoch"] == 30, f"{name}: best epoch is {best['epoch']}, report says 30")
    check(f"{best['val_total']:.4f} (30)" in plain, f"{name}: best val loss not in Table 7")
    check(f"{h[-1]['train_total']:.4f}" in plain, f"{name}: final train loss not in Table 7")
    cfg = json.loads((ROOT / f"checkpoints/{name}/config.json").read_text())
    check(cfg["seed"] == 42 and cfg["train"]["batch_size"] == 16 and cfg["train"]["lr"] == 2e-4
          and cfg["loss"] == {"w_hole": 6.0, "w_valid": 1.0, "w_perceptual": 0.05, "pretrained_vgg": True}
          and cfg["data"]["image_size"] == 128 and cfg["model"]["base_channels"] == 32 and cfg["train"]["amp"],
          f"{name}: config differs from Table 5")
full_runs = [p for p in (ROOT / "checkpoints").iterdir() if p.is_dir() and not p.name.startswith("pilot")]
check(sorted(p.name for p in full_runs) == ["gated_unet_4gb", "unet_4gb"], "more than one full run per model found")

# 9. Citations, references, figures and tables
body, refs = MD.split("\n# References", 1)
nums = {int(n) for grp in re.findall(r"\[(\d+(?:\], \[\d+)*)\]", body) for n in re.findall(r"\d+", grp)}
refnums = [int(n) for n in re.findall(r"^\[(\d+)\] ", refs, re.M)]
check(refnums == list(range(1, len(refnums) + 1)), "reference list not numbered consecutively")
check(nums == set(refnums), f"cited {sorted(nums - set(refnums))} not in list / uncited {sorted(set(refnums) - nums)}")
first = [int(n) for n in re.findall(r"\[(\d+)\]", body)]
seen = []
for n in first:
    if n not in seen:
        seen.append(n)
check(seen == sorted(seen), "citations not numbered in order of first appearance")
for f in re.findall(r"\]\((figures/[^)]+)\)", MD):
    check((FS / f).is_file(), f"missing figure file {f}")
for kind, n in (("Figure", 8), ("Table", 13)):
    for i in range(1, n + 1):
        caps = len(re.findall(rf"(?:!\[|Table: ){kind} {i}\.", MD))
        refs_in_text = len(re.findall(rf"{kind}s? (?:\d+ and )?{i}\b|{kind}s \d+ and {i}\b|{kind} {i}[ab]\b", body)) - caps
        check(caps == 1, f"{kind} {i}: caption count {caps}")
        check(refs_in_text >= 1, f"{kind} {i}: not referred to in the text")

# 10. Placeholders / leftovers
check(not re.search(r"\{\{[A-Za-z]|\[@", MD), "unfilled template placeholders or citation keys remain")
placeholders = sorted(set(re.findall(r"\[(?:Student Name|Register Number|Degree Name|Branch / Specialisation|"
                                     r"Faculty Guide|Designation|Department|Institution|Academic Year)\]", MD)))

print(f"checks passed: {passes}, failed: {len(fails)}")
for f in fails:
    print("  FAIL:", f)
print("title-page placeholders present (expected):", placeholders)
sys.exit(1 if fails else 0)
