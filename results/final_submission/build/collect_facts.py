"""Collect every number cited in the report from the project's saved files (read-only).

Run with the project venv:  .venv/bin/python results/final_submission/build/collect_facts.py
Writes results/final_submission/build/facts.json.
"""
import csv, json, statistics, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent / "facts.json"
METHODS = ["gated_unet", "unet", "telea", "mean_fill"]


def paired(rows, a, b, key):
    by = {}
    for r in rows:
        by.setdefault(r["method"], {})[int(r["index"])] = float(r[key])
    d = np.array([by[a][i] - by[b][i] for i in sorted(by[a])])
    se = d.std(ddof=1) / np.sqrt(len(d))
    return {"mean": float(d.mean()), "ci_lo": float(d.mean() - 1.96 * se), "ci_hi": float(d.mean() + 1.96 * se),
            "frac_better": float((d > 0).mean()), "n": int(len(d))}


def eval_block(metrics_dir):
    s = json.loads((metrics_dir / "summary.json").read_text())
    rows = list(csv.DictReader(open(metrics_dir / "per_image_metrics.csv")))
    out = {"protocol": s["protocol"], "overall": s["overall"], "by_mask_family": s["by_mask_family"],
           "by_hole_bucket": s["by_hole_bucket"], "n_rows": len(rows), "has_lpips": "full_lpips" in rows[0],
           "paired": {}}
    for a, b in (("gated_unet", "unet"), ("unet", "telea"), ("gated_unet", "telea"), ("telea", "mean_fill")):
        out["paired"][f"{a}-{b}"] = {k: paired(rows, a, b, k) for k in ("hole_psnr", "full_psnr", "full_ssim")}
    ratios = [float(r["hole_ratio"]) for r in rows if r["method"] == "unet"]
    out["hole_ratio"] = {"mean": float(np.mean(ratios)), "median": float(np.median(ratios)),
                         "min": float(np.min(ratios)), "max": float(np.max(ratios))}
    return out


facts = {"val": eval_block(ROOT / "results/val_eval/both_4gb/metrics"),
         "test": eval_block(ROOT / "results/metrics")}

facts["dataset"] = json.loads((ROOT / "data/splits/dataset_stats.json").read_text())
facts["splits"] = {k: len((ROOT / f"data/splits/{k}.txt").read_text().split()) for k in ("train", "val", "test")}

runs = {}
for name in ("unet_4gb", "gated_unet_4gb"):
    d = ROOT / "checkpoints" / name
    hist = json.loads((d / "history.json").read_text())
    ck = torch.load(d / "best.pt", map_location="cpu", weights_only=False)
    times = [h["time_s"] for h in hist]
    gpu = [l.split(", ") for l in (ROOT / f"results/logs/{name}_gpu.csv").read_text().splitlines() if l.strip()]
    best = min(hist, key=lambda h: h["val_total"])
    runs[name] = {
        "summary": json.loads((d / "summary.json").read_text()),
        "config": json.loads((d / "config.json").read_text()),
        "best_epoch": ck["epoch"], "best_val": ck["best_val"], "torch_version": ck["torch_version"],
        "epochs": len(hist), "first": hist[0], "last": hist[-1], "best_row": best,
        "median_epoch_s": statistics.median(times), "max_epoch_s": max(times),
        "epochs_over_1h": [h["epoch"] for h in hist if h["time_s"] > 3600],
        "gpu_peak_mib": max(int(r[1].split()[0]) for r in gpu),
        "gpu_max_temp_c": max(int(r[3]) for r in gpu if len(r) > 3),
    }
facts["runs"] = runs
facts["n_full_runs_per_model"] = {m: sorted(p.name for p in (ROOT / "checkpoints").iterdir()
                                            if p.is_dir() and not p.name.startswith("pilot")
                                            and json.loads((p / "config.json").read_text())["model"]["name"] == m)
                                  for m in ("unet", "gated_unet")}

import cv2, matplotlib, PIL, skimage, torchvision, yaml
facts["versions"] = {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda,
                     "cudnn": torch.backends.cudnn.version(), "torchvision": torchvision.__version__,
                     "numpy": np.__version__, "opencv": cv2.__version__, "pillow": PIL.__version__,
                     "pyyaml": yaml.__version__, "scikit_image": skimage.__version__,
                     "matplotlib": matplotlib.__version__}
facts["gpu_name"] = runs["unet_4gb"]["summary"]["gpu"]
OUT.write_text(json.dumps(facts, indent=1))
print("wrote", OUT)
for split in ("val", "test"):
    print(split, {m: round(facts[split]["overall"][m]["full_psnr"]["mean"], 2) for m in METHODS},
          "lpips:", facts[split]["has_lpips"], "rows:", facts[split]["n_rows"])
print({k: (v["best_epoch"], round(v["median_epoch_s"]), v["epochs_over_1h"], v["gpu_peak_mib"]) for k, v in runs.items()})
print("full runs per model:", facts["n_full_runs_per_model"], "| seeds:", {k: v["config"]["seed"] for k, v in runs.items()})
