"""Loads the trained checkpoints once and runs real inference.

Checkpoints are loaded with the project's own ``load_model_from_checkpoint``
(strict state-dict loading, so an incompatible file fails here rather than
producing garbage). A model whose checkpoint is missing or invalid is reported
as unavailable with the reason; random weights are never substituted.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from src.models import count_parameters
from src.training.trainer import load_model_from_checkpoint
from src.utils.common import get_device

from .config import MODEL_LABELS, Settings
from .imaging import composite, prediction_to_original, to_model_input

log = logging.getLogger("inpaint.engine")


class ModelUnavailable(RuntimeError):
    pass


@dataclass
class LoadedModel:
    name: str
    checkpoint: Path
    available: bool
    error: str | None = None
    model: torch.nn.Module | None = None
    info: dict = field(default_factory=dict)


@dataclass
class InpaintResult:
    result: np.ndarray        # (H, W, 3) uint8 composite at the original resolution
    model: str
    work_hw: tuple[int, int]
    hole_fraction: float
    inference_ms: float
    total_ms: float


def _test_metrics(path: Path | None) -> dict:
    """Test-set metrics from results/metrics/summary.json, if the evaluation has been run."""
    if not path or not path.is_file():
        return {}
    try:
        overall = json.loads(path.read_text())["overall"]
    except (ValueError, KeyError):
        return {}
    out = {}
    for name, m in overall.items():
        try:
            out[name] = {"n_images": m["n"], "psnr": m["full_psnr"]["mean"], "ssim": m["full_ssim"]["mean"],
                         "hole_psnr": m["hole_psnr"]["mean"]}
        except KeyError:
            continue
    return out


class Engine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.device = get_device(settings.device)
        self._lock = threading.Lock()  # one inference at a time on a small GPU
        metrics = _test_metrics(settings.test_metrics)
        self.models: dict[str, LoadedModel] = {}
        for name, path in settings.checkpoints.items():
            self.models[name] = self._load(name, path, metrics.get(name))

    def _load(self, name: str, path: Path, metrics: dict | None) -> LoadedModel:
        try:
            model, ckpt = load_model_from_checkpoint(path, self.device)
            if ckpt["model_name"] != name:
                raise ValueError(f"{path} contains a '{ckpt['model_name']}' model, not '{name}'.")
            size = int(ckpt.get("image_size", 128))
            with torch.no_grad():  # warm-up and sanity check of the loaded weights
                x = torch.zeros(1, 3, size, size, device=self.device)
                out = model(x, torch.ones(1, 1, size, size, device=self.device))
            if not torch.isfinite(out).all():
                raise ValueError(f"{path} produces non-finite outputs.")
        except Exception as exc:  # noqa: BLE001 - any load failure makes the model unavailable
            log.warning("Model %s unavailable: %s", name, exc)
            return LoadedModel(name, path, available=False, error=str(exc))
        info = {
            "epoch": ckpt.get("epoch"),
            "training_resolution": size,
            "base_channels": ckpt["base_channels"],
            "parameters": count_parameters(model),
            "best_val_loss": ckpt.get("best_val"),
            "saved_at": ckpt.get("saved_at"),
            "test_metrics": metrics,
        }
        log.info("Loaded %s from %s (epoch %s) on %s", name, path, info["epoch"], self.device)
        return LoadedModel(name, path, available=True, model=model, info=info)

    @property
    def available(self) -> list[str]:
        return [n for n, m in self.models.items() if m.available]

    @property
    def default_model(self) -> str | None:
        if self.settings.default_model in self.available:
            return self.settings.default_model
        return self.available[0] if self.available else None

    def default_work_size(self, name: str | None = None) -> int:
        if self.settings.work_size:
            return self.settings.work_size
        m = self.models.get(name or self.default_model or "")
        return int(m.info["training_resolution"]) if m and m.available else 128

    def describe(self) -> dict:
        return {
            "device": str(self.device),
            "cuda_device": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None,
            "default_model": self.default_model,
            "default_work_size": self.default_work_size(),
            "max_work_size": self.settings.max_work_size,
            "max_upload_mb": round(self.settings.max_upload_bytes / 1024 / 1024, 1),
            "max_pixels": self.settings.max_pixels,
            "mask_convention": "white (1) = missing region to reconstruct, black (0) = known region",
            "models": [
                {"name": m.name, "label": MODEL_LABELS.get(m.name, m.name), "available": m.available,
                 "checkpoint": _display_path(m.checkpoint), "error": m.error, **m.info}
                for m in self.models.values()
            ],
        }

    def inpaint(self, rgb: np.ndarray, mask: np.ndarray, model_name: str, work_size: int,
                resample: str = "normalized") -> InpaintResult:
        entry = self.models.get(model_name)
        if entry is None:
            raise ModelUnavailable(f"Unknown model '{model_name}'. Choose one of {list(self.models)}.")
        if not entry.available:
            raise ModelUnavailable(f"The {MODEL_LABELS.get(model_name, model_name)} checkpoint is unavailable: "
                                   f"{entry.error}")
        t0 = time.perf_counter()
        x = to_model_input(rgb, mask, work_size, resample)
        img_t = torch.from_numpy(x.image.transpose(2, 0, 1).copy())[None].to(self.device)
        mask_t = torch.from_numpy(x.mask)[None, None].to(self.device)
        with self._lock, torch.no_grad():
            t1 = time.perf_counter()
            # Same input convention as training: holes zeroed, mask as 4th channel.
            pred = entry.model(img_t * (1 - mask_t), mask_t)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            t2 = time.perf_counter()
        pred = pred[0].float().cpu().numpy().transpose(1, 2, 0)
        pred_full = prediction_to_original(pred, x.hw, mask.shape)
        result = composite(rgb, pred_full, mask)
        return InpaintResult(result=result, model=model_name, work_hw=x.hw, hole_fraction=float(mask.mean()),
                             inference_ms=(t2 - t1) * 1000, total_ms=(time.perf_counter() - t0) * 1000)


def _display_path(p: Path) -> str:
    from .config import PROJECT_ROOT
    try:
        return str(p.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)
