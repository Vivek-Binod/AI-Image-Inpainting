"""Backend settings, read from environment variables (all optional).

INPAINT_UNET_CHECKPOINT        default checkpoints/unet_4gb/best.pt
INPAINT_GATED_UNET_CHECKPOINT  default checkpoints/gated_unet_4gb/best.pt
INPAINT_DEFAULT_MODEL          default gated_unet (falls back to any available model)
INPAINT_DEVICE                 auto | cpu | cuda | cuda:N   (default auto)
INPAINT_WORK_SIZE              longer side, in px, at which the network runs (default: the
                               checkpoint's training resolution, 128)
INPAINT_MAX_WORK_SIZE          upper limit for a per-request work_size (default 512)
INPAINT_MAX_UPLOAD_MB          per-file upload limit (default 25)
INPAINT_MAX_PIXELS             image size limit in pixels (default 40,000,000)
INPAINT_FRONTEND_DIST          built frontend served at / (default frontend/dist; ignored if missing)
INPAINT_CORS_ORIGINS           comma-separated extra origins allowed by CORS
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_LABELS = {"unet": "U-Net", "gated_unet": "Gated U-Net"}


def _path(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else PROJECT_ROOT / p


@dataclass
class Settings:
    checkpoints: dict[str, Path] = field(default_factory=dict)
    default_model: str = "gated_unet"
    device: str = "auto"
    work_size: int | None = None
    max_work_size: int = 512
    max_upload_bytes: int = 25 * 1024 * 1024
    max_pixels: int = 40_000_000
    frontend_dist: Path | None = None
    cors_origins: list[str] = field(default_factory=list)
    test_metrics: Path | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ.get
        dist = _path(env("INPAINT_FRONTEND_DIST", "frontend/dist"))
        work = env("INPAINT_WORK_SIZE")
        return cls(
            checkpoints={
                "unet": _path(env("INPAINT_UNET_CHECKPOINT", "checkpoints/unet_4gb/best.pt")),
                "gated_unet": _path(env("INPAINT_GATED_UNET_CHECKPOINT", "checkpoints/gated_unet_4gb/best.pt")),
            },
            default_model=env("INPAINT_DEFAULT_MODEL", "gated_unet"),
            device=env("INPAINT_DEVICE", "auto"),
            work_size=int(work) if work else None,
            max_work_size=int(env("INPAINT_MAX_WORK_SIZE", "512")),
            max_upload_bytes=int(float(env("INPAINT_MAX_UPLOAD_MB", "25")) * 1024 * 1024),
            max_pixels=int(env("INPAINT_MAX_PIXELS", "40000000")),
            frontend_dist=dist if (dist / "index.html").is_file() else None,
            cors_origins=[o.strip() for o in env("INPAINT_CORS_ORIGINS", "").split(",") if o.strip()],
            test_metrics=_path("results/metrics/summary.json"),
        )
