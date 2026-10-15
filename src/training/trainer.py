"""Training loop with validation, checkpointing, resuming, early stopping and logging."""
from __future__ import annotations

import csv
import json
import math
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from src.datasets.inpainting_dataset import InpaintingDataset, read_split
from src.losses.inpainting_loss import InpaintingLoss
from src.masks import mask_config_from_dict
from src.models import build_model, composite, count_parameters
from src.utils.common import get_device, resolve_path, save_json, seed_everything


def build_dataloaders(cfg: dict) -> tuple[DataLoader, DataLoader]:
    data, train_cfg = cfg["data"], cfg["train"]
    root = resolve_path(data["root"])
    splits = resolve_path(data["splits_dir"])
    mask_cfg = mask_config_from_dict(cfg.get("mask"))

    train_files = read_split(splits / "train.txt")
    val_files = read_split(splits / "val.txt")
    if data.get("max_train_images"):
        train_files = train_files[: data["max_train_images"]]
    if data.get("max_val_images"):
        val_files = val_files[: data["max_val_images"]]

    size = data["image_size"]
    train_ds = InpaintingDataset(root, train_files, size, mask_cfg, augment=True, deterministic=False)
    # Validation masks are fixed (seeded per index) so validation losses are comparable across epochs.
    val_ds = InpaintingDataset(root, val_files, size, mask_cfg, augment=False, deterministic=True,
                               seed=cfg["seed"] + 1_000_000)

    workers = train_cfg.get("num_workers", 4)
    g = torch.Generator()
    g.manual_seed(cfg["seed"])
    train_dl = DataLoader(train_ds, batch_size=train_cfg["batch_size"], shuffle=True, drop_last=True,
                          num_workers=workers, pin_memory=True, generator=g,
                          persistent_workers=workers > 0)
    val_dl = DataLoader(val_ds, batch_size=train_cfg["batch_size"], shuffle=False,
                        num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    return train_dl, val_dl


def save_checkpoint(path: Path, model, optimizer, scheduler, scaler, epoch: int, best_val: float,
                    cfg: dict, history: list[dict], extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model_name": cfg["model"]["name"],
        "base_channels": cfg["model"]["base_channels"],
        "image_size": cfg["data"]["image_size"],
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "scaler_state": scaler.state_dict(),
        "epoch": epoch,
        "best_val": best_val,
        "config": cfg,
        "history": history,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "torch_version": torch.__version__,
    }
    if extra:
        state.update(extra)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)  # atomic replace so an interrupted save never corrupts a checkpoint


def load_model_from_checkpoint(path: str | Path, device: torch.device | str = "cpu"):
    """Return ``(model, checkpoint_dict)`` with weights loaded and the model in eval mode."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. Train a model first (scripts/train.py) - "
            "an untrained network cannot inpaint meaningfully.")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    for key in ("model_name", "base_channels", "model_state"):
        if key not in ckpt:
            raise ValueError(f"{path} is not a valid checkpoint for this project (missing '{key}').")
    model = build_model(ckpt["model_name"], ckpt["base_channels"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


class Trainer:
    def __init__(self, cfg: dict, run_dir: Path, device: str = "auto", resume: str | None = None):
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.device = get_device(device)
        seed_everything(cfg["seed"])

        tc = cfg["train"]
        self.model = build_model(cfg["model"]["name"], cfg["model"]["base_channels"]).to(self.device)
        lc = cfg["loss"]
        self.criterion = InpaintingLoss(lc["w_hole"], lc["w_valid"], lc["w_perceptual"],
                                        pretrained_vgg=lc.get("pretrained_vgg", True)).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=tc["lr"], betas=tuple(tc.get("betas", (0.9, 0.999))))
        self.scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=tc["epochs"],
                                                                     eta_min=tc["lr"] * 0.05)
                          if tc.get("cosine_schedule", True) else None)
        self.use_amp = bool(tc.get("amp", True)) and self.device.type == "cuda"
        if hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.train_dl, self.val_dl = build_dataloaders(cfg)

        self.start_epoch, self.best_val, self.history, self.bad_epochs = 1, math.inf, [], 0
        if resume:
            self._resume(resume)

    def _resume(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        if self.scheduler and ckpt.get("scheduler_state"):
            self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self.scaler.load_state_dict(ckpt["scaler_state"])
        self.start_epoch = ckpt["epoch"] + 1
        self.best_val = ckpt["best_val"]
        self.history = ckpt.get("history", [])
        print(f"Resumed from {path} at epoch {ckpt['epoch']} (best val {self.best_val:.4f})")

    def _step(self, batch: dict, train: bool) -> dict:
        image = batch["image"].to(self.device, non_blocking=True)
        mask = batch["mask"].to(self.device, non_blocking=True)
        masked = batch["masked"].to(self.device, non_blocking=True)
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
            pred = self.model(masked, mask)
        losses = self.criterion(pred.float(), image, mask)
        if not torch.isfinite(losses["total"]):
            raise FloatingPointError(f"Non-finite loss encountered: {losses}")
        if train:
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(losses["total"]).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["train"].get("grad_clip", 1.0))
            self.scaler.step(self.optimizer)
            self.scaler.update()
        # Hole-region MSE of the raw prediction, used to report a validation PSNR.
        with torch.no_grad():
            m3 = mask.expand_as(image)
            hole_mse = ((pred.float() - image) ** 2 * m3).sum() / m3.sum().clamp_min(1)
        out = {k: float(v.detach()) for k, v in losses.items()}
        out["hole_mse"] = float(hole_mse)
        return out

    def _run_epoch(self, loader: DataLoader, train: bool, max_steps: int | None = None) -> dict:
        self.model.train(train)
        totals, n = {}, 0
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for i, batch in enumerate(loader):
                if max_steps is not None and i >= max_steps:
                    break
                out = self._step(batch, train)
                bs = batch["image"].shape[0]
                for k, v in out.items():
                    totals[k] = totals.get(k, 0.0) + v * bs
                n += bs
        means = {k: v / max(n, 1) for k, v in totals.items()}
        means["hole_psnr"] = 10 * math.log10(1.0 / max(means.get("hole_mse", 1.0), 1e-10))
        return means

    @torch.no_grad()
    def _save_samples(self, epoch: int, n: int = 6) -> None:
        self.model.eval()
        batch = next(iter(self.val_dl))
        image, mask, masked = (batch[k][:n].to(self.device) for k in ("image", "mask", "masked"))
        pred = self.model(masked, mask).float()
        comp = composite(pred, image, mask)
        # Show holes in grey-white for readability in the "input" row.
        shown_input = masked + mask * 0.8
        grid = torch.cat([shown_input, pred, comp, image], dim=0)
        out = self.run_dir / "samples" / f"epoch_{epoch:03d}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        save_image(grid, out, nrow=n, padding=2, pad_value=1.0)

    def _log(self, row: dict) -> None:
        self.history.append(row)
        log_csv = self.run_dir / "training_log.csv"
        write_header = not log_csv.exists()
        with log_csv.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                w.writeheader()
            w.writerow(row)

    def fit(self) -> dict:
        tc = self.cfg["train"]
        self.run_dir.mkdir(parents=True, exist_ok=True)
        save_json(self.cfg, self.run_dir / "config.json")
        n_params = count_parameters(self.model)
        print(f"Model {self.cfg['model']['name']} | params {n_params:,} | device {self.device} | "
              f"AMP {self.use_amp} | train {len(self.train_dl.dataset)} | val {len(self.val_dl.dataset)}")
        patience = tc.get("early_stopping_patience", 0)
        max_steps = tc.get("max_steps_per_epoch")
        epoch = self.start_epoch - 1
        status = "completed"
        try:
            for epoch in range(self.start_epoch, tc["epochs"] + 1):
                t0 = time.time()
                tr = self._run_epoch(self.train_dl, train=True, max_steps=max_steps)
                va = self._run_epoch(self.val_dl, train=False, max_steps=tc.get("max_val_steps"))
                if self.scheduler:
                    self.scheduler.step()
                row = {"epoch": epoch, "lr": self.optimizer.param_groups[0]["lr"],
                       "time_s": round(time.time() - t0, 1)}
                row.update({f"train_{k}": round(v, 6) for k, v in tr.items()})
                row.update({f"val_{k}": round(v, 6) for k, v in va.items()})
                self._log(row)
                improved = va["total"] < self.best_val
                if improved:
                    self.best_val, self.bad_epochs = va["total"], 0
                else:
                    self.bad_epochs += 1
                save_checkpoint(self.run_dir / "last.pt", self.model, self.optimizer, self.scheduler,
                                self.scaler, epoch, self.best_val, self.cfg, self.history)
                if improved:
                    save_checkpoint(self.run_dir / "best.pt", self.model, self.optimizer, self.scheduler,
                                    self.scaler, epoch, self.best_val, self.cfg, self.history)
                if epoch % tc.get("sample_every", 1) == 0:
                    self._save_samples(epoch)
                print(f"epoch {epoch:3d} | train {tr['total']:.4f} | val {va['total']:.4f} | "
                      f"val hole PSNR {va['hole_psnr']:.2f} dB | {row['time_s']}s"
                      f"{' | *best*' if improved else ''}", flush=True)
                if patience and self.bad_epochs >= patience:
                    status = f"early_stopped_at_epoch_{epoch}"
                    print(f"Early stopping: no validation improvement for {patience} epochs.")
                    break
        except KeyboardInterrupt:
            status = f"interrupted_at_epoch_{epoch}"
            print("Interrupted - saving checkpoint to interrupted.pt")
            save_checkpoint(self.run_dir / "interrupted.pt", self.model, self.optimizer, self.scheduler,
                            self.scaler, epoch, self.best_val, self.cfg, self.history)
        summary = {"status": status, "epochs_completed": len(self.history), "best_val_total": self.best_val,
                   "parameters": n_params, "device": str(self.device),
                   "gpu": torch.cuda.get_device_name(0) if self.device.type == "cuda" else None}
        save_json(summary, self.run_dir / "summary.json")
        (self.run_dir / "history.json").write_text(json.dumps(self.history, indent=2))
        return summary
