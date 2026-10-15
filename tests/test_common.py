"""Unit tests for utility functions (src/utils/common.py)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.utils.common import (
    PROJECT_ROOT,
    deep_update,
    get_device,
    resolve_path,
    save_json,
    seed_everything,
    set_overrides,
)


def test_deep_update():
    base = {"a": 1, "nested": {"x": 10, "y": 20}}
    override = {"nested": {"y": 99, "z": 30}, "b": 2}
    result = deep_update(base, override)

    assert result == {"a": 1, "nested": {"x": 10, "y": 99, "z": 30}, "b": 2}
    # Base dictionary must not be mutated
    assert base["nested"]["y"] == 20


def test_set_overrides():
    cfg = {"train": {"epochs": 10, "lr": 0.001}, "model": {"name": "unet"}}
    overrides = ["train.epochs=25", "train.lr=0.0005", "eval.batch_size=16"]
    updated = set_overrides(cfg, overrides)

    assert updated["train"]["epochs"] == 25
    assert updated["train"]["lr"] == 0.0005
    assert updated["eval"]["batch_size"] == 16
    assert updated["model"]["name"] == "unet"


def test_set_overrides_invalid_format():
    cfg = {"a": 1}
    with pytest.raises(ValueError, match="Override must look like section.key=value"):
        set_overrides(cfg, ["invalid_override_without_equals"])


def test_resolve_path():
    rel = Path("configs/default.yaml")
    resolved = resolve_path(rel)
    assert resolved == PROJECT_ROOT / "configs/default.yaml"

    abs_path = Path("/tmp/some_file.txt").resolve()
    assert resolve_path(abs_path) == abs_path


def test_seed_everything():
    seed_everything(42)
    val_np_1 = np.random.rand()
    val_torch_1 = torch.rand(1).item()

    seed_everything(42)
    val_np_2 = np.random.rand()
    val_torch_2 = torch.rand(1).item()

    assert val_np_1 == val_np_2
    assert val_torch_1 == val_torch_2


def test_get_device():
    cpu_device = get_device("cpu")
    assert cpu_device.type == "cpu"

    auto_device = get_device("auto")
    assert auto_device.type in ("cuda", "cpu")


def test_save_json(tmp_path: Path):
    target_file = tmp_path / "subfolder" / "test.json"
    data = {"key": "value", "count": 42}
    save_json(data, target_file)

    assert target_file.is_file()
    loaded = json.loads(target_file.read_text())
    assert loaded == data
