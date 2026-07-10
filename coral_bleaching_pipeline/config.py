from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path("configs/default.yaml")


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge nested dictionaries without mutating the original base."""
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML config and merge it with configs/default.yaml."""
    default_path = DEFAULT_CONFIG_PATH
    with default_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if path is not None:
        path = Path(path)
        if path.resolve() != default_path.resolve():
            with path.open("r", encoding="utf-8") as f:
                override = yaml.safe_load(f) or {}
            cfg = _deep_update(cfg, override)

    return cfg


def apply_fast_dev_run(cfg: dict[str, Any]) -> dict[str, Any]:
    """Shrink training for a smoke test while preserving the pipeline shape."""
    cfg = deepcopy(cfg)
    cfg["training"]["epochs"] = 2
    cfg["training"]["patience"] = 1
    cfg["training"]["batch_size"] = min(int(cfg["training"]["batch_size"]), 64)
    cfg["models"]["random_forest"]["n_estimators"] = 50
    cfg["models"]["xgboost"]["n_estimators"] = 80
    cfg["models"]["lightgbm"]["n_estimators"] = 80
    cfg["models"]["deep"]["hidden_size"] = 32
    cfg["models"]["deep"]["graph_hidden_size"] = 32
    cfg["models"]["deep"]["cnn_channels"] = 32
    return cfg


def ensure_dirs(cfg: dict[str, Any]) -> None:
    """Create configured output directories."""
    for path in [
        cfg["data"]["processed_dir"],
        cfg["training"]["output_dir"],
        cfg["eda"]["output_dir"],
        cfg["evaluation"]["output_dir"],
    ]:
        Path(path).mkdir(parents=True, exist_ok=True)


def select_models(cfg: dict[str, Any], names: str | None) -> dict[str, Any]:
    """Override enabled models from a comma-separated CLI value."""
    if not names:
        return cfg
    cfg = deepcopy(cfg)
    cfg["models"]["enabled"] = [name.strip() for name in names.split(",") if name.strip()]
    return cfg

