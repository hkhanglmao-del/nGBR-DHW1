from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .config import ensure_dirs
from .data import (
    PreparedData,
    make_sequence_arrays,
    make_spatiotemporal_arrays,
    prepare_feature_frames,
    save_prepared_frames,
)
from .features import temporal_split
from .metrics import make_leaderboard, save_classification_artifacts, save_metrics
from .models.classical import train_classical_models
from .models.deep import (
    build_sequence_model,
    fit_deep_model,
    make_loader,
    resolve_device,
    set_torch_seed,
    standardize_sequence_splits,
)
from .visualize import plot_eda, plot_leaderboard, plot_training_histories


DEEP_MODELS = {"lstm", "gru", "cnn_lstm", "tft", "st_gnn"}
TABULAR_MODELS = {"random_forest", "xgboost", "lightgbm"}


def _split_indices(meta: pd.DataFrame, train_end: str, val_end: str, split_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = pd.to_datetime(meta[split_col])
    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)
    train_idx = np.where(times <= train_end_ts)[0]
    val_idx = np.where((times > train_end_ts) & (times <= val_end_ts))[0]
    test_idx = np.where(times > val_end_ts)[0]
    return train_idx, val_idx, test_idx


def _subset(x: np.ndarray, y_reg: np.ndarray, y_cls: np.ndarray, idx: np.ndarray):
    return x[idx], y_reg[idx], y_cls[idx]


def _num_classes(*arrays: np.ndarray) -> int:
    max_class = 0
    for arr in arrays:
        if arr.size:
            max_class = max(max_class, int(np.nanmax(arr)))
    return max_class + 1


def _train_sequence_family(
    prepared: PreparedData,
    cfg: dict[str, Any],
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    enabled = set(cfg["models"]["enabled"]) & {"lstm", "gru", "cnn_lstm", "tft"}
    if not enabled:
        return [], {}, {}

    horizon = int(cfg["targets"]["forecast_horizon_days"])
    target_time_col = f"target_time_h{horizon}"
    seq_len = int(cfg["targets"]["sequence_length_days"])
    frame = prepared.aggregate_frame.sort_values("time").copy()
    frame = frame.dropna(subset=[prepared.target_regression, prepared.target_classification])
    feature_cols = [col for col in prepared.sequence_features if col in frame.columns]

    x, y_reg, y_cls, meta = make_sequence_arrays(
        frame,
        feature_cols,
        prepared.target_regression,
        prepared.target_classification,
        seq_len,
        target_time_col=target_time_col,
    )
    train_idx, val_idx, test_idx = _split_indices(
        meta,
        cfg["split"]["train_end"],
        cfg["split"]["val_end"],
        split_col=target_time_col,
    )
    if min(len(train_idx), len(val_idx), len(test_idx)) == 0:
        raise ValueError("Sequence temporal split produced an empty train/val/test set.")

    train_x, train_y_reg, train_y_cls = _subset(x, y_reg, y_cls, train_idx)
    val_x, val_y_reg, val_y_cls = _subset(x, y_reg, y_cls, val_idx)
    test_x, test_y_reg, test_y_cls = _subset(x, y_reg, y_cls, test_idx)
    train_x, val_x, test_x, scaler = standardize_sequence_splits(train_x, val_x, test_x)
    Path(cfg["training"]["output_dir"]).mkdir(parents=True, exist_ok=True)
    # Persist scaler as a simple numpy bundle so inference can reuse it.
    np.savez(
        Path(cfg["training"]["output_dir"]) / "aggregate_sequence_scaler.npz",
        mean=scaler.mean_,
        scale=scaler.scale_,
        feature_cols=np.asarray(feature_cols),
    )

    train_loader = make_loader(train_x, train_y_reg, train_y_cls, cfg, shuffle=True)
    val_loader = make_loader(val_x, val_y_reg, val_y_cls, cfg, shuffle=False)
    test_loader = make_loader(test_x, test_y_reg, test_y_cls, cfg, shuffle=False)
    num_classes = _num_classes(train_y_cls, val_y_cls, test_y_cls)

    rows: list[dict[str, Any]] = []
    predictions: dict[str, pd.DataFrame] = {}
    histories: dict[str, pd.DataFrame] = {}
    eval_dir = Path(cfg["evaluation"]["output_dir"])
    test_meta = meta.iloc[test_idx].reset_index(drop=True)

    for name in sorted(enabled):
        model = build_sequence_model(
            name,
            input_size=train_x.shape[-1],
            sequence_length=seq_len,
            num_classes=num_classes,
            cfg=cfg,
        )
        metrics, pred, history = fit_deep_model(name, model, train_loader, val_loader, test_loader, cfg, device)
        pred = pd.concat([test_meta.reset_index(drop=True), pred], axis=1)
        rows.append(metrics)
        predictions[name] = pred
        histories[name] = history
        pred.to_csv(eval_dir / f"{name}_predictions.csv", index=False)
        history.to_csv(eval_dir / f"{name}_history.csv", index=False)
        save_classification_artifacts(pred["y_alert_true"], pred["y_alert_pred"], eval_dir, name)

    return rows, predictions, histories


def _train_spatiotemporal_gnn(
    prepared: PreparedData,
    cfg: dict[str, Any],
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    if "st_gnn" not in set(cfg["models"]["enabled"]):
        return [], {}, {}

    horizon = int(cfg["targets"]["forecast_horizon_days"])
    target_time_col = f"target_time_h{horizon}"
    seq_len = int(cfg["targets"]["sequence_length_days"])
    frame = prepared.node_frame.dropna(subset=[prepared.target_regression, prepared.target_classification]).copy()
    feature_cols = [col for col in prepared.sequence_features if col in frame.columns]

    x, y_reg, y_cls, meta = make_spatiotemporal_arrays(
        frame,
        feature_cols,
        prepared.target_regression,
        prepared.target_classification,
        seq_len,
        prepared.nodes,
        target_time_col=target_time_col,
    )
    train_idx, val_idx, test_idx = _split_indices(
        meta,
        cfg["split"]["train_end"],
        cfg["split"]["val_end"],
        split_col=target_time_col,
    )
    if min(len(train_idx), len(val_idx), len(test_idx)) == 0:
        raise ValueError("ST-GNN temporal split produced an empty train/val/test set.")

    train_x, train_y_reg, train_y_cls = _subset(x, y_reg, y_cls, train_idx)
    val_x, val_y_reg, val_y_cls = _subset(x, y_reg, y_cls, val_idx)
    test_x, test_y_reg, test_y_cls = _subset(x, y_reg, y_cls, test_idx)
    train_x, val_x, test_x, scaler = standardize_sequence_splits(train_x, val_x, test_x)
    np.savez(
        Path(cfg["training"]["output_dir"]) / "st_gnn_scaler.npz",
        mean=scaler.mean_,
        scale=scaler.scale_,
        feature_cols=np.asarray(feature_cols),
    )

    train_loader = make_loader(train_x, train_y_reg, train_y_cls, cfg, shuffle=True)
    val_loader = make_loader(val_x, val_y_reg, val_y_cls, cfg, shuffle=False)
    test_loader = make_loader(test_x, test_y_reg, test_y_cls, cfg, shuffle=False)
    num_classes = _num_classes(train_y_cls, val_y_cls, test_y_cls)

    model = build_sequence_model(
        "st_gnn",
        input_size=train_x.shape[-1],
        sequence_length=seq_len,
        num_classes=num_classes,
        cfg=cfg,
        adjacency=prepared.adjacency,
    )
    metrics, pred, history = fit_deep_model("st_gnn", model, train_loader, val_loader, test_loader, cfg, device)

    test_meta = meta.iloc[test_idx].reset_index(drop=True)
    repeated_meta = test_meta.loc[test_meta.index.repeat(len(prepared.nodes))].reset_index(drop=True)
    tiled_nodes = pd.concat([prepared.nodes] * len(test_meta), ignore_index=True)
    pred = pd.concat([repeated_meta, tiled_nodes, pred], axis=1)

    eval_dir = Path(cfg["evaluation"]["output_dir"])
    pred.to_csv(eval_dir / "st_gnn_predictions.csv", index=False)
    history.to_csv(eval_dir / "st_gnn_history.csv", index=False)
    save_classification_artifacts(pred["y_alert_true"], pred["y_alert_pred"], eval_dir, "st_gnn")
    return [metrics], {"st_gnn": pred}, {"st_gnn": history}


def run_prepare(cfg: dict[str, Any]) -> PreparedData:
    ensure_dirs(cfg)
    prepared = prepare_feature_frames(cfg)
    save_prepared_frames(prepared, cfg["data"]["processed_dir"])
    return prepared


def run_eda(cfg: dict[str, Any], prepared: PreparedData | None = None) -> PreparedData:
    ensure_dirs(cfg)
    prepared = prepared or prepare_feature_frames(cfg)
    plot_eda(prepared, cfg)
    return prepared


def run_train(cfg: dict[str, Any], prepared: PreparedData | None = None) -> pd.DataFrame:
    ensure_dirs(cfg)
    set_torch_seed(int(cfg["project"]["seed"]))
    prepared = prepared or prepare_feature_frames(cfg)
    eval_dir = Path(cfg["evaluation"]["output_dir"])
    all_rows: list[dict[str, Any]] = []
    histories: dict[str, pd.DataFrame] = {}

    enabled = set(cfg["models"]["enabled"])
    if enabled & TABULAR_MODELS:
        horizon = int(cfg["targets"]["forecast_horizon_days"])
        split = temporal_split(
            prepared.tabular_frame,
            cfg["split"]["train_end"],
            cfg["split"]["val_end"],
            time_col=f"target_time_h{horizon}",
        )
        rows, _ = train_classical_models(
            split.train,
            split.val,
            split.test,
            prepared.tabular_features,
            prepared.target_regression,
            prepared.target_classification,
            cfg,
        )
        all_rows.extend(rows)

    device = resolve_device()
    if enabled & DEEP_MODELS:
        rows, _, seq_histories = _train_sequence_family(prepared, cfg, device)
        all_rows.extend(rows)
        histories.update(seq_histories)
        rows, _, st_histories = _train_spatiotemporal_gnn(prepared, cfg, device)
        all_rows.extend(rows)
        histories.update(st_histories)

    leaderboard = make_leaderboard(all_rows)
    leaderboard.to_csv(eval_dir / "leaderboard.csv", index=False)
    save_metrics({"leaderboard": leaderboard.to_dict(orient="records")}, eval_dir / "leaderboard.json")
    plot_leaderboard(leaderboard, cfg)
    plot_training_histories(histories, cfg)
    return leaderboard


def run_all(cfg: dict[str, Any]) -> pd.DataFrame:
    prepared = run_prepare(cfg)
    run_eda(cfg, prepared)
    return run_train(cfg, prepared)

