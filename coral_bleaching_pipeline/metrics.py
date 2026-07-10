from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:  # pragma: no cover - old sklearn fallback
    root_mean_squared_error = None


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if root_mean_squared_error is None:
        rmse = float(mean_squared_error(y_true, y_pred) ** 0.5)
    else:
        rmse = float(root_mean_squared_error(y_true, y_pred))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)),
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        balanced = float(balanced_accuracy_score(y_true, y_pred))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": balanced,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)),
    }


def alert_event_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    positive_threshold: int = 3,
) -> dict[str, float]:
    true_event = np.asarray(y_true, dtype=int) >= positive_threshold
    pred_event = np.asarray(y_pred, dtype=int) >= positive_threshold
    tp = float(np.logical_and(true_event, pred_event).sum())
    fp = float(np.logical_and(~true_event, pred_event).sum())
    fn = float(np.logical_and(true_event, ~pred_event).sum())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "event_precision": precision,
        "event_recall": recall,
        "event_f1": f1,
    }


def combined_metrics(
    y_reg_true: np.ndarray,
    y_reg_pred: np.ndarray,
    y_cls_true: np.ndarray,
    y_cls_pred: np.ndarray,
    positive_threshold: int,
) -> dict[str, float]:
    metrics = regression_metrics(y_reg_true, y_reg_pred)
    metrics.update(classification_metrics(y_cls_true, y_cls_pred))
    metrics.update(alert_event_metrics(y_cls_true, y_cls_pred, positive_threshold))
    return metrics


def save_metrics(metrics: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def save_classification_artifacts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: str | Path,
    model_name: str,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = sorted(set(np.asarray(y_true, dtype=int).tolist()) | set(np.asarray(y_pred, dtype=int).tolist()))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(output_dir / f"{model_name}_confusion_matrix.csv")
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(output_dir / f"{model_name}_classification_report.csv")


def make_leaderboard(rows: list[dict[str, Any]]) -> pd.DataFrame:
    leaderboard = pd.DataFrame(rows)
    if leaderboard.empty:
        return leaderboard
    sort_cols = [col for col in ["rmse", "macro_f1"] if col in leaderboard.columns]
    ascending = [True if col == "rmse" else False for col in sort_cols]
    return leaderboard.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
