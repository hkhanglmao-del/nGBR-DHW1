from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .data import PreparedData


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_eda(prepared: PreparedData, cfg: dict[str, Any]) -> None:
    out = Path(cfg["eda"]["output_dir"])
    node = prepared.node_frame.copy()
    aggregate = prepared.aggregate_frame.copy()
    node["time"] = pd.to_datetime(node["time"])
    aggregate["time"] = pd.to_datetime(aggregate["time"])

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(aggregate["time"], aggregate["CRW_DHW"], color="#b3202a", linewidth=1.2, label="DHW")
    ax.set_ylabel("Degree Heating Weeks")
    ax.set_xlabel("Time")
    ax.set_title("Spatially averaged NOAA CRW DHW")
    ax.grid(True, alpha=0.25)
    _save(fig, out / "dhw_timeseries.png")

    annual = aggregate.assign(year=aggregate["time"].dt.year).groupby("year", as_index=False)["CRW_DHW"].max()
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(data=annual, x="year", y="CRW_DHW", color="#d75f4b", ax=ax)
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylabel("Annual max DHW")
    ax.set_xlabel("")
    ax.set_title("Annual maximum heat stress")
    _save(fig, out / "annual_max_dhw.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    counts = node["alert_label"].value_counts(dropna=False)
    sns.barplot(x=counts.values, y=counts.index, color="#4677a8", ax=ax)
    ax.set_xlabel("Node-days")
    ax.set_ylabel("")
    ax.set_title("Bleaching alert class distribution")
    _save(fig, out / "alert_distribution.png")

    corr_cols = [c for c in prepared.sequence_features if c in aggregate.columns]
    corr = aggregate[corr_cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(max(8, len(corr_cols) * 0.45), max(6, len(corr_cols) * 0.35)))
    sns.heatmap(corr, cmap="vlag", center=0, linewidths=0.2, ax=ax)
    ax.set_title("Feature correlation")
    _save(fig, out / "feature_correlation.png")

    missing = node[prepared.sequence_features].isna().mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, max(4, len(missing) * 0.25)))
    sns.barplot(x=missing.values, y=missing.index, color="#6a8f4e", ax=ax)
    ax.set_xlabel("Missing fraction")
    ax.set_ylabel("")
    ax.set_title("Feature missingness")
    _save(fig, out / "missingness.png")

    fig, ax = plt.subplots(figsize=(5, 5))
    latest = node.sort_values("time").groupby("node_id", as_index=False).tail(1)
    scatter = ax.scatter(
        latest["longitude"],
        latest["latitude"],
        c=latest["CRW_DHW"],
        s=130,
        cmap="Reds",
        edgecolor="black",
    )
    fig.colorbar(scatter, ax=ax, label="Latest DHW")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("CRW grid nodes around Lizard Island")
    _save(fig, out / "node_map_latest_dhw.png")


def plot_leaderboard(leaderboard: pd.DataFrame, cfg: dict[str, Any]) -> None:
    if leaderboard.empty:
        return
    out = Path(cfg["evaluation"]["output_dir"])
    fig, ax = plt.subplots(figsize=(9, max(4, 0.45 * len(leaderboard))))
    sns.barplot(data=leaderboard, x="rmse", y="model", hue="family", dodge=False, ax=ax)
    ax.set_title("Model comparison by DHW RMSE")
    ax.set_xlabel("RMSE, lower is better")
    ax.set_ylabel("")
    _save(fig, out / "leaderboard_rmse.png")

    if "macro_f1" in leaderboard:
        fig, ax = plt.subplots(figsize=(9, max(4, 0.45 * len(leaderboard))))
        sns.barplot(data=leaderboard, x="macro_f1", y="model", hue="family", dodge=False, ax=ax)
        ax.set_title("Model comparison by alert macro F1")
        ax.set_xlabel("Macro F1, higher is better")
        ax.set_ylabel("")
        _save(fig, out / "leaderboard_macro_f1.png")


def plot_training_histories(histories: dict[str, pd.DataFrame], cfg: dict[str, Any]) -> None:
    out = Path(cfg["evaluation"]["output_dir"]) / "training_curves"
    for name, history in histories.items():
        if history.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(history["epoch"], history["train_loss"], label="train")
        ax.plot(history["epoch"], history["val_loss"], label="validation")
        ax.set_title(f"{name} training loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Multi-task loss")
        ax.grid(True, alpha=0.25)
        ax.legend()
        _save(fig, out / f"{name}_loss.png")
