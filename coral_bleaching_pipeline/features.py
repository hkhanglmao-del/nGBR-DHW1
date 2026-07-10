from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


CRW_BASE_FEATURES = [
    "CRW_SST",
    "CRW_SSTANOMALY",
    "CRW_HOTSPOT",
    "CRW_DHW",
    "CRW_BAA",
    "CRW_BAA_7D_MAX",
]

ALERT_LABELS = {
    0: "No Stress",
    1: "Watch",
    2: "Warning",
    3: "Alert Level 1",
    4: "Alert Level 2",
    5: "Alert Level 3",
    6: "Alert Level 4",
    7: "Alert Level 5",
}


@dataclass(frozen=True)
class DatasetSplit:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def compute_alert_level(
    dhw: pd.Series | np.ndarray,
    hotspot: pd.Series | np.ndarray,
    scheme: str = "noaa_extended",
) -> np.ndarray:
    """Compute bleaching alert classes from HotSpot and DHW.

    The repository's CRW file contains BAA values 0-4. The extended scheme keeps
    the same first five classes and adds Alert Levels 3-5 for extreme DHW.
    """
    dhw_arr = np.asarray(dhw, dtype=float)
    hotspot_arr = np.asarray(hotspot, dtype=float)
    alert = np.zeros_like(dhw_arr, dtype=np.int64)

    alert[(hotspot_arr > 0.0) & (hotspot_arr < 1.0)] = 1
    alert[(hotspot_arr >= 1.0) & (dhw_arr < 4.0)] = 2
    alert[(hotspot_arr >= 1.0) & (dhw_arr >= 4.0) & (dhw_arr < 8.0)] = 3
    alert[(hotspot_arr >= 1.0) & (dhw_arr >= 8.0) & (dhw_arr < 12.0)] = 4

    if scheme == "noaa_extended":
        alert[(hotspot_arr >= 1.0) & (dhw_arr >= 12.0) & (dhw_arr < 16.0)] = 5
        alert[(hotspot_arr >= 1.0) & (dhw_arr >= 16.0) & (dhw_arr < 20.0)] = 6
        alert[(hotspot_arr >= 1.0) & (dhw_arr >= 20.0)] = 7
    elif scheme == "legacy_baa":
        alert[(hotspot_arr >= 1.0) & (dhw_arr >= 12.0)] = 4
    else:
        raise ValueError(f"Unknown alert scheme: {scheme}")

    return alert


def add_monitoring_targets(df: pd.DataFrame, alert_scheme: str) -> pd.DataFrame:
    out = df.copy()
    out["alert_level"] = compute_alert_level(
        out["CRW_DHW"],
        out["CRW_HOTSPOT"],
        scheme=alert_scheme,
    )
    out["alert_label"] = out["alert_level"].map(ALERT_LABELS)
    return out


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    time = pd.to_datetime(out["time"])
    day_of_year = time.dt.dayofyear.astype(float)
    days_in_year = np.where(time.dt.is_leap_year, 366.0, 365.0)
    out["year"] = time.dt.year
    out["month"] = time.dt.month
    out["day_of_year"] = day_of_year
    out["doy_sin"] = np.sin(2.0 * math.pi * day_of_year / days_in_year)
    out["doy_cos"] = np.cos(2.0 * math.pi * day_of_year / days_in_year)
    out["month_sin"] = np.sin(2.0 * math.pi * out["month"] / 12.0)
    out["month_cos"] = np.cos(2.0 * math.pi * out["month"] / 12.0)
    return out


def add_ocean_current_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if {"current_u", "current_v"}.issubset(out.columns):
        out["current_speed"] = np.sqrt(out["current_u"] ** 2 + out["current_v"] ** 2)
        out["current_direction_sin"] = np.sin(np.arctan2(out["current_v"], out["current_u"]))
        out["current_direction_cos"] = np.cos(np.arctan2(out["current_v"], out["current_u"]))
    return out


def add_lag_features(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    lags: Iterable[int],
    group_col: str = "node_id",
) -> pd.DataFrame:
    out = df.sort_values([group_col, "time"]).copy()
    for col in feature_cols:
        if col not in out.columns:
            continue
        grouped = out.groupby(group_col, sort=False)[col]
        for lag in lags:
            out[f"{col}_lag{lag}"] = grouped.shift(lag)
    return out


def add_rolling_features(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    windows: Iterable[int],
    group_col: str = "node_id",
) -> pd.DataFrame:
    out = df.sort_values([group_col, "time"]).copy()
    for col in feature_cols:
        if col not in out.columns:
            continue
        for window in windows:
            grouped = out.groupby(group_col, sort=False)[col]
            shifted = grouped.shift(1)
            out[f"{col}_roll{window}_mean"] = (
                shifted.groupby(out[group_col], sort=False)
                .rolling(window, min_periods=max(2, window // 3))
                .mean()
                .reset_index(level=0, drop=True)
            )
            out[f"{col}_roll{window}_max"] = (
                shifted.groupby(out[group_col], sort=False)
                .rolling(window, min_periods=max(2, window // 3))
                .max()
                .reset_index(level=0, drop=True)
            )
    return out


def add_forecast_targets(
    df: pd.DataFrame,
    horizon_days: int,
    regression_col: str,
    classification_col: str,
    group_col: str = "node_id",
) -> pd.DataFrame:
    out = df.sort_values([group_col, "time"]).copy()
    grouped = out.groupby(group_col, sort=False)
    out[f"target_{regression_col}_h{horizon_days}"] = grouped[regression_col].shift(-horizon_days)
    out[f"target_{classification_col}_h{horizon_days}"] = grouped[classification_col].shift(-horizon_days)
    out[f"target_time_h{horizon_days}"] = out["time"] + pd.to_timedelta(horizon_days, unit="D")
    return out


def temporal_split(df: pd.DataFrame, train_end: str, val_end: str, time_col: str = "time") -> DatasetSplit:
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col])
    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)
    train = out[out[time_col] <= train_end_ts].copy()
    val = out[(out[time_col] > train_end_ts) & (out[time_col] <= val_end_ts)].copy()
    test = out[out[time_col] > val_end_ts].copy()
    return DatasetSplit(train=train, val=val, test=test)


def infer_feature_columns(
    df: pd.DataFrame,
    target_cols: Iterable[str],
    excluded: Iterable[str] | None = None,
) -> list[str]:
    excluded_set = {
        "time",
        "node_id",
        "latitude",
        "longitude",
        "alert_label",
        *list(target_cols),
        *(list(excluded) if excluded else []),
    }
    feature_cols = []
    for col in df.columns:
        if col in excluded_set or col.startswith("target_"):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)
    return feature_cols


def drop_supervised_na(df: pd.DataFrame, feature_cols: list[str], target_cols: list[str]) -> pd.DataFrame:
    needed = feature_cols + target_cols
    return df.dropna(subset=needed).reset_index(drop=True)


def build_node_adjacency(nodes: pd.DataFrame, sigma_km: float = 12.0) -> np.ndarray:
    """Build a distance-weighted adjacency matrix for reef grid cells."""
    coords = nodes[["latitude", "longitude"]].to_numpy(dtype=float)
    n_nodes = len(coords)
    adjacency = np.eye(n_nodes, dtype=np.float32)
    if n_nodes == 1:
        return adjacency

    lat_rad = np.deg2rad(coords[:, 0])
    lon_rad = np.deg2rad(coords[:, 1])
    earth_radius_km = 6371.0
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j:
                continue
            dlat = lat_rad[j] - lat_rad[i]
            dlon = lon_rad[j] - lon_rad[i]
            a = (
                np.sin(dlat / 2.0) ** 2
                + np.cos(lat_rad[i]) * np.cos(lat_rad[j]) * np.sin(dlon / 2.0) ** 2
            )
            distance = 2.0 * earth_radius_km * np.arcsin(np.sqrt(a))
            adjacency[i, j] = np.exp(-distance / sigma_km)

    degree = adjacency.sum(axis=1, keepdims=True)
    return adjacency / np.clip(degree, a_min=1e-6, a_max=None)
