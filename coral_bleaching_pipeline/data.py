from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import netcdf_file

from .features import (
    CRW_BASE_FEATURES,
    add_calendar_features,
    add_forecast_targets,
    add_lag_features,
    add_monitoring_targets,
    add_ocean_current_features,
    add_rolling_features,
    build_node_adjacency,
    drop_supervised_na,
    infer_feature_columns,
)


@dataclass
class PreparedData:
    node_frame: pd.DataFrame
    aggregate_frame: pd.DataFrame
    tabular_frame: pd.DataFrame
    tabular_features: list[str]
    sequence_features: list[str]
    target_regression: str
    target_classification: str
    nodes: pd.DataFrame
    adjacency: np.ndarray


def _decode_time_seconds(values: np.ndarray) -> pd.Series:
    return pd.to_datetime(values.astype(float), unit="s", utc=True).tz_convert(None)


def _fill_to_nan(values: np.ndarray, fill_value: Any) -> np.ndarray:
    arr = np.asarray(values).astype(float)
    if fill_value is not None:
        try:
            arr = np.where(arr == float(fill_value), np.nan, arr)
        except (TypeError, ValueError):
            pass
    arr = np.where(arr <= -300.0, np.nan, arr)
    return arr


def _read_crw_with_scipy(path: Path) -> pd.DataFrame:
    with netcdf_file(path, "r", mmap=False) as ds:
        time = _decode_time_seconds(ds.variables["time"].data.copy())
        lat = ds.variables["latitude"].data.copy().astype(float)
        lon = ds.variables["longitude"].data.copy().astype(float)
        records: dict[str, np.ndarray] = {}

        for name in CRW_BASE_FEATURES:
            variable = ds.variables[name]
            records[name] = _fill_to_nan(
                variable.data.copy(),
                getattr(variable, "_FillValue", None),
            )

    rows = []
    node_id = 0
    for lat_idx, latitude in enumerate(lat):
        for lon_idx, longitude in enumerate(lon):
            frame = pd.DataFrame({"time": time})
            frame["latitude"] = float(latitude)
            frame["longitude"] = float(longitude)
            frame["node_id"] = node_id
            for name, values in records.items():
                frame[name] = values[:, lat_idx, lon_idx]
            rows.append(frame)
            node_id += 1

    return pd.concat(rows, ignore_index=True)


def _read_crw_with_xarray(path: Path) -> pd.DataFrame:
    try:
        import xarray as xr
    except ImportError as exc:
        raise RuntimeError("xarray is required for NetCDF4 CRW reading") from exc

    ds = xr.open_dataset(path, decode_times=True)
    try:
        keep = [name for name in CRW_BASE_FEATURES if name in ds.data_vars]
        df = ds[keep].to_dataframe().reset_index()
    finally:
        ds.close()

    df = df.rename(columns={"latitude": "latitude", "longitude": "longitude"})
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
    nodes = (
        df[["latitude", "longitude"]]
        .drop_duplicates()
        .sort_values(["latitude", "longitude"], ascending=[False, True])
        .reset_index(drop=True)
    )
    nodes["node_id"] = np.arange(len(nodes))
    return df.merge(nodes, on=["latitude", "longitude"], how="left")


def read_crw_netcdf(path: str | Path) -> pd.DataFrame:
    """Read NOAA CRW NetCDF into node-level long format.

    The current repository file is NetCDF3 and can be read with SciPy. If a
    future NetCDF4 file is provided, xarray/netCDF4 will be used instead.
    """
    path = Path(path)
    try:
        return _read_crw_with_scipy(path)
    except TypeError:
        return _read_crw_with_xarray(path)


def read_external_csvs(external_dir: str | Path | None) -> list[pd.DataFrame]:
    if external_dir is None:
        return []
    directory = Path(external_dir)
    if not directory.exists():
        return []

    frames: list[pd.DataFrame] = []
    for path in sorted(directory.glob("*.csv")):
        frame = pd.read_csv(path)
        if "date" in frame.columns and "time" not in frame.columns:
            frame = frame.rename(columns={"date": "time"})
        if "time" not in frame.columns:
            continue
        frame["time"] = pd.to_datetime(frame["time"]).dt.tz_localize(None)
        frames.append(frame)
    return frames


def read_enso_csv(path: str | Path | None) -> pd.DataFrame | None:
    if not path:
        return None
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if "date" in frame.columns and "time" not in frame.columns:
        frame = frame.rename(columns={"date": "time"})
    if "time" not in frame.columns:
        raise ValueError("ENSO CSV must include a 'time' or 'date' column.")
    frame["time"] = pd.to_datetime(frame["time"]).dt.tz_localize(None)

    if "enso_index" not in frame.columns:
        numeric = [c for c in frame.columns if c != "time" and pd.api.types.is_numeric_dtype(frame[c])]
        if not numeric:
            raise ValueError("ENSO CSV must include an 'enso_index' column or another numeric index column.")
        frame = frame.rename(columns={numeric[0]: "enso_index"})

    daily = (
        frame[["time", "enso_index"]]
        .dropna()
        .drop_duplicates("time")
        .sort_values("time")
        .set_index("time")
        .resample("D")
        .ffill()
        .reset_index()
    )
    return daily


def merge_external_features(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    external_frames = read_external_csvs(cfg["data"].get("external_dir"))
    enso = read_enso_csv(cfg["data"].get("enso_csv"))
    if enso is not None:
        external_frames.append(enso)

    for frame in external_frames:
        cols = [c for c in frame.columns if c != "time"]
        if not cols:
            continue
        if "node_id" in frame.columns:
            out = out.merge(frame, on=["time", "node_id"], how="left")
        else:
            # Treat time-only tables as reef-wide forcing and broadcast to nodes.
            out = out.merge(frame, on="time", how="left")

    return out


def make_spatial_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        col
        for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col]) and col not in {"node_id", "latitude", "longitude"}
    ]
    agg = df.groupby("time", as_index=False)[numeric_cols].mean(numeric_only=True)
    agg["latitude"] = df["latitude"].mean()
    agg["longitude"] = df["longitude"].mean()
    agg["node_id"] = 0
    if "alert_level" in agg.columns:
        agg["alert_level"] = np.rint(agg["alert_level"]).astype(int)
    return agg


def prepare_feature_frames(cfg: dict[str, Any]) -> PreparedData:
    horizon = int(cfg["targets"]["forecast_horizon_days"])
    target_regression = f"target_{cfg['targets']['regression']}_h{horizon}"
    target_classification = f"target_{cfg['targets']['classification']}_h{horizon}"

    node_frame = read_crw_netcdf(cfg["data"]["crw_netcdf"])
    node_frame = merge_external_features(node_frame, cfg)
    node_frame = add_monitoring_targets(node_frame, cfg["data"].get("alert_scheme", "noaa_extended"))
    node_frame = add_ocean_current_features(node_frame)
    if cfg["features"].get("add_calendar", True):
        node_frame = add_calendar_features(node_frame)
    node_frame = add_forecast_targets(
        node_frame,
        horizon_days=horizon,
        regression_col=cfg["targets"]["regression"],
        classification_col=cfg["targets"]["classification"],
    )

    aggregate_frame = make_spatial_aggregate(node_frame).drop(columns=["alert_level"], errors="ignore")
    aggregate_frame = add_monitoring_targets(aggregate_frame, cfg["data"].get("alert_scheme", "noaa_extended"))
    aggregate_frame = add_forecast_targets(
        aggregate_frame.drop(columns=[target_regression, target_classification], errors="ignore"),
        horizon_days=horizon,
        regression_col=cfg["targets"]["regression"],
        classification_col=cfg["targets"]["classification"],
    )

    base_features = [col for col in cfg["features"]["base"] if col in node_frame.columns]
    optional_features = [col for col in cfg["features"].get("optional", []) if col in node_frame.columns]
    sequence_features = infer_feature_columns(
        node_frame,
        target_cols=[target_regression, target_classification],
        excluded=["year", "month", "day_of_year"],
    )

    tabular_frame = add_lag_features(
        node_frame,
        feature_cols=base_features + optional_features,
        lags=cfg["features"].get("lags_days", []),
    )
    tabular_frame = add_rolling_features(
        tabular_frame,
        feature_cols=base_features + optional_features,
        windows=cfg["features"].get("rolling_windows_days", []),
    )
    tabular_features = infer_feature_columns(
        tabular_frame,
        target_cols=[target_regression, target_classification],
        excluded=["year", "month", "day_of_year"],
    )
    tabular_frame = drop_supervised_na(
        tabular_frame,
        feature_cols=tabular_features,
        target_cols=[target_regression, target_classification],
    )

    nodes = (
        node_frame[["node_id", "latitude", "longitude"]]
        .drop_duplicates()
        .sort_values("node_id")
        .reset_index(drop=True)
    )
    adjacency = build_node_adjacency(nodes)

    return PreparedData(
        node_frame=node_frame,
        aggregate_frame=aggregate_frame,
        tabular_frame=tabular_frame,
        tabular_features=tabular_features,
        sequence_features=sequence_features,
        target_regression=target_regression,
        target_classification=target_classification,
        nodes=nodes,
        adjacency=adjacency,
    )


def save_prepared_frames(prepared: PreparedData, processed_dir: str | Path) -> None:
    directory = Path(processed_dir)
    directory.mkdir(parents=True, exist_ok=True)
    prepared.node_frame.to_csv(directory / "node_features.csv", index=False)
    prepared.aggregate_frame.to_csv(directory / "aggregate_features.csv", index=False)
    prepared.tabular_frame.to_csv(directory / "tabular_supervised.csv", index=False)
    prepared.nodes.to_csv(directory / "nodes.csv", index=False)
    np.save(directory / "adjacency.npy", prepared.adjacency)


def make_sequence_arrays(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_regression: str,
    target_classification: str,
    sequence_length: int,
    target_time_col: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    frame = df.sort_values("time").reset_index(drop=True)
    values = frame[feature_cols].to_numpy(dtype=np.float32)
    y_reg = frame[target_regression].to_numpy(dtype=np.float32)
    y_cls = frame[target_classification].to_numpy(dtype=np.int64)

    xs, yr, yc, rows = [], [], [], []
    for end_idx in range(sequence_length - 1, len(frame)):
        if np.isnan(y_reg[end_idx]) or pd.isna(y_cls[end_idx]):
            continue
        window = values[end_idx - sequence_length + 1 : end_idx + 1]
        if np.isnan(window).any():
            continue
        xs.append(window)
        yr.append(y_reg[end_idx])
        yc.append(int(y_cls[end_idx]))
        meta_cols = ["time", "node_id", "latitude", "longitude"]
        if target_time_col and target_time_col in frame.columns:
            meta_cols.append(target_time_col)
        rows.append(frame.loc[end_idx, meta_cols])

    if not xs:
        raise ValueError("No valid sequence samples after filtering NaNs.")
    meta = pd.DataFrame(rows).reset_index(drop=True)
    return np.stack(xs), np.asarray(yr, dtype=np.float32), np.asarray(yc, dtype=np.int64), meta


def make_spatiotemporal_arrays(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_regression: str,
    target_classification: str,
    sequence_length: int,
    nodes: pd.DataFrame,
    target_time_col: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    frame = df.sort_values(["time", "node_id"]).copy()
    complete_times = (
        frame.groupby("time")["node_id"]
        .nunique()
        .loc[lambda s: s == len(nodes)]
        .index
    )
    frame = frame[frame["time"].isin(complete_times)]
    pivot_features = {
        col: frame.pivot(index="time", columns="node_id", values=col).sort_index()
        for col in feature_cols
    }
    target_reg = frame.pivot(index="time", columns="node_id", values=target_regression).sort_index()
    target_cls = frame.pivot(index="time", columns="node_id", values=target_classification).sort_index()
    target_time = None
    if target_time_col and target_time_col in frame.columns:
        target_time = frame.pivot(index="time", columns="node_id", values=target_time_col).sort_index()
    times = target_reg.index

    feature_tensor = np.stack(
        [pivot_features[col].to_numpy(dtype=np.float32) for col in feature_cols],
        axis=-1,
    )
    y_reg_values = target_reg.to_numpy(dtype=np.float32)
    y_cls_values = target_cls.to_numpy(dtype=np.float32)

    xs, yr, yc, rows = [], [], [], []
    for end_idx in range(sequence_length - 1, len(times)):
        y_reg = y_reg_values[end_idx]
        y_cls = y_cls_values[end_idx]
        window = feature_tensor[end_idx - sequence_length + 1 : end_idx + 1]
        if np.isnan(window).any() or np.isnan(y_reg).any() or np.isnan(y_cls).any():
            continue
        xs.append(window)
        yr.append(y_reg)
        yc.append(y_cls.astype(np.int64))
        row = {"time": times[end_idx]}
        if target_time is not None:
            row[target_time_col] = target_time.iloc[end_idx].dropna().iloc[0]
        rows.append(row)

    if not xs:
        raise ValueError("No valid spatio-temporal samples after filtering NaNs.")
    return (
        np.stack(xs),
        np.stack(yr).astype(np.float32),
        np.stack(yc).astype(np.int64),
        pd.DataFrame(rows),
    )
