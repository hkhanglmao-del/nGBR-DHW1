from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from ..metrics import combined_metrics, save_classification_artifacts


def _xy(frame: pd.DataFrame, feature_cols: list[str], target_reg: str, target_cls: str):
    x = frame[feature_cols].to_numpy(dtype=np.float32)
    y_reg = frame[target_reg].to_numpy(dtype=np.float32)
    y_cls = frame[target_cls].to_numpy(dtype=np.int64)
    return x, y_reg, y_cls


def _fit_xgboost(
    train_x: np.ndarray,
    train_y_reg: np.ndarray,
    train_y_cls: np.ndarray,
    val_x: np.ndarray,
    val_y_reg: np.ndarray,
    val_y_cls: np.ndarray,
    cfg: dict[str, Any],
):
    try:
        from xgboost import XGBClassifier, XGBRegressor
    except ImportError:
        print("xgboost is not installed; skipping xgboost.")
        return None

    params = dict(cfg["models"].get("xgboost", {}))
    cls_params = params.copy()
    cls_params["objective"] = "multi:softprob"
    cls_params["num_class"] = int(max(train_y_cls.max(), val_y_cls.max()) + 1)
    reg = XGBRegressor(
        objective="reg:squarederror",
        random_state=int(cfg["project"]["seed"]),
        **params,
    )
    cls = XGBClassifier(
        random_state=int(cfg["project"]["seed"]),
        eval_metric="mlogloss",
        **cls_params,
    )
    try:
        reg.fit(
            train_x,
            train_y_reg,
            eval_set=[(val_x, val_y_reg)],
            verbose=False,
            early_stopping_rounds=cfg["training"]["patience"],
        )
    except TypeError:
        reg.fit(train_x, train_y_reg)
    try:
        cls.fit(
            train_x,
            train_y_cls,
            eval_set=[(val_x, val_y_cls)],
            verbose=False,
            early_stopping_rounds=cfg["training"]["patience"],
        )
    except TypeError:
        cls.fit(train_x, train_y_cls)
    return reg, cls


def _fit_lightgbm(
    train_x: np.ndarray,
    train_y_reg: np.ndarray,
    train_y_cls: np.ndarray,
    val_x: np.ndarray,
    val_y_reg: np.ndarray,
    val_y_cls: np.ndarray,
    cfg: dict[str, Any],
):
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping, log_evaluation
    except ImportError:
        print("lightgbm is not installed; skipping lightgbm.")
        return None

    params = dict(cfg["models"].get("lightgbm", {}))
    callbacks = [
        early_stopping(stopping_rounds=int(cfg["training"]["patience"]), verbose=False),
        log_evaluation(period=0),
    ]
    reg = LGBMRegressor(
        random_state=int(cfg["project"]["seed"]),
        objective="regression",
        **params,
    )
    cls = LGBMClassifier(
        random_state=int(cfg["project"]["seed"]),
        objective="multiclass",
        class_weight="balanced",
        **params,
    )
    reg.fit(train_x, train_y_reg, eval_set=[(val_x, val_y_reg)], callbacks=callbacks)
    cls.fit(train_x, train_y_cls, eval_set=[(val_x, val_y_cls)], callbacks=callbacks)
    return reg, cls


def train_classical_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_reg: str,
    target_cls: str,
    cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, pd.DataFrame]]:
    output_dir = Path(cfg["training"]["output_dir"]) / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = Path(cfg["evaluation"]["output_dir"])

    train_x, train_y_reg, train_y_cls = _xy(train_df, feature_cols, target_reg, target_cls)
    val_x, val_y_reg, val_y_cls = _xy(val_df, feature_cols, target_reg, target_cls)
    test_x, test_y_reg, test_y_cls = _xy(test_df, feature_cols, target_reg, target_cls)

    imputer = SimpleImputer(strategy="median")
    train_x_imp = imputer.fit_transform(train_x)
    val_x_imp = imputer.transform(val_x)
    test_x_imp = imputer.transform(test_x)
    dump(imputer, output_dir / "tabular_imputer.joblib")

    enabled = set(cfg["models"]["enabled"])
    rows: list[dict[str, Any]] = []
    predictions: dict[str, pd.DataFrame] = {}

    fitted: dict[str, tuple[Any, Any]] = {}
    if "random_forest" in enabled:
        rf_cfg = cfg["models"].get("random_forest", {})
        reg = RandomForestRegressor(
            n_estimators=int(rf_cfg.get("n_estimators", 600)),
            max_depth=rf_cfg.get("max_depth"),
            min_samples_leaf=int(rf_cfg.get("min_samples_leaf", 2)),
            n_jobs=int(rf_cfg.get("n_jobs", -1)),
            random_state=int(cfg["project"]["seed"]),
        )
        cls = RandomForestClassifier(
            n_estimators=int(rf_cfg.get("n_estimators", 600)),
            max_depth=rf_cfg.get("max_depth"),
            min_samples_leaf=int(rf_cfg.get("min_samples_leaf", 2)),
            n_jobs=int(rf_cfg.get("n_jobs", -1)),
            class_weight="balanced_subsample",
            random_state=int(cfg["project"]["seed"]),
        )
        reg.fit(train_x_imp, train_y_reg)
        cls.fit(train_x_imp, train_y_cls)
        fitted["random_forest"] = (reg, cls)

    if "xgboost" in enabled:
        model = _fit_xgboost(train_x_imp, train_y_reg, train_y_cls, val_x_imp, val_y_reg, val_y_cls, cfg)
        if model is not None:
            fitted["xgboost"] = model

    if "lightgbm" in enabled:
        model = _fit_lightgbm(train_x_imp, train_y_reg, train_y_cls, val_x_imp, val_y_reg, val_y_cls, cfg)
        if model is not None:
            fitted["lightgbm"] = model

    for name, (reg, cls) in fitted.items():
        y_reg_pred = np.asarray(reg.predict(test_x_imp), dtype=float)
        y_cls_pred = np.asarray(cls.predict(test_x_imp), dtype=int)
        metrics = combined_metrics(
            test_y_reg,
            y_reg_pred,
            test_y_cls,
            y_cls_pred,
            positive_threshold=int(cfg["evaluation"]["alert_positive_threshold"]),
        )
        rows.append({"model": name, "family": "tabular", **metrics})

        pred = test_df[["time", "node_id", "latitude", "longitude"]].copy()
        pred["y_dhw_true"] = test_y_reg
        pred["y_dhw_pred"] = y_reg_pred
        pred["y_alert_true"] = test_y_cls
        pred["y_alert_pred"] = y_cls_pred
        predictions[name] = pred

        dump(reg, output_dir / f"{name}_regressor.joblib")
        dump(cls, output_dir / f"{name}_classifier.joblib")
        pred.to_csv(eval_dir / f"{name}_predictions.csv", index=False)
        save_classification_artifacts(test_y_cls, y_cls_pred, eval_dir, name)

    return rows, predictions
