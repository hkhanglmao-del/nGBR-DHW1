from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from ..metrics import combined_metrics, save_classification_artifacts


class SequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, y_reg: np.ndarray, y_cls: np.ndarray):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y_reg = torch.as_tensor(y_reg, dtype=torch.float32)
        self.y_cls = torch.as_tensor(y_cls, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y_reg[idx], self.y_cls[idx]


class SpatialSequenceDataset(SequenceDataset):
    pass


class LSTMForecastNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float, num_classes: int):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.reg_head = nn.Sequential(nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Linear(hidden_size // 2, 1))
        self.cls_head = nn.Sequential(nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Linear(hidden_size // 2, num_classes))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output, _ = self.encoder(x)
        hidden = self.norm(output[:, -1])
        return self.reg_head(hidden).squeeze(-1), self.cls_head(hidden)


class GRUForecastNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float, num_classes: int):
        super().__init__()
        self.encoder = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.reg_head = nn.Sequential(nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Linear(hidden_size // 2, 1))
        self.cls_head = nn.Sequential(nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Linear(hidden_size // 2, num_classes))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output, _ = self.encoder(x)
        hidden = self.norm(output[:, -1])
        return self.reg_head(hidden).squeeze(-1), self.cls_head(hidden)


class CNNLSTMForecastNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        cnn_channels: int,
        num_classes: int,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, cnn_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.encoder = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.reg_head = nn.Linear(hidden_size, 1)
        self.cls_head = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        conv = self.conv(x.transpose(1, 2)).transpose(1, 2)
        output, _ = self.encoder(conv)
        hidden = self.norm(output[:, -1])
        return self.reg_head(hidden).squeeze(-1), self.cls_head(hidden)


class TemporalFusionTransformerLite(nn.Module):
    """A compact TFT-style attention model for long reef time windows.

    This is intentionally dependency-light. It uses feature projection,
    positional embeddings, gated residual blocks, and transformer attention.
    """

    def __init__(
        self,
        input_size: int,
        sequence_length: int,
        hidden_size: int,
        dropout: float,
        heads: int,
        layers: int,
        num_classes: int,
    ):
        super().__init__()
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.positional_embedding = nn.Parameter(torch.zeros(1, sequence_length, hidden_size))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.gate = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Sigmoid())
        self.norm = nn.LayerNorm(hidden_size)
        self.reg_head = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_size, 1))
        self.cls_head = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_size, num_classes))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        projected = self.input_projection(x)
        projected = projected + self.positional_embedding[:, : projected.shape[1]]
        attended = self.encoder(projected)
        pooled = attended[:, -1]
        hidden = self.norm(self.gate(pooled) * pooled)
        return self.reg_head(hidden).squeeze(-1), self.cls_head(hidden)


class SpatioTemporalGNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        adjacency: np.ndarray,
        hidden_size: int,
        dropout: float,
        num_classes: int,
    ):
        super().__init__()
        self.register_buffer("adjacency", torch.as_tensor(adjacency, dtype=torch.float32))
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.graph_projection = nn.Linear(hidden_size, hidden_size)
        self.encoder = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)
        self.reg_head = nn.Linear(hidden_size, 1)
        self.cls_head = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: batch, time, node, feature
        hidden = torch.relu(self.input_projection(x))
        propagated = torch.einsum("ij,btjf->btif", self.adjacency, hidden)
        hidden = torch.relu(self.graph_projection(propagated))
        batch_size, seq_len, n_nodes, n_features = hidden.shape
        node_sequences = hidden.permute(0, 2, 1, 3).reshape(batch_size * n_nodes, seq_len, n_features)
        encoded, _ = self.encoder(node_sequences)
        last = self.norm(self.dropout(encoded[:, -1]))
        reg = self.reg_head(last).reshape(batch_size, n_nodes)
        cls = self.cls_head(last).reshape(batch_size, n_nodes, -1)
        return reg, cls


@dataclass
class DeepRun:
    rows: list[dict[str, Any]]
    predictions: dict[str, pd.DataFrame]
    histories: dict[str, pd.DataFrame]


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_torch_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True


def _loss(
    reg_pred: torch.Tensor,
    cls_logits: torch.Tensor,
    y_reg: torch.Tensor,
    y_cls: torch.Tensor,
    cfg: dict[str, Any],
) -> tuple[torch.Tensor, float, float]:
    reg_loss_fn = nn.SmoothL1Loss()
    cls_loss_fn = nn.CrossEntropyLoss()
    reg_loss = reg_loss_fn(reg_pred, y_reg)
    cls_loss = cls_loss_fn(cls_logits.reshape(-1, cls_logits.shape[-1]), y_cls.reshape(-1))
    total = (
        float(cfg["training"]["regression_loss_weight"]) * reg_loss
        + float(cfg["training"]["classification_loss_weight"]) * cls_loss
    )
    return total, float(reg_loss.detach().cpu()), float(cls_loss.detach().cpu())


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    cfg: dict[str, Any],
    scaler: Any | None,
) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    total_reg = 0.0
    total_cls = 0.0
    count = 0
    autocast_enabled = bool(cfg["training"].get("amp", True)) and device.type == "cuda"

    for x, y_reg, y_cls in loader:
        x = x.to(device, non_blocking=True)
        y_reg = y_reg.to(device, non_blocking=True)
        y_cls = y_cls.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        if hasattr(torch, "amp"):
            autocast_context = torch.amp.autocast(device_type=device.type, enabled=autocast_enabled)
        else:  # pragma: no cover - old torch fallback
            autocast_context = torch.cuda.amp.autocast(enabled=autocast_enabled)

        with autocast_context:
            reg_pred, cls_logits = model(x)
            loss, reg_loss, cls_loss = _loss(reg_pred, cls_logits, y_reg, y_cls, cfg)

        if train:
            assert scaler is not None
            scaler.scale(loss).backward()
            if cfg["training"].get("gradient_clip"):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"]["gradient_clip"]))
            scaler.step(optimizer)
            scaler.update()

        batch = x.shape[0]
        total_loss += float(loss.detach().cpu()) * batch
        total_reg += reg_loss * batch
        total_cls += cls_loss * batch
        count += batch

    return {
        "loss": total_loss / max(count, 1),
        "reg_loss": total_reg / max(count, 1),
        "cls_loss": total_cls / max(count, 1),
    }


def _predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    reg_true, reg_pred, cls_true, cls_pred = [], [], [], []
    with torch.no_grad():
        for x, y_reg, y_cls in loader:
            x = x.to(device, non_blocking=True)
            pred_reg, logits = model(x)
            pred_cls = torch.argmax(logits, dim=-1)
            reg_true.append(y_reg.cpu().numpy().reshape(-1))
            reg_pred.append(pred_reg.cpu().numpy().reshape(-1))
            cls_true.append(y_cls.cpu().numpy().reshape(-1))
            cls_pred.append(pred_cls.cpu().numpy().reshape(-1))
    return (
        np.concatenate(reg_true),
        np.concatenate(reg_pred),
        np.concatenate(cls_true),
        np.concatenate(cls_pred),
    )


def fit_deep_model(
    name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    cfg: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    output_dir = Path(cfg["training"]["output_dir"]) / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    scaler_enabled = bool(cfg["training"].get("amp", True)) and device.type == "cuda"
    if hasattr(torch, "amp"):
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
        except TypeError:  # pragma: no cover - old torch.amp signature
            scaler = torch.amp.GradScaler(enabled=scaler_enabled)
    else:  # pragma: no cover - old torch fallback
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
    best_loss = float("inf")
    best_path = output_dir / f"{name}.pt"
    history_rows: list[dict[str, float]] = []
    bad_epochs = 0

    progress = tqdm(range(1, int(cfg["training"]["epochs"]) + 1), desc=f"train {name}", leave=False)
    for epoch in progress:
        train_stats = _run_epoch(model, train_loader, optimizer, device, cfg, scaler)
        val_stats = _run_epoch(model, val_loader, None, device, cfg, None)
        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"val_{k}": v for k, v in val_stats.items()},
        }
        history_rows.append(row)
        progress.set_postfix(train=f"{train_stats['loss']:.4f}", val=f"{val_stats['loss']:.4f}")

        if val_stats["loss"] < best_loss - float(cfg["training"]["min_delta"]):
            best_loss = val_stats["loss"]
            bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                    "model_name": name,
                    "best_val_loss": best_loss,
                },
                best_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= int(cfg["training"]["patience"]):
                break

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    y_reg_true, y_reg_pred, y_cls_true, y_cls_pred = _predict(model, test_loader, device)
    metrics = combined_metrics(
        y_reg_true,
        y_reg_pred,
        y_cls_true,
        y_cls_pred,
        positive_threshold=int(cfg["evaluation"]["alert_positive_threshold"]),
    )
    metrics = {"model": name, "family": "deep", **metrics, "best_val_loss": best_loss}

    predictions = pd.DataFrame(
        {
            "y_dhw_true": y_reg_true,
            "y_dhw_pred": y_reg_pred,
            "y_alert_true": y_cls_true,
            "y_alert_pred": y_cls_pred,
        }
    )
    history = pd.DataFrame(history_rows)
    return metrics, predictions, history


def standardize_sequence_splits(
    train_x: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    original_shape = train_x.shape
    scaler.fit(train_x.reshape(-1, train_x.shape[-1]))
    train_scaled = scaler.transform(train_x.reshape(-1, train_x.shape[-1])).reshape(original_shape)
    val_scaled = scaler.transform(val_x.reshape(-1, val_x.shape[-1])).reshape(val_x.shape)
    test_scaled = scaler.transform(test_x.reshape(-1, test_x.shape[-1])).reshape(test_x.shape)
    return train_scaled, val_scaled, test_scaled, scaler


def make_loader(x: np.ndarray, y_reg: np.ndarray, y_cls: np.ndarray, cfg: dict[str, Any], shuffle: bool) -> DataLoader:
    dataset = SequenceDataset(x, y_reg, y_cls)
    return DataLoader(
        dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=shuffle,
        num_workers=int(cfg["training"].get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )


def build_sequence_model(
    name: str,
    input_size: int,
    sequence_length: int,
    num_classes: int,
    cfg: dict[str, Any],
    adjacency: np.ndarray | None = None,
) -> nn.Module:
    deep_cfg = cfg["models"]["deep"]
    hidden = int(deep_cfg["hidden_size"])
    layers = int(deep_cfg["num_layers"])
    dropout = float(deep_cfg["dropout"])
    if name == "lstm":
        return LSTMForecastNet(input_size, hidden, layers, dropout, num_classes)
    if name == "gru":
        return GRUForecastNet(input_size, hidden, layers, dropout, num_classes)
    if name == "cnn_lstm":
        return CNNLSTMForecastNet(input_size, hidden, layers, dropout, int(deep_cfg["cnn_channels"]), num_classes)
    if name == "tft":
        return TemporalFusionTransformerLite(
            input_size=input_size,
            sequence_length=sequence_length,
            hidden_size=hidden,
            dropout=dropout,
            heads=int(deep_cfg["transformer_heads"]),
            layers=int(deep_cfg["transformer_layers"]),
            num_classes=num_classes,
        )
    if name == "st_gnn":
        if adjacency is None:
            raise ValueError("st_gnn requires an adjacency matrix")
        return SpatioTemporalGNN(
            input_size=input_size,
            adjacency=adjacency,
            hidden_size=int(deep_cfg["graph_hidden_size"]),
            dropout=dropout,
            num_classes=num_classes,
        )
    raise ValueError(f"Unknown deep model: {name}")
