"""DLinear, iTransformer, and PatchTST residual baseline training."""
from __future__ import annotations
import argparse
import random
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from bf_itransformer_model import BFiTransformerLite
from patchtst_model import PatchTSTRegressor


class DLinearResidual(nn.Module):
    def __init__(self, window: int, channels: int, kernel_size: int = 5) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.trend_linear = nn.Linear(window, 1)
        self.seasonal_linear = nn.Linear(window, 1)
        self.combine = nn.Sequential(
            nn.Linear(channels * 2, max(16, channels)),
            nn.GELU(),
            nn.Linear(max(16, channels), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padding = self.kernel_size // 2
        trend = nn.functional.avg_pool1d(
            x.transpose(1, 2),
            kernel_size=self.kernel_size,
            stride=1,
            padding=padding,
            count_include_pad=False,
        )
        seasonal = x.transpose(1, 2) - trend
        trend_out = self.trend_linear(trend).squeeze(-1)
        seasonal_out = self.seasonal_linear(seasonal).squeeze(-1)
        return self.combine(torch.cat([trend_out, seasonal_out], dim=1)).squeeze(-1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_deep_model(name: str, window: int, channels: int) -> nn.Module:
    if name == "PatchTST":
        return PatchTSTRegressor(
            channels=channels,
            window=window,
            patch_len=8,
            stride=4,
            d_model=48,
            n_heads=4,
            n_layers=2,
            d_ff=96,
            dropout=0.1,
        )
    if name == "iTransformer":
        return BFiTransformerLite(
            window=window,
            channels=channels,
            d_model=32,
            num_heads=4,
            ffn_hidden=64,
            num_layers=2,
            dropout=0.1,
            use_revin=True,
        )
    if name == "DLinear":
        return DLinearResidual(window, channels)
    raise ValueError(name)


def forward_model(name: str, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if name == "iTransformer":
        return model(x.unsqueeze(1))["si"]
    return model(x)


def train_deep(
    name: str,
    sequence: np.ndarray,
    y: np.ndarray,
    current: np.ndarray,
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    set_seed(seed)
    feature_mean = sequence[train].reshape(-1, sequence.shape[-1]).mean(axis=0)
    feature_std = sequence[train].reshape(-1, sequence.shape[-1]).std(axis=0)
    feature_std[feature_std < 1e-6] = 1.0
    x = ((sequence - feature_mean) / feature_std).astype(np.float32)
    delta = (y - current).astype(np.float32)
    delta_scale = float(np.std(delta[train]) or 1.0)
    delta_scaled = delta / delta_scale
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_deep_model(name, x.shape[1], x.shape[2]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss(beta=0.5)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x[train]),
            torch.from_numpy(delta_scaled[train]),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )

    def infer(mask: np.ndarray) -> np.ndarray:
        model.eval()
        outputs = []
        infer_loader = DataLoader(
            TensorDataset(torch.from_numpy(x[mask])),
            batch_size=args.batch_size,
            shuffle=False,
        )
        with torch.no_grad():
            for (batch,) in infer_loader:
                outputs.append(
                    forward_model(name, model, batch.to(device)).cpu().numpy()
                )
        return current[mask] + np.concatenate(outputs) * delta_scale

    best_state = None
    best_score = (-1.0, -np.inf)
    best_epoch = 0
    stale = 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            prediction = forward_model(name, model, batch_x)
            loss = loss_fn(prediction, batch_y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        val_prediction = infer(val)
        error = np.abs(y[val] - val_prediction)
        score = (float(np.mean(error <= 0.05)), -float(error.mean()))
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    train_seconds = time.perf_counter() - started
    if best_state is None:
        raise RuntimeError(f"{name} did not produce a checkpoint.")
    model.load_state_dict(best_state)
    return infer(val), infer(test), {
        "best_epoch": best_epoch,
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "train_seconds": train_seconds,
        "device": str(device),
    }
