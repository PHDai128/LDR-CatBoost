"""Furnace loading, rolling splits, and validation-only residual scaling."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from data_preprocessing import FURNACE_1_TARGET, FORBIDDEN_SI_COLUMNS, read_hourly

ROLLING_RATIOS = [(0.50, 0.60, 0.70), (0.60, 0.70, 0.80),
                  (0.70, 0.80, 0.90)]


def read_furnace(path: Path) -> tuple[pd.DataFrame, pd.Series, str]:
    if path.stem.startswith("furnace_1"):
        return read_hourly(
            path,
            required_target=FURNACE_1_TARGET,
            forbidden_columns=FORBIDDEN_SI_COLUMNS,
        )
    return read_hourly(path)


def rolling_boundaries(
    target: pd.Series,
) -> list[tuple[int, int, int, int]]:
    observed = np.flatnonzero(target.notna().to_numpy())
    folds = []
    for fold, (train_ratio, val_ratio, test_ratio) in enumerate(ROLLING_RATIOS, 1):
        train_end = int(observed[int(len(observed) * train_ratio)])
        val_end = int(observed[int(len(observed) * val_ratio)])
        test_end = int(observed[int(len(observed) * test_ratio)])
        folds.append((fold, train_end, val_end, test_end))
    return folds


def optimize_adjustment(
    y: np.ndarray,
    raw_prediction: np.ndarray,
    persistence: np.ndarray,
    threshold: float,
    allow_bias: bool,
) -> tuple[float, float]:
    best = (-1.0, -np.inf, 0.0, 0.0)
    biases = np.linspace(-0.04, 0.04, 33) if allow_bias else [0.0]
    for alpha in np.linspace(0.0, 1.25, 51):
        for bias in biases:
            prediction = persistence + alpha * (raw_prediction - persistence) + bias
            error = np.abs(y - prediction)
            score = (
                float(np.mean(error <= threshold)),
                -float(error.mean()),
                float(alpha),
                float(bias),
            )
            if score > best:
                best = score
    return best[2], best[3]


def apply_adjustment(
    raw_prediction: np.ndarray,
    persistence: np.ndarray,
    alpha: float,
    bias: float,
) -> np.ndarray:
    return persistence + alpha * (raw_prediction - persistence) + bias
