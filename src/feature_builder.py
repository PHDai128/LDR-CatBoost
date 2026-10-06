"""Exact availability rule and multiscale feature builder used in the paper."""
from __future__ import annotations
import numpy as np
import pandas as pd

LAGS = [2, 3, 4, 6, 8, 12, 24]
SCALES = [2, 4, 8, 12, 24]


def flatten_features(sequence: np.ndarray) -> np.ndarray:
    features = [sequence[-1]]
    for lag in LAGS:
        if lag <= len(sequence):
            features.append(sequence[-lag])
    for length in SCALES:
        if length > len(sequence):
            continue
        segment = sequence[-length:]
        features.extend(
            [
                segment.mean(axis=0),
                segment.std(axis=0),
                segment.min(axis=0),
                segment.max(axis=0),
                segment[-1] - segment[0],
            ]
        )
    return np.concatenate(features).astype(np.float32)


def build_adr_samples(
    process: pd.DataFrame,
    target: pd.Series,
    window: int,
    delay: int,
) -> dict[str, np.ndarray]:
    process_values = process.to_numpy(dtype=np.float32)
    y = target.to_numpy(dtype=np.float32)
    observed = np.isfinite(y)
    si_available = pd.Series(y).shift(delay).ffill().to_numpy(dtype=np.float32)

    process_x, si_x = [], []
    targets, current, rows = [], [], []
    for end in range(window - 1, len(y) - 1):
        target_idx = end + 1
        if not observed[target_idx]:
            continue
        si_window = si_available[end - window + 1 : end + 1]
        if not np.isfinite(si_window).all():
            continue
        process_window = process_values[end - window + 1 : end + 1]
        process_features = flatten_features(process_window)
        si_features = flatten_features(si_window[:, None])
        process_x.append(process_features)
        si_x.append(si_features)
        targets.append(y[target_idx])
        current.append(si_window[-1])
        rows.append(target_idx)
    return {
        "process_x": np.asarray(process_x, dtype=np.float32),
        "si_x": np.asarray(si_x, dtype=np.float32),
        "y": np.asarray(targets, dtype=np.float32),
        "current": np.asarray(current, dtype=np.float32),
        "rows": np.asarray(rows, dtype=np.int64),
    }
