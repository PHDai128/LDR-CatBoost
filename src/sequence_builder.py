"""Historical sequence input for the neural residual baselines."""
from __future__ import annotations
import numpy as np
import pandas as pd


def build_samples(
    process: pd.DataFrame,
    target: pd.Series,
    window: int,
    horizon: int,
    si_delay_hours: int = 0,
    include_si_history: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    process_values = process.to_numpy(dtype=np.float32)
    y = target.to_numpy(dtype=np.float32)
    observed = np.isfinite(y)
    if si_delay_hours < 0:
        raise ValueError("si_delay_hours must be nonnegative")
    # At feature time t, only Si observations released by t-delay are allowed.
    # Leading missing values stay missing so the first future observation can
    # never be copied backward into an earlier feature window.
    si_history = pd.Series(y).shift(si_delay_hours).ffill().to_numpy(dtype=np.float32)
    scales = [length for length in [2, 4, 8, 12, 24] if length <= window]

    cat_x, seq_x, targets, current, rows = [], [], [], [], []
    for end in range(window - 1, len(y) - horizon):
        target_idx = end + horizon
        if not observed[target_idx]:
            continue
        if include_si_history and not np.isfinite(
            si_history[end - window + 1 : end + 1]
        ).all():
            continue
        process_window = process_values[end - window + 1 : end + 1]
        if include_si_history:
            si_window = si_history[end - window + 1 : end + 1, None]
            sequence = np.concatenate([process_window, si_window], axis=1)
        else:
            sequence = process_window

        features = [sequence[-1]]
        for lag in [2, 3, 4, 6, 8, 12, 24]:
            if lag <= window:
                features.append(sequence[-lag])
        for length in scales:
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
        cat_x.append(np.concatenate(features))
        seq_x.append(sequence)
        targets.append(y[target_idx])
        current.append(si_history[end] if np.isfinite(si_history[end]) else np.nan)
        rows.append(target_idx)
    return (
        np.asarray(cat_x, dtype=np.float32),
        np.asarray(seq_x, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        np.asarray(current, dtype=np.float32),
        np.asarray(rows, dtype=np.int64),
    )
