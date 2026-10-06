"""Hourly target loading, causal process preprocessing, and metrics."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

TARGET_PRIORITY = ["小时平均SI", "平均si", "铁水硅（小时平均插值法）"]
FURNACE_1_TARGET = "铁水硅（小时平均插值法）"
FORBIDDEN_SI_COLUMNS = ["预测硅", "铁水硅（跨整点线性插值法）"]


def infer_target(columns: list[str]) -> str:
    for target in TARGET_PRIORITY:
        if target in columns:
            return target
    raise ValueError("No supported hourly Si target column was found.")


def read_hourly(
    path: Path,
    required_target: str | None = None,
    forbidden_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, str]:
    raw = pd.read_csv(path, encoding="utf-8-sig")
    timestamp = pd.to_datetime(raw.iloc[:, 0], errors="coerce")
    raw = raw.loc[timestamp.notna()].copy()
    raw.insert(0, "timestamp", timestamp[timestamp.notna()].to_numpy())
    raw = raw.drop(columns=[raw.columns[1]])
    raw = raw.drop_duplicates("timestamp", keep="last").sort_values("timestamp")
    raw = raw.set_index("timestamp")
    raw = raw.reindex(pd.date_range(raw.index.min(), raw.index.max(), freq="h"))

    target_name = required_target or infer_target(list(raw.columns))
    if target_name not in raw.columns:
        raise ValueError(f"Required target column {target_name!r} was not found.")
    target = pd.to_numeric(raw[target_name], errors="coerce")
    # A physical guard only: values such as 54.0 are data-entry errors. Do not
    # sigma-filter the test target because that would make hit rate optimistic.
    target = target.where(target.between(0.0, 4.0))

    forbidden = forbidden_columns or []
    missing_forbidden = [column for column in forbidden if column not in raw.columns]
    if missing_forbidden:
        raise ValueError(f"Forbidden Si columns were not found: {missing_forbidden}")
    raw = raw.drop(columns=forbidden)
    si_like = [column for column in raw.columns if "si" in column.lower() or "硅" in column]
    process = raw.drop(columns=si_like, errors="ignore").apply(pd.to_numeric, errors="coerce")
    unexpected_si = [column for column in process.columns if "si" in column.lower() or "硅" in column]
    if unexpected_si:
        raise AssertionError(f"Unexpected Si columns entered process features: {unexpected_si}")
    process = process.replace([np.inf, -np.inf], np.nan)
    return process, target, target_name


def causal_fill_process(process: pd.DataFrame, train_end: int) -> pd.DataFrame:
    usable = process.columns[process.iloc[:train_end].notna().mean() >= 0.20]
    process = process.loc[:, usable].copy()
    train = process.iloc[:train_end]
    lower = train.quantile(0.001)
    upper = train.quantile(0.999)
    process = process.clip(lower=lower, upper=upper, axis=1)
    process = process.ffill()
    process = process.fillna(train.median()).fillna(0.0)
    process = process.loc[:, process.iloc[:train_end].nunique() > 1]
    return process.astype(np.float32)


def metric_row(y: np.ndarray, pred: np.ndarray, threshold: float) -> dict[str, float]:
    error = np.abs(y - pred)
    return {
        "mae": float(error.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(y - pred)))),
        "hr_0.05": float(np.mean(error <= 0.05)),
        "hr_0.10": float(np.mean(error <= 0.10)),
        "hr_0.15": float(np.mean(error <= 0.15)),
        "primary_hit_rate": float(np.mean(error <= threshold)),
    }
