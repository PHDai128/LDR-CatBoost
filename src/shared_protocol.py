from __future__ import annotations

import random

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostRegressor

from data_preprocessing import causal_fill_process


CONTINUOUS_TEMPERATURE_COLUMNS = ["铁水连续测温"]


def prepare_process(process_raw: pd.DataFrame, train_end: int) -> pd.DataFrame:
    process = causal_fill_process(process_raw, train_end)
    process = process.drop(columns=CONTINUOUS_TEMPERATURE_COLUMNS, errors="ignore")
    remaining = [
        column
        for column in process.columns
        if "连续测温" in str(column)
    ]
    if remaining:
        raise AssertionError(
            f"Continuous hot-metal temperature columns remain: {remaining}"
        )
    return process


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cdr_model(seed: int, iterations: int) -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=iterations,
        depth=7,
        learning_rate=0.035,
        loss_function="Huber:delta=0.05",
        eval_metric="MAE",
        l2_leaf_reg=5.0,
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        task_type="GPU" if torch.cuda.is_available() else "CPU",
    )


def fit_cdr(
    x: np.ndarray,
    target: np.ndarray,
    train: np.ndarray,
    val: np.ndarray,
    seed: int,
    iterations: int,
) -> CatBoostRegressor:
    model = cdr_model(seed, iterations)
    model.fit(
        x[train],
        target[train],
        eval_set=(x[val], target[val]),
        early_stopping_rounds=80,
    )
    return model


def deep_seeds(first_seed: int, count: int) -> list[int]:
    if count < 1:
        raise ValueError("deep seed count must be positive")
    return list(range(first_seed, first_seed + count))
