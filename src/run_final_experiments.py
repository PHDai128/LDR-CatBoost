from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from feature_builder import build_adr_samples
from sequence_builder import build_samples
from data_preprocessing import metric_row
from data_splits import (
    apply_adjustment,
    optimize_adjustment,
    read_furnace,
    rolling_boundaries,
)
from neural_training import train_deep
from shared_protocol import deep_seeds, fit_cdr, prepare_process, set_seed


DEFAULT_FURNACES = ["furnace_1", "furnace_2", "furnace_3"]
DEEP_MODELS = ["PatchTST", "iTransformer", "DLinear"]
STANDARD_MODEL = "Ours: Standard CDR-CatBoost"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified final baseline experiments.")
    parser.add_argument("--data-dir", type=Path, default=Path("Dataset"))
    parser.add_argument("--furnaces", nargs="+", default=DEFAULT_FURNACES)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/final_baselines")
    )
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--delay", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--deep-seed-count", type=int, default=5)
    parser.add_argument("--cat-iterations", type=int, default=500)
    return parser.parse_args()


def persistence_segment(error: np.ndarray) -> np.ndarray:
    return np.select(
        [error <= 0.03, error <= 0.08],
        ["easy_persistence", "medium_correction"],
        default="hard_correction",
    )


def append_metrics(
    records: list[dict[str, object]],
    furnace: str,
    fold: int,
    model: str,
    random_seed: int,
    y: np.ndarray,
    prediction: np.ndarray,
    alpha: float,
    train_seconds: float,
    metadata: dict[str, object] | None = None,
) -> None:
    row = {
        "furnace": furnace,
        "fold": fold,
        "model": model,
        "random_seed": random_seed,
        "samples": len(y),
        "alpha": alpha,
        "train_seconds": train_seconds,
        **metric_row(y, prediction, 0.05),
    }
    if metadata:
        row.update(metadata)
    records.append(row)


def append_predictions(
    frames: list[pd.DataFrame],
    furnace: str,
    fold: int,
    model: str,
    random_seed: int,
    rows: np.ndarray,
    y: np.ndarray,
    current: np.ndarray,
    prediction: np.ndarray,
) -> None:
    persistence_error = np.abs(y - current)
    frames.append(
        pd.DataFrame(
            {
                "furnace": furnace,
                "fold": fold,
                "model": model,
                "random_seed": random_seed,
                "row": rows,
                "y": y,
                "current_si": current,
                "prediction": prediction,
                "persistence_error": persistence_error,
                "segment": persistence_segment(persistence_error),
                "abs_error": np.abs(y - prediction),
            }
        )
    )


def fit_xgboost(
    x: np.ndarray,
    delta: np.ndarray,
    train: np.ndarray,
    val: np.ndarray,
    seed: int,
) -> tuple[XGBRegressor, float]:
    model = XGBRegressor(
        n_estimators=1000,
        max_depth=7,
        learning_rate=0.03,
        min_child_weight=8,
        subsample=0.85,
        colsample_bytree=0.80,
        reg_alpha=0.05,
        reg_lambda=5.0,
        objective="reg:squarederror",
        eval_metric="mae",
        random_state=seed,
        n_jobs=-1,
        early_stopping_rounds=100,
    )
    started = time.perf_counter()
    model.fit(x[train], delta[train], eval_set=[(x[val], delta[val])], verbose=False)
    return model, time.perf_counter() - started


def summarize_metrics(metrics: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    return (
        metrics.groupby(groups)
        .agg(
            runs=("fold", "size"),
            folds=("fold", "nunique"),
            seeds=("random_seed", "nunique"),
            samples=("samples", "sum"),
            hr_005_mean=("hr_0.05", "mean"),
            hr_005_std=("hr_0.05", "std"),
            hr_010_mean=("hr_0.10", "mean"),
            hr_010_std=("hr_0.10", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
        )
        .reset_index()
    )


def segment_metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for keys, group in predictions.groupby(
        ["furnace", "fold", "model", "random_seed", "segment"]
    ):
        furnace, fold, model, random_seed, segment = keys
        rows.append(
            {
                "furnace": furnace,
                "fold": fold,
                "model": model,
                "random_seed": random_seed,
                "segment": segment,
                "samples": len(group),
                **metric_row(
                    group["y"].to_numpy(),
                    group["prediction"].to_numpy(),
                    0.05,
                ),
            }
        )
    fold = pd.DataFrame(rows)
    summary = summarize_metrics(fold, ["furnace", "model", "segment"])
    return fold, summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    records: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []

    for furnace in args.furnaces:
        process_raw, target, _ = read_furnace(args.data_dir / f"{furnace}.csv")
        for fold, train_end, val_end, test_end in rolling_boundaries(target):
            process = prepare_process(process_raw, train_end)
            data = build_adr_samples(process, target, args.window, args.delay)
            flat_x = np.concatenate([data["process_x"], data["si_x"]], axis=1)
            y = data["y"]
            current = data["current"]
            rows = data["rows"]

            _, sequence_x, deep_y, deep_current, deep_rows = build_samples(
                process,
                target,
                args.window,
                1,
                si_delay_hours=args.delay,
                include_si_history=True,
            )
            if not (
                np.array_equal(rows, deep_rows)
                and np.allclose(y, deep_y)
                and np.allclose(current, deep_current)
            ):
                raise AssertionError("Tree and deep samples are not aligned.")

            train = rows < train_end
            val = (rows >= train_end) & (rows < val_end)
            test = (rows >= val_end) & (rows < test_end)
            delta = y - current
            tree_seed = args.seed + fold

            append_metrics(
                records,
                furnace,
                fold,
                "Persistence",
                tree_seed,
                y[test],
                current[test],
                0.0,
                0.0,
            )
            append_predictions(
                prediction_frames,
                furnace,
                fold,
                "Persistence",
                tree_seed,
                rows[test],
                y[test],
                current[test],
                current[test],
            )

            started = time.perf_counter()
            cdr = fit_cdr(
                flat_x, delta, train, val, tree_seed, args.cat_iterations
            )
            cdr_seconds = time.perf_counter() - started
            residual_test = cdr.predict(flat_x[test])

            standard = current[test] + residual_test
            common_metadata = {"best_iteration": int(cdr.get_best_iteration())}
            append_metrics(
                records,
                furnace,
                fold,
                STANDARD_MODEL,
                tree_seed,
                y[test],
                standard,
                1.0,
                cdr_seconds,
                common_metadata,
            )
            append_predictions(
                prediction_frames,
                furnace,
                fold,
                STANDARD_MODEL,
                tree_seed,
                rows[test],
                y[test],
                current[test],
                standard,
            )

            xgb, xgb_seconds = fit_xgboost(flat_x, delta, train, val, tree_seed)
            xgb_val = current[val] + xgb.predict(flat_x[val])
            xgb_alpha, _ = optimize_adjustment(
                y[val], xgb_val, current[val], 0.05, allow_bias=False
            )
            xgb_test = current[test] + xgb.predict(flat_x[test])
            xgb_prediction = apply_adjustment(
                xgb_test, current[test], xgb_alpha, 0.0
            )
            append_metrics(
                records,
                furnace,
                fold,
                "XGBoost residual",
                tree_seed,
                y[test],
                xgb_prediction,
                xgb_alpha,
                xgb_seconds,
                {"best_iteration": int(xgb.best_iteration) + 1},
            )
            append_predictions(
                prediction_frames,
                furnace,
                fold,
                "XGBoost residual",
                tree_seed,
                rows[test],
                y[test],
                current[test],
                xgb_prediction,
            )

            for deep_seed in deep_seeds(args.seed, args.deep_seed_count):
                for name in DEEP_MODELS:
                    val_prediction, test_prediction, metadata = train_deep(
                        name,
                        sequence_x,
                        y,
                        current,
                        train,
                        val,
                        test,
                        args,
                        deep_seed,
                    )
                    deep_alpha, _ = optimize_adjustment(
                        y[val],
                        val_prediction,
                        current[val],
                        0.05,
                        allow_bias=False,
                    )
                    prediction = apply_adjustment(
                        test_prediction, current[test], deep_alpha, 0.0
                    )
                    model_name = f"{name} residual"
                    append_metrics(
                        records,
                        furnace,
                        fold,
                        model_name,
                        deep_seed,
                        y[test],
                        prediction,
                        deep_alpha,
                        metadata["train_seconds"],
                        {
                            "best_epoch": metadata["best_epoch"],
                            "parameters": metadata["parameters"],
                            "device": metadata["device"],
                        },
                    )
                    append_predictions(
                        prediction_frames,
                        furnace,
                        fold,
                        model_name,
                        deep_seed,
                        rows[test],
                        y[test],
                        current[test],
                        prediction,
                    )
                    print(
                        f"done {furnace} fold={fold} seed={deep_seed} "
                        f"{model_name}",
                        flush=True,
                    )

            pd.DataFrame(records).to_csv(
                args.output_dir / "fold_metrics.partial.csv",
                index=False,
                encoding="utf-8-sig",
            )
            pd.concat(prediction_frames, ignore_index=True).to_csv(
                args.output_dir / "predictions.partial.csv",
                index=False,
                encoding="utf-8-sig",
            )

    metrics = pd.DataFrame(records)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    segment_fold, segment_by_furnace = segment_metrics(predictions)
    overall = summarize_metrics(metrics, ["model"])
    furnace = summarize_metrics(metrics, ["furnace", "model"])
    segment_overall = summarize_metrics(segment_fold, ["model", "segment"])

    metrics.to_csv(
        args.output_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig"
    )
    predictions.to_csv(
        args.output_dir / "predictions.csv", index=False, encoding="utf-8-sig"
    )
    overall.to_csv(
        args.output_dir / "overall_summary.csv", index=False, encoding="utf-8-sig"
    )
    furnace.to_csv(
        args.output_dir / "furnace_summary.csv", index=False, encoding="utf-8-sig"
    )
    segment_fold.to_csv(
        args.output_dir / "segment_fold_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    segment_by_furnace.to_csv(
        args.output_dir / "segment_by_furnace.csv",
        index=False,
        encoding="utf-8-sig",
    )
    segment_overall.to_csv(
        args.output_dir / "segment_overall.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (args.output_dir / "report.json").write_text(
        json.dumps(
            {
                "furnaces": args.furnaces,
                "delay_hours": args.delay,
                "deep_seeds": deep_seeds(args.seed, args.deep_seed_count),
                "tree_seed_rule": "seed + fold",
                "continuous_temperature_removed": True,
                "cdr_protocol": "Unscaled residual prediction (alpha=1).",
                "final_model": STANDARD_MODEL,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(overall.sort_values("hr_005_mean", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
