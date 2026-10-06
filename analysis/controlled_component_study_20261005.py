"""Unscaled, same-split component and target-formulation study for LDR-CatBoost.

This uses the established offline protocol and pre-existing hourly target CSVs.
It does not establish that the upstream hourly targets were causally constructed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, __version__ as catboost_version


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from feature_builder import LAGS, build_adr_samples  # noqa: E402
from data_splits import read_furnace, rolling_boundaries  # noqa: E402
from shared_protocol import prepare_process  # noqa: E402

FURNACES = ("furnace_1", "furnace_2", "furnace_3")
VARIANTS = ("full_residual", "process_residual", "si_residual", "no_rolling_residual", "full_direct", "full_residual_rmse")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "controlled_components")
    ap.add_argument("--device", choices=("GPU", "CPU"), default="GPU")
    ap.add_argument("--iterations", type=int, default=500)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    config = {
        "furnaces": FURNACES, "variants": VARIANTS, "delay_hours": 2,
        "window_hours": 24, "horizon_hours": 1, "iterations": args.iterations,
        "device": args.device, "catboost_version": catboost_version,
        "loss": {"default": "Huber:delta=0.05", "full_residual_rmse": "RMSE"},
        "eval_metric": "MAE", "depth": 7, "learning_rate": 0.035,
        "l2_leaf_reg": 5.0, "early_stopping_rounds": 80,
        "seed_rule": "2026 + fold", "residual_scale": 1.0, "bias": 0.0,
        "data_sha256": {f: hashlib.sha256((args.data_dir / f"{f}.csv").read_bytes()).hexdigest() for f in FURNACES},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "target_provenance_caveat": "Hourly CSV target generation and true release timestamps are not independently verified.",
    }
    config_path = args.out / "config.json"
    if config_path.exists():
        if not args.resume or json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise RuntimeError("Existing output has a different configuration; choose a new output directory.")
    else:
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    for furnace in FURNACES:
        process_raw, target, target_name = read_furnace(args.data_dir / f"{furnace}.csv")
        for fold, train_end, val_end, test_end in rolling_boundaries(target):
            process = prepare_process(process_raw, train_end)
            data = build_adr_samples(process, target, 24, 2)
            rows, y, anchor = data["rows"], data["y"], data["current"]
            train, val, test = rows < train_end, (rows >= train_end) & (rows < val_end), (rows >= val_end) & (rows < test_end)
            assert train.any() and val.any() and test.any()
            px, sx = data["process_x"], data["si_x"]
            lag_blocks = 1 + sum(lag <= 24 for lag in LAGS)
            no_rolling = np.concatenate([px[:, :lag_blocks * process.shape[1]], sx[:, :lag_blocks]], axis=1)
            features = {
                "full_residual": np.concatenate([px, sx], axis=1),
                "process_residual": px,
                "si_residual": sx,
                "no_rolling_residual": no_rolling,
                "full_direct": np.concatenate([px, sx], axis=1),
                "full_residual_rmse": np.concatenate([px, sx], axis=1),
            }
            for variant in VARIANTS:
                run = args.out / "runs" / f"{furnace}_f{fold}_{variant}"
                if args.resume and (run / "metrics.csv").exists() and (run / "predictions.csv").exists():
                    print("SKIP", run.name, flush=True)
                    continue
                run.mkdir(parents=True, exist_ok=True)
                direct = variant == "full_direct"
                labels = y if direct else y - anchor
                model = CatBoostRegressor(
                    iterations=args.iterations, depth=7, learning_rate=0.035,
                    loss_function="RMSE" if variant == "full_residual_rmse" else "Huber:delta=0.05",
                    eval_metric="MAE", l2_leaf_reg=5.0, random_seed=2026 + fold,
                    verbose=False, allow_writing_files=False, task_type=args.device,
                )
                start = time.perf_counter()
                x = features[variant]
                model.fit(x[train], labels[train], eval_set=(x[val], labels[val]), early_stopping_rounds=80)
                pred = model.predict(x[test])
                if not direct:
                    pred = pred + anchor[test]
                err = np.abs(y[test] - pred)
                metric = {
                    "furnace": furnace, "fold": fold, "variant": variant, "target_column": target_name,
                    "n_train": int(train.sum()), "n_val": int(val.sum()), "n_test": int(test.sum()),
                    "n_features": int(x.shape[1]), "hr005": float(np.mean(err <= 0.05)),
                    "hr010": float(np.mean(err <= 0.10)), "mae": float(err.mean()),
                    "best_iteration": int(model.get_best_iteration()), "train_seconds": time.perf_counter() - start,
                }
                pd.DataFrame([metric]).to_csv(run / "metrics.csv", index=False)
                pd.DataFrame({
                    "furnace": furnace, "fold": fold, "variant": variant,
                    "target_row": rows[test], "target_timestamp": target.index[rows[test]],
                    "y_true": y[test], "y_pred": pred, "latest_available_si": anchor[test],
                }).to_csv(run / "predictions.csv", index=False)
                model.save_model(str(run / "model.cbm"))
                print("DONE", run.name, f"MAE={metric['mae']:.5f}", f"HR005={metric['hr005']:.5f}", flush=True)

    folds = pd.concat([pd.read_csv(p) for p in sorted((args.out / "runs").glob("*/metrics.csv"))], ignore_index=True)
    folds.to_csv(args.out / "fold_metrics.csv", index=False)
    overall = folds.groupby("variant", sort=False)[["hr005", "hr010", "mae"]].mean().reset_index()
    overall.to_csv(args.out / "overall_metrics.csv", index=False)
    assert len(folds) == len(FURNACES) * 3 * len(VARIANTS), len(folds)
    print(overall.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
