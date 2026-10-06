"""Fixed laboratory-release-delay sensitivity analysis (manuscript Table 9)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from feature_builder import build_adr_samples
from data_preprocessing import metric_row
from data_splits import read_furnace, rolling_boundaries
from shared_protocol import fit_cdr, prepare_process


FURNACES = ("furnace_1", "furnace_2", "furnace_3")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/delay_sensitivity"))
    parser.add_argument("--furnaces", nargs="+", default=FURNACES)
    parser.add_argument("--delays", nargs="+", type=int, default=[0, 1, 2, 4])
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for furnace in args.furnaces:
        raw, target, _ = read_furnace(args.data_dir / f"{furnace}.csv")
        for fold, train_end, val_end, test_end in rolling_boundaries(target):
            process = prepare_process(raw, train_end)
            for delay in args.delays:
                data = build_adr_samples(process, target, args.window, delay)
                rows = data["rows"]
                train = rows < train_end
                val = (rows >= train_end) & (rows < val_end)
                test = (rows >= val_end) & (rows < test_end)
                if not (train.any() and val.any() and test.any()):
                    raise ValueError(f"Empty split: {furnace}, fold {fold}, delay {delay}")
                features = np.concatenate([data["process_x"], data["si_x"]], axis=1)
                target_values = data["y"]
                anchor = data["current"]
                model = fit_cdr(features, target_values - anchor, train, val,
                                args.seed + fold, args.iterations)
                prediction = anchor[test] + model.predict(features[test])
                records.append({
                    "furnace": furnace, "fold": fold, "delay_hours": delay,
                    "train_samples": int(train.sum()),
                    "validation_samples": int(val.sum()),
                    "test_samples": int(test.sum()),
                    "residual_scale": 1.0,
                    **metric_row(target_values[test], prediction, 0.05),
                })
                print(f"{furnace} fold={fold} delay={delay} complete", flush=True)

    folds = pd.DataFrame(records)
    folds.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    summary = (folds.groupby(["furnace", "delay_hours"], as_index=False)
               .agg(folds=("fold", "nunique"), test_samples=("test_samples", "sum"),
                    hr005=("hr_0.05", "mean"), hr010=("hr_0.10", "mean"),
                    mae=("mae", "mean")))
    summary.to_csv(args.output_dir / "furnace_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
