"""Paired block bootstrap for the primary LDR comparisons.

Each furnace-fold receives equal weight. For DLinear, five random seeds are
averaged at each aligned test timestamp before forming paired differences.
Blocks never cross a gap in the observed hourly target rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def contiguous_runs(rows: np.ndarray) -> list[np.ndarray]:
    cuts = np.r_[0, np.flatnonzero(np.diff(rows) != 1) + 1, len(rows)]
    return [np.arange(a, b) for a, b in zip(cuts[:-1], cuts[1:]) if b > a]


def bootstrap_fold_draws(values: np.ndarray, runs: list[np.ndarray], block: int,
                         repeats: int, rng: np.random.Generator) -> np.ndarray:
    """Circular moving blocks within each consecutive-hour run, vectorized."""
    total = np.zeros(repeats, dtype=np.float64)
    total_n = 0
    for run in runs:
        v = values[run]
        length = len(v)
        width = min(block, length)
        q, remainder = divmod(length, width)
        circular = np.r_[v, v[:width]]
        prefix = np.r_[0.0, np.cumsum(circular)]
        full_sums = prefix[width:width + length] - prefix[:length]
        starts = rng.integers(0, length, size=(repeats, q))
        total += full_sums[starts].sum(axis=1)
        if remainder:
            partial_sums = prefix[remainder:remainder + length] - prefix[:length]
            total += partial_sums[rng.integers(0, length, size=repeats)]
        total_n += length
    return total / total_n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, default=ROOT / "results" / "primary" / "predictions.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "paired_bootstrap")
    ap.add_argument("--repeats", type=int, default=5000)
    ap.add_argument("--blocks", nargs="+", type=int, default=[12, 24, 48])
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.predictions)
    ours_name = "Ours: Standard CDR-CatBoost"
    ours = data.loc[data.model == ours_name, ["furnace", "fold", "row", "y", "prediction"]].copy()
    assert not ours.duplicated(["furnace", "fold", "row"]).any()
    sequence_rows = []
    for (furnace, fold), g in ours.groupby(["furnace", "fold"], sort=True):
        lengths = [len(run) for run in contiguous_runs(g.sort_values("row").row.to_numpy())]
        sequence_rows.append({"furnace": furnace, "fold": fold, "test_rows": len(g),
                              "contiguous_runs": len(lengths), "shortest_run_hours": min(lengths),
                              "longest_run_hours": max(lengths)})
    pd.DataFrame(sequence_rows).to_csv(args.out / "fold_sequence_structure.csv", index=False)
    out_rows = []
    for baseline_name in ("XGBoost residual", "DLinear residual"):
        base = data.loc[data.model == baseline_name, ["furnace", "fold", "row", "y", "prediction", "random_seed"]].copy()
        keys = ["furnace", "fold", "row"]
        assert base.groupby(keys).size().nunique() == 1
        seed_count = int(base.groupby(keys).size().iloc[0])
        for threshold in (0.05, 0.10):
            base[f"hit_{threshold}"] = (base.y - base.prediction).abs().le(threshold).astype(float)
        base["ae"] = (base.y - base.prediction).abs()
        base = base.groupby(keys, as_index=False).agg({"y": "first", "ae": "mean", "hit_0.05": "mean", "hit_0.1": "mean"})
        aligned = ours.merge(base, on=keys, suffixes=("_ours", "_base"), validate="one_to_one")
        assert len(aligned) == len(ours) == len(base)
        assert np.allclose(aligned.y_ours, aligned.y_base)
        err_ours = (aligned.y_ours - aligned.prediction).abs()
        aligned["mae_gain"] = aligned.ae - err_ours
        aligned["hr005_gain"] = err_ours.le(.05).astype(float) - aligned["hit_0.05"]
        aligned["hr010_gain"] = err_ours.le(.10).astype(float) - aligned["hit_0.1"]
        folds = [g.sort_values("row") for _, g in aligned.groupby(["furnace", "fold"], sort=True)]
        run_indices = [contiguous_runs(g.row.to_numpy()) for g in folds]
        assert len(folds) == 9
        for metric in ("mae_gain", "hr005_gain", "hr010_gain"):
            point = float(np.mean([g[metric].mean() for g in folds]))
            fold_values = [g[metric].to_numpy() for g in folds]
            for block in args.blocks:
                rng = np.random.default_rng(20261005 + block)
                samples = np.mean([bootstrap_fold_draws(values, runs, block, args.repeats, rng)
                                   for values, runs in zip(fold_values, run_indices)], axis=0)
                lo, hi = np.quantile(samples, [.025, .975])
                p = min(1.0, 2 * min((np.count_nonzero(samples <= 0) + 1) / (args.repeats + 1),
                                     (np.count_nonzero(samples >= 0) + 1) / (args.repeats + 1)))
                out_rows.append({"baseline": baseline_name, "baseline_seeds_per_fold": seed_count,
                                 "metric": metric, "block_hours": block, "repeats": args.repeats,
                                 "point_gain": point, "ci_low": float(lo), "ci_high": float(hi),
                                 "two_sided_bootstrap_p": float(p), "test_rows": len(aligned), "folds": 9})
    pd.DataFrame(out_rows).to_csv(args.out / "results.csv", index=False)
    (args.out / "method.json").write_text(json.dumps({
        "source": str(args.predictions), "weighting": "equal furnace-fold means",
        "DLinear_seed_handling": "mean of per-timestamp error/hit over five seeds; seeds are not independent datasets",
        "blocks": "circular moving blocks drawn within contiguous observed hourly target runs; no block crosses a missing-hour gap",
        "short_runs": "block length clipped to the length of a contiguous run",
        "confidence_interval": "2.5th and 97.5th percentiles of bootstrap gain distribution",
        "p_value": "two times smaller tail mass relative to zero with plus-one correction",
    }, indent=2), encoding="utf-8")
    print(pd.DataFrame(out_rows).to_string(index=False))


if __name__ == "__main__":
    main()
