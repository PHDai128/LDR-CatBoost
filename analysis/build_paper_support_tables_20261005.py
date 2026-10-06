"""Extract reproducibility and segment summaries from the primary saved run."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data_splits import read_furnace, rolling_boundaries  # noqa: E402
from shared_protocol import prepare_process  # noqa: E402

PRIMARY = ROOT / "results" / "primary"
OUT = ROOT / "results" / "support_tables"
OUT.mkdir(parents=True, exist_ok=True)

folds = pd.read_csv(PRIMARY / "fold_metrics.csv")
folds[["furnace", "fold", "model", "random_seed", "samples", "alpha", "best_iteration", "best_epoch", "device"]].to_csv(
    OUT / "primary_fold_parameters.csv", index=False)

rows = []
variables = []
for furnace in ("furnace_1", "furnace_2", "furnace_3"):
    raw, target, target_name = read_furnace(ROOT / "data" / f"{furnace}.csv")
    for fold, train_end, val_end, test_end in rolling_boundaries(target):
        process = prepare_process(raw, train_end)
        candidate = raw.drop(columns=[c for c in raw.columns if "连续测温" in str(c)], errors="ignore")
        rows.append({"furnace": furnace, "fold": fold, "target_column": target_name,
                     "candidate_process_columns": candidate.shape[1], "selected_process_columns": process.shape[1],
                     "full_feature_dimension": 33 * (process.shape[1] + 1),
                     "train_boundary": str(target.index[train_end]), "validation_boundary": str(target.index[val_end]),
                     "test_boundary": str(target.index[test_end])})
        variables.extend({"furnace": furnace, "fold": fold, "process_variable": name,
                          "unit_in_csv": "not supplied"} for name in process.columns)
pd.DataFrame(rows).to_csv(OUT / "feature_dimensions_by_fold.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(variables).to_csv(OUT / "selected_process_variables_by_fold.csv", index=False, encoding="utf-8-sig")

pred = pd.read_csv(PRIMARY / "predictions.csv")
base = pred[pred.model == "Persistence"]
seg = base.groupby("segment", as_index=False).agg(n_test=("y", "size"))
seg["test_fraction"] = seg.n_test / seg.n_test.sum()
seg.to_csv(OUT / "segment_prevalence.csv", index=False)
seg_fold = base.groupby(["furnace", "fold", "segment"], as_index=False).agg(n_test=("y", "size"))
seg_fold.to_csv(OUT / "segment_prevalence_by_fold.csv", index=False)

print("Saved", OUT)
