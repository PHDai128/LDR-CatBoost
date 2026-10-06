"""Describe the supplied hourly target files without assuming lab-event provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "hourly_si_audit"
OUT.mkdir(parents=True, exist_ok=True)
TARGETS = {
    "furnace_1": "铁水硅（小时平均插值法）",
    "furnace_2": "小时平均SI",
    "furnace_3": "平均si",
}
records = []
for furnace, target_name in TARGETS.items():
    path = ROOT / "data" / f"{furnace}.csv"
    raw = pd.read_csv(path, encoding="utf-8-sig")
    stamp = pd.to_datetime(raw.iloc[:, 0], errors="coerce")
    y = pd.to_numeric(raw[target_name], errors="coerce")
    valid = stamp.notna() & y.between(0, 4)
    valid_y = y[valid]
    records.append({
        "furnace": furnace, "csv_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "target_column": target_name, "rows": len(raw), "valid_si_rows": int(valid.sum()),
        "valid_fraction": float(valid.mean()), "first_timestamp": str(stamp.min()),
        "last_timestamp": str(stamp.max()), "si_mean": float(valid_y.mean()),
        "si_std": float(valid_y.std()), "si_min": float(valid_y.min()),
        "si_q05": float(valid_y.quantile(.05)), "si_median": float(valid_y.median()),
        "si_q95": float(valid_y.quantile(.95)), "si_max": float(valid_y.max()),
        "si_unit_user_confirmed": "wt.%",
        "original_lab_event_timestamp_available": False,
        "result_release_timestamp_available": False,
        "hourly_target_generation_rule_verified": False,
    })
pd.DataFrame(records).to_csv(OUT / "target_descriptives.csv", index=False, encoding="utf-8-sig")
(OUT / "audit_limitations.json").write_text(json.dumps({
    "finding": "The files contain hourly indexed Si target columns, not raw laboratory event and release timestamps.",
    "furnace_1_warning": "The selected Furnace No. 1 target column is explicitly named hourly-average interpolation method; the source algorithm is not present here.",
    "cannot_conclude": "A shift of this series cannot by itself establish what was truly available at prediction time.",
    "needed_from_provider": ["raw Si sampling timestamp", "actual result release timestamp or documented delay",
                              "hourly target construction and interpolation procedure"],
    "unit_note": "The user confirmed that Si values are in wt.%; this is not encoded in the CSV files.",
}, ensure_ascii=False, indent=2), encoding="utf-8")
print(pd.DataFrame(records).drop(columns=["csv_sha256"]).to_string(index=False))
