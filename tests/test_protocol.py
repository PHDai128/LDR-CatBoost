"""Synthetic checks for the exact feature builder used in the paper."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from feature_builder import build_adr_samples  # noqa: E402
from data_splits import read_furnace, rolling_boundaries  # noqa: E402


class AvailabilityTests(unittest.TestCase):
    def test_anonymous_furnace_1_excludes_other_si_columns(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "furnace_1.csv"
            pd.DataFrame({
                "time": pd.date_range("2024-01-01", periods=30, freq="h"),
                "铁水硅（小时平均插值法）": np.linspace(.2, .4, 30),
                "预测硅": np.linspace(.9, 1.0, 30),
                "铁水硅（跨整点线性插值法）": np.linspace(.7, .8, 30),
                "blast": np.arange(30),
            }).to_csv(path, index=False)
            process, target, selected = read_furnace(path)
            self.assertEqual(selected, "铁水硅（小时平均插值法）")
            self.assertEqual(list(process.columns), ["blast"])
            self.assertEqual(len(target), 30)

    def test_anchor_uses_only_released_si(self) -> None:
        n = 90
        index = pd.date_range("2024-01-01", periods=n, freq="h")
        values = 0.35 + np.arange(n, dtype=np.float32) * 0.001
        values[[14, 29, 52]] = np.nan
        target = pd.Series(values, index=index)
        process = pd.DataFrame({"blast": np.arange(n, dtype=np.float32),
                                "pressure": np.ones(n, dtype=np.float32)}, index=index)
        samples = build_adr_samples(process, target, window=24, delay=2)
        available = target.shift(2).ffill()
        for row, anchor in zip(samples["rows"], samples["current"]):
            self.assertAlmostEqual(float(anchor), float(available.iloc[int(row) - 1]), places=6)
            self.assertTrue(np.isfinite(target.iloc[int(row)]))
        self.assertEqual(samples["si_x"].shape[1], 33)
        self.assertEqual(samples["process_x"].shape[1], 66)

    def test_rolling_test_intervals_do_not_overlap(self) -> None:
        target = pd.Series(np.linspace(.2, .5, 100))
        folds = rolling_boundaries(target)
        self.assertEqual([(a, b, c) for _, a, b, c in folds],
                         [(50, 60, 70), (60, 70, 80), (70, 80, 90)])
        self.assertEqual(folds[0][3], folds[1][2])
        self.assertEqual(folds[1][3], folds[2][2])


if __name__ == "__main__":
    unittest.main()
