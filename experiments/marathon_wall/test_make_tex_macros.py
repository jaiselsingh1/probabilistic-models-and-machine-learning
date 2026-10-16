"""Regression checks for emitted TeX and the complete exclusion count."""

from pathlib import Path
import re
import tempfile
import unittest

import pandas as pd

from .make_tex_macros import _macro, build


class TexMacroTests(unittest.TestCase):
    def test_year_groups_emit_valid_distinct_control_words(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            pd.DataFrame({
                "group": ["F x 2013", "F x 2014", "M x 2013"],
                "pi_mean": [0.17, 0.18, 0.29],
                "pi_lo": [0.14, 0.15, 0.27],
                "pi_hi": [0.20, 0.21, 0.32],
            }).to_csv(out_dir / "pi_by_group.csv", index=False)
            lines = [line for line in build(out_dir) if line.startswith("\\newcommand")]
        names = []
        for line in lines:
            match = re.fullmatch(r"\\newcommand\{\\([A-Za-z]+)\}\{[^{}]*\}", line)
            self.assertIsNotNone(match, line)
            names.append(match.group(1))
        self.assertEqual(len(names), 9)
        self.assertEqual(len(set(names)), 9)
        with self.assertRaises(ValueError):
            _macro("wallPiFx2013", "17")

    def test_exclusion_count_includes_missing_metric_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            pd.DataFrame({
                "race": ["a", "a", "b", "b"],
                "n_before": [10, 10, 10, 10],
                "n_after": [10, 8, 10, 9],
                "dropped": [0, 2, 0, 1],
            }).to_csv(out_dir / "hygiene.csv", index=False)
            # Hygiene leaves 17 rows; missing metric-defining splits remove 2 more.
            pd.DataFrame({"y": [0.1] * 15, "y_raw": [0.2] * 15}).to_csv(out_dir / "metric.csv", index=False)
            lines = build(out_dir)
            self.assertIn(r"\newcommand{\wallRawRows}{20}", lines)
            self.assertIn(r"\newcommand{\wallDropped}{5}", lines)
            self.assertIn(r"\newcommand{\wallN}{15}", lines)
            (out_dir / "metric.csv").unlink()
            lines = build(out_dir)
            self.assertIn(r"\newcommand{\wallDropped}{3}", lines)
            self.assertTrue(any("hygiene exclusions only" in line for line in lines))


if __name__ == "__main__":
    unittest.main()
