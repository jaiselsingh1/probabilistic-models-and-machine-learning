"""Regression checks for consequential split-data handling errors."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from .data import (SCHEMAS, HygieneReport, _enforce_monotone,
                   _interpolate_interior, load_race, parse_clock_series)
from .metric import TerrainReference, build_metric


class SplitDataTests(unittest.TestCase):
    def test_mixed_clock_rows_and_missing(self):
        out = parse_clock_series(pd.Series(['1:02:30', '02:30', '62.5', '-', '0',
                                            '1:75:00', '2:61', 'bad']), 'clock', ('-',))
        np.testing.assert_allclose(out[:3], [62.5, 2.5, 62.5])
        self.assertTrue(out[3:].isna().all())
        self.assertTrue(parse_clock_series(pd.Series(['-', '-']), 'clock', ('-',)).isna().all())

    def test_isolated_not_consecutive_imputation(self):
        frame = pd.DataFrame([[10., np.nan, 30., 40.], [10., np.nan, np.nan, 40.]])
        filled, mask, count = _interpolate_interior(frame, np.array([5., 10., 20., 25.]))
        self.assertAlmostEqual(filled.iloc[0, 1], 10 + 20 / 3)
        self.assertTrue(filled.iloc[1, 1:3].isna().all())
        self.assertEqual(count, 1)
        self.assertEqual(int(mask.to_numpy().sum()), 1)

    def test_nonmonotone_does_not_pass_as_observed(self):
        report = HygieneReport('test')
        fixed = _enforce_monotone(pd.DataFrame([[10., 9., 30.]]), report)
        self.assertTrue(np.isnan(fixed.iloc[0, 1]))
        self.assertEqual(report.blanked_mats, 1)

    def test_first_segment_in_quality_filter(self):
        row = dict(bib='1', gender='M', age='30', **{
            '5k': '5', '10k': '30', '20k': '80', 'half': '85.4875',
            '25k': '105', '30k': '130', '35k': '155', '40k': '180', 'official': '190.975'})
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'race.csv'
            pd.DataFrame([row]).to_csv(path, index=False)
            data, _ = load_race(path, SCHEMAS['llimllib'], 2014)
        self.assertEqual(len(data), 0)

    def test_metric_never_uses_interpolated_required_mat(self):
        frame = pd.DataFrame({'t_10k': [50., 50.], 't_25k': [125., 125.],
                              't_30k': [150., 150.], 't_40k': [210., 210.],
                              'imp_10k': [False, False], 'imp_25k': [False, False],
                              'imp_30k': [False, True], 'imp_40k': [False, False]})
        reference = TerrainReference(np.array([4., 6.]), np.array([.1, .1]), .25, 2)
        metric, _ = build_metric(frame, reference=reference)
        self.assertEqual(len(metric), 1)
        self.assertAlmostEqual(metric['y'].iloc[0], np.log(1.2) - .1)


if __name__ == '__main__':
    unittest.main()
