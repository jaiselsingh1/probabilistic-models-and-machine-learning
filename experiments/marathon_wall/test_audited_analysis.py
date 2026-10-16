"""Focused checks for the held-out analysis; run with python -m unittest."""

import unittest

import numpy as np
import pandas as pd
from scipy import integrate

from .audited_analysis import (count_modes, density_cdf, fit_mixture, fit_variant,
                               log_density, score_comparisons, stratified_split)


class AuditedAnalysisTests(unittest.TestCase):
    def test_density_normalization_and_mode_count(self):
        one = {"weight_upper": .2, "mu_lower": 0., "mu_upper": 1.,
               "sigma_lower": 1., "sigma_upper": 2.}
        two = {"weight_upper": .5, "mu_lower": -3., "mu_upper": 3.,
               "sigma_lower": .5, "sigma_upper": .5}
        for p, modes in ((one, 1), (two, 2)):
            area, _ = integrate.quad(lambda y: float(np.exp(log_density("mixture", y, p))), -np.inf, np.inf)
            self.assertAlmostEqual(area, 1., places=8)
            self.assertEqual(count_modes(p), modes)
            self.assertTrue(np.all(np.diff(density_cdf("mixture", np.linspace(-10, 10, 100), p)) >= 0))

    def test_mixture_recovers_separated_synthetic_components(self):
        rng = np.random.default_rng(7)
        z = rng.random(5000) < .3
        y = rng.normal(np.where(z, 3., 0.), .4)
        p = fit_mixture(y)
        self.assertTrue(p["converged"])
        self.assertLess(p["mu_lower"], p["mu_upper"])
        self.assertAlmostEqual(p["weight_upper"], .3, delta=.025)
        self.assertAlmostEqual(p["mu_upper"], 3., delta=.05)
        self.assertFalse(p["floor_active"])

    def test_holdout_values_cannot_change_reference_or_fits(self):
        rng = np.random.default_rng(12)
        n = 3000
        frame = pd.DataFrame({"row_id": np.arange(n), "year": 2014, "sex": "F", "bib": np.arange(n),
                              "group": "F x 2014", "ability": rng.uniform(3, 8, n),
                              "raw_ratio": rng.normal(0, .1, n) + rng.exponential(.1, n)})
        train = stratified_split(frame)
        self.assertEqual(int(train.sum()), 2400)
        np.testing.assert_array_equal(train, stratified_split(frame))
        fits, _, reference = fit_variant(frame, train, .25)
        altered = frame.copy()
        altered.loc[~train, "raw_ratio"] += .5
        other_fits, _, other_reference = fit_variant(altered, train, .25)
        self.assertEqual(fits, other_fits)
        np.testing.assert_array_equal(reference.delta_knots, other_reference.delta_knots)

    def test_total_standard_error_respects_fixed_stratum_sizes(self):
        scores = pd.DataFrame({"group": ["a", "a", "b", "b"],
                               "logp_mixture": [0., 2., 100., 102.],
                               "logp_gaussian": 0., "logp_skewnormal": 0.})
        rows = score_comparisons(scores, "test")
        self.assertAlmostEqual(rows[0]["se_total_nats"], np.sqrt(8))
        self.assertAlmostEqual(rows[0]["gain_total_nats"], 204.)


if __name__ == "__main__":
    unittest.main()
