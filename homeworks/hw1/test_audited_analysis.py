"""Focused checks for the held-out analysis; run with python -m unittest."""

import unittest

import numpy as np
import pandas as pd
from scipy import integrate

from .audited_analysis import (count_modes, density_cdf, fit_mixture, fit_variant,
                               fit_year_variant, log_density, mixture_crossover_diagnostics,
                               score_comparisons, score_gain_partition, stratified_split,
                               training_gain_thresholds)
from .metric import TerrainReference


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

    def test_crossover_uses_weights_and_has_upper_responsibility_in_both_tails(self):
        p = {"weight_upper": .2, "mu_lower": 0., "mu_upper": 1.,
             "sigma_lower": 1., "sigma_upper": 2.}
        diagnostic = mixture_crossover_diagnostics(p)
        self.assertEqual(diagnostic["n_real_crossovers"], 2)
        self.assertEqual(diagnostic["sd_ratio"], 2.)
        self.assertEqual(diagnostic["variance_ratio"], 4.)
        left, right = diagnostic["left_crossover_y"], diagnostic["right_crossover_y"]
        # Evaluate responsibilities independently from weighted component PDFs.
        def responsibility(y):
            lower = (1 - p["weight_upper"]) / p["sigma_lower"] * np.exp(-.5 * ((y - p["mu_lower"]) / p["sigma_lower"]) ** 2)
            upper = p["weight_upper"] / p["sigma_upper"] * np.exp(-.5 * ((y - p["mu_upper"]) / p["sigma_upper"]) ** 2)
            return upper / (lower + upper)
        self.assertAlmostEqual(responsibility(left), .5, places=12)
        self.assertAlmostEqual(responsibility(right), .5, places=12)
        self.assertGreater(responsibility(left - .1), .5)
        self.assertGreater(responsibility(right + .1), .5)
        self.assertLess(responsibility((left + right) / 2), .5)
        unweighted = mixture_crossover_diagnostics({**p, "weight_upper": .5})
        self.assertGreater(abs(left - unweighted["left_crossover_y"]), .1)

    def test_score_partition_is_exhaustive_additive_and_train_only(self):
        reference = TerrainReference(np.array([0., 20.]), np.zeros(2), .25, 2)
        frame = pd.DataFrame({"group": ["a"] * 9 + ["b"] * 9,
                              "raw_ratio": [0., 1., 2., 3., 4., -1., .2, 3.8, 5.,
                                            10., 11., 12., 13., 14., 9., 10.2, 13.8, 15.],
                              "ability": 5.})
        train = np.tile([True] * 5 + [False] * 4, 2)
        scores = frame.loc[~train, ["group"]].copy()
        scores["y"] = frame.loc[~train, "raw_ratio"]
        scores["logp_mixture"] = [1., -2., -1., 4., 2., 3., -1., 4.]
        scores["logp_skewnormal"] = 0.
        table = score_gain_partition(frame, train, scores, reference)
        total = table.loc[table["group"] == "all"].set_index("region")
        self.assertEqual(total["n_test"].sum(), len(scores))
        self.assertEqual(total.loc["central", "n_test"], 4)  # Both boundary ties belong here.
        self.assertEqual(total.loc["left", "n_test"], 2)
        self.assertEqual(total.loc["right", "n_test"], 2)
        self.assertAlmostEqual(total["gain_total_nats"].sum(), scores["logp_mixture"].sum())
        self.assertAlmostEqual(total["percent_of_total_gain"].sum(), 100.)
        self.assertLess(total.loc["central", "gain_total_nats"], 0.)
        self.assertAlmostEqual(total.loc["central", "gain_total_nats"], -1.)
        altered = frame.copy()
        altered.loc[~train, "raw_ratio"] += 1000
        self.assertEqual(training_gain_thresholds(frame, train, reference),
                         training_gain_thresholds(altered, train, reference))
        pd.testing.assert_frame_equal(table, score_gain_partition(altered, train, scores, reference))

    def test_year_sensitivity_excludes_other_year_and_preserves_split(self):
        rng = np.random.default_rng(19)
        n = 3500
        years = np.r_[np.full(500, 2013), np.full(3000, 2014)]
        frame = pd.DataFrame({"row_id": np.arange(n) + 20, "year": years, "sex": "F", "bib": np.arange(n),
                              "group": [f"F x {year}" for year in years], "ability": rng.uniform(3, 8, n),
                              "raw_ratio": rng.normal(0, .1, n) + rng.exponential(.1, n)})
        train = stratified_split(frame)
        fits, scores, reference = fit_year_variant(frame, train, 2014)
        changed = frame.copy()
        changed.loc[changed["year"] == 2013, ["ability", "raw_ratio"]] += 100
        other_fits, other_scores, other_reference = fit_year_variant(changed, train, 2014)
        self.assertEqual(fits, other_fits)
        pd.testing.assert_frame_equal(scores, other_scores)
        np.testing.assert_array_equal(reference.ability_knots, other_reference.ability_knots)
        np.testing.assert_array_equal(reference.delta_knots, other_reference.delta_knots)
        self.assertEqual(set(fits), {"F x 2014"})
        self.assertEqual(fits["F x 2014"]["n_train"], 2400)
        self.assertEqual(set(scores["year"]), {2014})
        np.testing.assert_array_equal(scores["row_id"], frame.loc[(frame["year"] == 2014) & ~train, "row_id"])


if __name__ == "__main__":
    unittest.main()
