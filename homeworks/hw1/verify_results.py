"""Independent recomputation from saved outputs, without importing fitting code."""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logsumexp
from scipy.stats import norm, skewnorm


def read_csv(path):
    # Preserve ties and bin-edge membership when rebuilding training references.
    return pd.read_csv(path, float_precision="round_trip")


def close(actual, expected, atol=1e-10):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=atol)


def rebuild_reference(training, quantile, n_bins=40, min_per_bin=50):
    """Reconstruct the documented empirical reference using only this cohort."""
    edges = np.unique(np.quantile(training.ability, np.linspace(0, 1, n_bins + 1)))
    ability, ratio = training.ability.to_numpy(), training.raw_ratio.to_numpy()
    knots = []
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        # Quantile-edge ties belong to the bin on their right; include the maximum.
        member = (ability >= left) & ((ability < right) if index < len(edges) - 2 else (ability <= right))
        if member.sum() >= min_per_bin:
            knots.append([np.median(ability[member]), np.quantile(ratio[member], quantile)])
    return np.asarray(sorted(knots))


def log_odds(y, p):
    return (np.log(p["weight_upper"]) + norm.logpdf(y, p["mu_upper"], p["sigma_upper"])
            - np.log1p(-p["weight_upper"]) - norm.logpdf(y, p["mu_lower"], p["sigma_lower"]))


def numerical_crossovers(p):
    """Bracket both 50% posterior crossings instead of using a quadratic formula."""
    assert p["sigma_upper"] > p["sigma_lower"]
    vertex = ((p["mu_lower"] / p["sigma_lower"] ** 2 - p["mu_upper"] / p["sigma_upper"] ** 2)
              / (1 / p["sigma_lower"] ** 2 - 1 / p["sigma_upper"] ** 2))
    assert log_odds(vertex, p) < 0
    step = max(p["sigma_lower"], p["sigma_upper"])
    for _ in range(100):
        if log_odds(vertex - step, p) > 0 and log_odds(vertex + step, p) > 0:
            break
        step *= 2
    else:
        raise AssertionError("Could not bracket the responsibility crossovers")
    left = brentq(lambda x: log_odds(x, p), vertex - step, vertex, xtol=1e-14)
    right = brentq(lambda x: log_odds(x, p), vertex, vertex + step, xtol=1e-14)
    close(expit(log_odds(np.array([left, right]), p)), [.5, .5], atol=1e-12)
    assert log_odds(left - step, p) > 0 and log_odds(right + step, p) > 0
    return left, right


def verify_macros(path, meta, cohort, hygiene, comparisons, diagnostics, partition):
    """Check displayed numerical values against independently checked tables."""
    actual = {}
    for line in path.read_text().splitlines():
        match = re.fullmatch(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}", line)
        if match:
            assert match[1] not in actual
            actual[match[1]] = match[2].replace("{,}", ",")
    table = comparisons.set_index(["variant", "group", "comparison"])
    expected = {"auditN": f"{len(cohort):,}",
                "auditTrainN": f"{(cohort.partition == 'train').sum():,}",
                "auditTestN": f"{(cohort.partition == 'test').sum():,}",
                "auditRawN": f"{meta['n_raw']:,}", "auditExcludedN": f"{meta['n_excluded']:,}",
                "auditSeed": str(meta["seed"]), "auditReferenceBins": str(meta["reference_n_bins"])}
    for year, suffix in ((2013, "Thirteen"), (2014, "Fourteen")):
        rows = hygiene.loc[(hygiene.race == f"llimllib-{year}") & (hygiene.step == "rows in raw file")]
        expected[f"auditRawYear{suffix}"] = f"{int(rows.n_before.sum()):,}"
    for suffix, step in (("Wheelchair", "drop wheelchair / handcycle / push-rim"),
                         ("Duplicate", "drop duplicate bibs"),
                         ("PaceExcluded", "drop segment pace outside"),
                         ("EndpointExcluded", "require real positive metric endpoints")):
        expected[f"audit{suffix}N"] = f"{int(hygiene.loc[hygiene.step.str.startswith(step), 'dropped'].sum()):,}"
    expected["auditNonmonotoneN"] = f"{int(hygiene.groupby('race').blanked_mats.first().sum()):,}"
    expected.update(auditBulkLowerPercentile="5", auditBulkUpperPercentile="95")
    specifications = (("q25", "all", "skewnormal", "Skew"),
                      ("q25", "all", "gaussian", "Gauss"),
                      ("raw", "all", "skewnormal", "Raw"),
                      ("q10", "all", "skewnormal", "Ten"),
                      ("q50", "all", "skewnormal", "Fifty"),
                      ("q25", "2014 only", "skewnormal", "YearFourteen"),
                      ("q25_2014", "all", "skewnormal", "YearFourteenSeparate"))
    for variant, group, other, suffix in specifications:
        row = table.loc[(variant, group, f"mixture_minus_{other}")]
        expected[f"auditGain{suffix}"] = f"{row.gain_total_nats:.1f}"
        expected[f"auditSe{suffix}"] = f"{row.se_total_nats:.1f}"
    row = table.loc[("q25", "all", "mixture_minus_skewnormal")]
    expected["auditGainSkewPerRunner"] = f"{row.gain_per_runner_nats:.4f}"
    expected["auditSeSkewPerRunner"] = f"{row.se_per_runner_nats:.4f}"
    mode_counts = [g["parameters"]["mixture"]["density_modes"] for g in meta["fits"]["q25"].values()]
    expected.update(auditGroups=str(len(mode_counts)), auditUnimodalGroups=str(sum(m == 1 for m in mode_counts)))
    for column, prefix, decimals in (("sd_ratio", "SdRatio", 3), ("variance_ratio", "VarianceRatio", 3),
                                     ("left_crossover_y", "LeftCrossover", 3)):
        expected[f"audit{prefix}Min"] = f"{diagnostics[column].min():.{decimals}f}"
        expected[f"audit{prefix}Max"] = f"{diagnostics[column].max():.{decimals}f}"
    for group, suffix in (("F x 2013", "FemaleThirteen"), ("F x 2014", "FemaleFourteen"),
                          ("M x 2013", "MaleThirteen"), ("M x 2014", "MaleFourteen")):
        row = diagnostics.loc[diagnostics.group == group].iloc[0]
        expected[f"auditLeftCrossover{suffix}"] = f"{row.left_crossover_y:.3f}"
    for region, suffix in (("central", "Bulk"), ("left", "LeftTail"), ("right", "RightTail")):
        row = partition.loc[(partition.group == "all") & (partition.region == region)].iloc[0]
        expected[f"auditGain{suffix}"] = f"{row.gain_total_nats:.1f}"
        expected[f"auditGain{suffix}Percent"] = f"{row.percent_of_total_gain:.1f}"
        expected[f"audit{suffix}N"] = f"{int(row.n_test):,}"
    assert actual == expected, {"missing": expected.keys() - actual.keys(),
                                "unexpected": actual.keys() - expected.keys(),
                                "different": {key: (actual.get(key), value) for key, value in expected.items()
                                              if actual.get(key) != value}}
    return len(expected)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--fig-dir", type=Path, default=Path(__file__).resolve().parents[1] / "figures",
                        help="Directory containing the generated audit_numbers.tex")
    args = parser.parse_args(argv)
    out = args.out_dir
    required = ("results.json", "cohort.csv", "heldout_scores.csv", "comparisons.csv", "hygiene.csv",
                "mixture_diagnostics.csv", "score_gain_partition.csv", "training_year_composition.csv", "diagnostics.json")
    missing = [name for name in required if not (out / name).exists()]
    if not (args.fig_dir / "audit_numbers.tex").exists():
        missing.append(str(args.fig_dir / "audit_numbers.tex"))
    if missing:
        parser.error("Missing " + ", ".join(missing) + "; run python -m homeworks.hw1.audited_analysis first.")
    meta = json.loads((out / "results.json").read_text())
    cohort = read_csv(out / "cohort.csv").set_index("row_id")
    scores = read_csv(out / "heldout_scores.csv")
    comparisons = read_csv(out / "comparisons.csv")
    hygiene = read_csv(out / "hygiene.csv")
    diagnostics = read_csv(out / "mixture_diagnostics.csv")
    partition = read_csv(out / "score_gain_partition.csv")
    composition = read_csv(out / "training_year_composition.csv")
    details = json.loads((out / "diagnostics.json").read_text())
    assert cohort.index.is_unique and set(cohort.partition) == {"train", "test"}
    assert not scores.duplicated(["variant", "row_id"]).any()
    assert set(scores.variant) == set(comparisons.variant) == set(meta["fits"])
    close([len(cohort), (cohort.partition == "train").sum(), (cohort.partition == "test").sum()],
          [meta["n"], meta["n_train"], meta["n_test"]])
    assert int(hygiene.loc[hygiene.step == "rows in raw file", "n_before"].sum()) == meta["n_raw"]
    assert meta["n_excluded"] == meta["n_raw"] - len(cohort)
    # Confirm that the original split is preserved, including in the 2014-only check.
    rng = np.random.default_rng(meta["seed"])
    expected_training = set()
    for _, group in cohort.groupby("group", sort=True):
        indices = rng.permutation(group.index.to_numpy())
        expected_training.update(indices[:int(np.floor(meta["train_fraction"] * len(indices)))])
    assert expected_training == set(cohort.index[cohort.partition == "train"])

    maximum_error, modes, references, expected_rows = 0.0, {}, {}, 0
    for variant, fit_groups in meta["fits"].items():
        population = cohort.loc[cohort.year == 2014] if variant == "q25_2014" else cohort
        training = population.loc[population.partition == "train"]
        held = scores.loc[scores.variant == variant]
        assert set(held.row_id) == set(population.index[population.partition == "test"])
        assert set(fit_groups) == set(population.group)
        expected_rows += int((population.partition == "test").sum())
        raw = cohort.loc[held.row_id]
        assert (raw.group.to_numpy() == held.group.to_numpy()).all()
        assert (raw.year.to_numpy() == held.year.to_numpy()).all()
        reference = None
        if variant != "raw":
            quantile = {"q10": .1, "q25": .25, "q50": .5, "q25_2014": .25}[variant]
            reference = read_csv(out / f"reference_{variant}.csv").to_numpy()
            close(reference, rebuild_reference(training, quantile, meta["reference_n_bins"], meta["reference_min_per_bin"]), atol=1e-14)
            references[variant] = reference
        settings = meta["variant_training"][variant]
        assert settings["reference_quantile"] == (None if variant == "raw" else quantile)
        assert settings["fit_years"] == sorted(population.year.unique().tolist())
        assert settings["reference_years"] == ([] if variant == "raw" else settings["fit_years"])
        y = raw.raw_ratio.to_numpy(copy=True)
        if reference is not None:
            y -= np.interp(raw.ability, reference[:, 0], reference[:, 1])
        close(y, held.y, atol=1e-14)
        for group, fit in fit_groups.items():
            sub = held.loc[held.group == group]
            yy, pars = sub.y.to_numpy(), fit["parameters"]
            train_group = training.loc[training.group == group]
            train_y = train_group.raw_ratio.to_numpy(copy=True)
            if reference is not None:
                train_y -= np.interp(train_group.ability, reference[:, 0], reference[:, 1])
            assert (fit["n_train"], fit["n_test"]) == (len(train_group), len(sub))
            close([pars["gaussian"]["mu"], pars["gaussian"]["sigma"]], [train_y.mean(), train_y.std()])
            p, w = pars["mixture"], pars["mixture"]["weight_upper"]
            assert p["mu_lower"] < p["mu_upper"] and 0 < w < 1
            terms = np.array([np.log1p(-w) + norm.logpdf(yy, p["mu_lower"], p["sigma_lower"]),
                              np.log(w) + norm.logpdf(yy, p["mu_upper"], p["sigma_upper"])])
            vals = {"gaussian": norm.logpdf(yy, pars["gaussian"]["mu"], pars["gaussian"]["sigma"]),
                    "skewnormal": skewnorm.logpdf(yy, pars["skewnormal"]["shape"], pars["skewnormal"]["loc"], pars["skewnormal"]["scale"]),
                    "mixture": logsumexp(terms, axis=0)}
            for model, value in vals.items():
                maximum_error = max(maximum_error, float(np.max(np.abs(value - sub[f"logp_{model}"]))))
                close(value, sub[f"logp_{model}"], atol=1e-11)
            close(expit(log_odds(yy, p)), sub.responsibility_upper, atol=1e-13)
            close(norm.cdf(yy, pars["gaussian"]["mu"], pars["gaussian"]["sigma"]), sub.pit_gaussian)
            close(skewnorm.cdf(yy, pars["skewnormal"]["shape"], pars["skewnormal"]["loc"], pars["skewnormal"]["scale"]), sub.pit_skewnormal)
            close((1 - w) * norm.cdf(yy, p["mu_lower"], p["sigma_lower"]) + w * norm.cdf(yy, p["mu_upper"], p["sigma_upper"]), sub.pit_mixture)
            train_terms = np.array([np.log1p(-w) + norm.logpdf(train_y, p["mu_lower"], p["sigma_lower"]),
                                    np.log(w) + norm.logpdf(train_y, p["mu_upper"], p["sigma_upper"])])
            close(logsumexp(train_terms, axis=0).mean(), p["mean_log_likelihood"])
            # Both derivatives have the same sign outside the component means.
            def derivative(x):
                return ((1 - w) * norm.pdf(x, p["mu_lower"], p["sigma_lower"]) * (p["mu_lower"] - x) / p["sigma_lower"] ** 2
                        + w * norm.pdf(x, p["mu_upper"], p["sigma_upper"]) * (p["mu_upper"] - x) / p["sigma_upper"] ** 2)
            grid = np.linspace(p["mu_lower"], p["mu_upper"], 100001)
            d = derivative(grid)
            roots = [brentq(derivative, grid[i], grid[i + 1]) for i in np.flatnonzero(d[:-1] * d[1:] < 0)]
            assert len(roots) == p["density_modes"] == 1
            assert derivative(roots[0] - 1e-6) > 0 and derivative(roots[0] + 1e-6) < 0
            modes[f"{variant}:{group}"] = roots[0]
        expected_keys = {(group, f"mixture_minus_{model}") for group in ["all", "2014 only", *fit_groups]
                         for model in ("gaussian", "skewnormal")}
        rows = comparisons.loc[comparisons.variant == variant]
        assert len(rows) == len(expected_keys) and set(zip(rows.group, rows.comparison)) == expected_keys
        for _, row in rows.iterrows():
            sub = held if row.group == "all" else held.loc[held.year == 2014] if row.group == "2014 only" else held.loc[held.group == row.group]
            other = row.comparison.removeprefix("mixture_minus_")
            dd = sub.logp_mixture - sub[f"logp_{other}"]
            se = np.sqrt(sum(len(group) * np.var(dd.loc[group.index], ddof=1) for _, group in sub.groupby("group")))
            assert row.n_test == len(sub)
            close([dd.sum(), se, dd.mean(), se / len(sub)],
                  [row.gain_total_nats, row.se_total_nats, row.gain_per_runner_nats, row.se_per_runner_nats])
    assert len(scores) == expected_rows
    # The second export is checked as well so it cannot silently become stale.
    for key, table in (("mixture", diagnostics), ("score_gain_partition", partition),
                       ("training_year_composition", composition)):
        reconstructed = pd.DataFrame(details[key])
        pd.testing.assert_frame_equal(reconstructed, table, check_dtype=False, rtol=0, atol=1e-12)
    assert details["variant"] == "q25" and details["comparison"] == "mixture_minus_skewnormal"
    assert details["gain_partition_definition"]["training_quantiles"] == [.05, .95]

    primary = scores.loc[scores.variant == "q25"]
    assert set(diagnostics.group) == set(meta["fits"]["q25"]) and not diagnostics.group.duplicated().any()
    for _, row in diagnostics.iterrows():
        p = meta["fits"]["q25"][row.group]["parameters"]["mixture"]
        ratio = p["sigma_upper"] / p["sigma_lower"]
        left, right = numerical_crossovers(p)
        close([ratio, ratio ** 2, left, right], [row.sd_ratio, row.variance_ratio, row.left_crossover_y, row.right_crossover_y])
        assert row.n_real_crossovers == 2 and row.upper_component_dominates_both_extreme_tails
        grid = np.linspace(left - .1, right + .1, 101)
        close(np.polyval([row.log_odds_quadratic, row.log_odds_linear, row.log_odds_constant], grid), log_odds(grid, p))

    total_gain = float((primary.logp_mixture - primary.logp_skewnormal).sum())
    assert len(partition) == 3 * (1 + len(meta["fits"]["q25"]))
    assert not partition.duplicated(["group", "region"]).any()
    regional_totals = {region: [0, 0.0] for region in ("left", "central", "right")}
    reference = references["q25"]
    for group in meta["fits"]["q25"]:
        training = cohort.loc[(cohort.partition == "train") & (cohort.group == group)]
        train_y = training.raw_ratio - np.interp(training.ability, reference[:, 0], reference[:, 1])
        q05, q95 = np.quantile(train_y, [.05, .95])
        sub = primary.loc[primary.group == group]
        masks = {"left": sub.y < q05, "central": (sub.y >= q05) & (sub.y <= q95), "right": sub.y > q95}
        for region, mask in masks.items():
            row = partition.loc[(partition.group == group) & (partition.region == region)].iloc[0]
            gain = float((sub.loc[mask, "logp_mixture"] - sub.loc[mask, "logp_skewnormal"]).sum())
            close([q05, q95, mask.sum(), gain, 100 * gain / total_gain],
                  [row.training_q05, row.training_q95, row.n_test, row.gain_total_nats, row.percent_of_total_gain])
            regional_totals[region][0] += int(mask.sum())
            regional_totals[region][1] += gain
    for region, (count, gain) in regional_totals.items():
        row = partition.loc[(partition.group == "all") & (partition.region == region)].iloc[0]
        assert np.isnan(row.training_q05) and np.isnan(row.training_q95)
        close([count, gain, 100 * gain / total_gain], [row.n_test, row.gain_total_nats, row.percent_of_total_gain])
    assert sum(count for count, _ in regional_totals.values()) == len(primary)
    close(sum(gain for _, gain in regional_totals.values()), total_gain)
    expected_composition_keys = {(lower, upper, year) for lower, upper in ((5, 5.5), (5.5, 6), (6, 6.5))
                                 for year in cohort.year.unique()}
    assert len(composition) == len(expected_composition_keys)
    assert set(zip(composition.baseline_pace_lower, composition.baseline_pace_upper, composition.year)) == expected_composition_keys
    for _, row in composition.iterrows():
        pace_bin = cohort.loc[(cohort.partition == "train") & (cohort.ability >= row.baseline_pace_lower)
                              & (cohort.ability < row.baseline_pace_upper)]
        year_bin = pace_bin.loc[pace_bin.year == row.year]
        close([len(year_bin), len(year_bin) / len(pace_bin), year_bin.raw_ratio.quantile(.25), year_bin.raw_ratio.median()],
              [row.n_train, row.share_of_bin, row.raw_ratio_q25, row.raw_ratio_median])
    macros_checked = verify_macros(args.fig_dir / "audit_numbers.tex", meta, cohort, hygiene, comparisons, diagnostics, partition)
    result = {"status": "passed", "pointwise_rows_checked": len(scores), "maximum_log_density_error": maximum_error,
              "comparison_rows_checked": len(comparisons), "references_rebuilt_from_training_only": list(references),
              "mixtures_checked": len(modes), "all_mixtures_have_one_derivative_root": True, "derivative_roots": modes,
              "responsibility_crossovers_checked": len(diagnostics) * 2, "gain_partition_rows_checked": len(partition),
              "year_composition_rows_checked": len(composition),
              "latex_macros_checked": macros_checked,
              "scope": "Independent split/reference/score/SE/response/tail/bulk/macro recomputation; numerical mode cross-check, not proof of global optimization or an audit of raw data collection."}
    (out / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "derivative_roots"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
