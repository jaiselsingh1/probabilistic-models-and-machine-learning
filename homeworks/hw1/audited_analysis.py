"""A reproducible, held-out density comparison for HW1 Question 3.

Run from the repository root with::

    python -m homeworks.hw1.audited_analysis

The estimand is predictive density for eligible finisher records from these
sex/year strata, conditional on a fitted training set. No mixture component is
interpreted as a biological state. All parameters are maximum-likelihood
estimates; reported score standard errors are conditional on the trained fits,
not posterior intervals, repeated-split uncertainty, or uncertainty across races.
Only numpy, pandas, scipy and matplotlib are required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import optimize, signal, special, stats

from .data import SCHEMAS, load_races
from .metric import MetricWindows, TerrainReference, segment_pace

SEED = 20261016
REFERENCE_BINS = 40
REFERENCE_MIN_PER_BIN = 50
GAIN_QUANTILES = (.05, .95)
MODEL_NAMES = ("gaussian", "skewnormal", "mixture")
COLORS = {"gaussian": "#7d8793", "skewnormal": "#c17c22", "mixture": "#236c8c"}


def observed_metric(tidy: pd.DataFrame) -> pd.DataFrame:
    """Select real endpoint reads and compute raw ratios without fitting anything."""
    windows = MetricWindows()
    complete = np.ones(len(tidy), dtype=bool)
    for checkpoint in windows.required:
        complete &= tidy[f"t_{checkpoint}"].notna().to_numpy()
        complete &= ~tidy[f"imp_{checkpoint}"].to_numpy(dtype=bool)
    frame = tidy.loc[complete].copy().reset_index(drop=True)
    frame["ability"] = segment_pace(frame, *windows.baseline)
    frame["late_pace"] = segment_pace(frame, *windows.late)
    good = (frame["ability"] > 0) & (frame["late_pace"] > 0)
    frame = frame.loc[good].copy().reset_index(drop=True)
    frame["raw_ratio"] = np.log(frame["late_pace"] / frame["ability"])
    frame["group"] = frame["sex"].astype(str) + " x " + frame["year"].astype(str)
    frame["row_id"] = np.arange(len(frame))
    return frame


def stratified_split(frame: pd.DataFrame, seed: int = SEED) -> np.ndarray:
    """An 80/20 split made before any outcome-dependent reference is estimated."""
    rng = np.random.default_rng(seed)
    train = np.zeros(len(frame), dtype=bool)
    for group in sorted(frame["group"].unique()):
        ix = np.flatnonzero((frame["group"] == group).to_numpy())
        ix = rng.permutation(ix)
        train[ix[: int(np.floor(0.8 * len(ix)))]] = True
    return train


def log_density(name: str, y: np.ndarray, p: dict) -> np.ndarray:
    if name == "gaussian":
        return stats.norm.logpdf(y, p["mu"], p["sigma"])
    if name == "skewnormal":
        return stats.skewnorm.logpdf(y, p["shape"], p["loc"], p["scale"])
    return np.logaddexp(
        np.log1p(-p["weight_upper"]) + stats.norm.logpdf(y, p["mu_lower"], p["sigma_lower"]),
        np.log(p["weight_upper"]) + stats.norm.logpdf(y, p["mu_upper"], p["sigma_upper"]),
    )


def density_cdf(name: str, y: np.ndarray, p: dict) -> np.ndarray:
    if name == "gaussian":
        return stats.norm.cdf(y, p["mu"], p["sigma"])
    if name == "skewnormal":
        return stats.skewnorm.cdf(y, p["shape"], p["loc"], p["scale"])
    return ((1 - p["weight_upper"]) * stats.norm.cdf(y, p["mu_lower"], p["sigma_lower"])
            + p["weight_upper"] * stats.norm.cdf(y, p["mu_upper"], p["sigma_upper"]))


def fit_gaussian(y: np.ndarray) -> dict:
    return {"mu": float(np.mean(y)), "sigma": float(np.std(y)), "converged": True}


def fit_skewnormal(y: np.ndarray) -> dict:
    """Multistart likelihood optimization on standardized data."""
    center, scale = float(np.mean(y)), float(np.std(y))
    z = (y - center) / scale

    def objective(p):
        a, loc, log_scale = p
        sd = np.exp(log_scale)
        t = (z - loc) / sd
        return float(-np.mean(np.log(2.0) - log_scale - 0.5 * t * t
                              - 0.5 * np.log(2 * np.pi) + special.log_ndtr(a * t)))

    fits = []
    for shape in (-5.0, -1.0, 0.0, 1.0, 3.0, 8.0):
        delta = shape / np.sqrt(1 + shape * shape)
        omega = 1 / np.sqrt(1 - 2 * delta * delta / np.pi)
        xi = -omega * delta * np.sqrt(2 / np.pi)
        fit = optimize.minimize(objective, [shape, xi, np.log(omega)], method="L-BFGS-B",
                                bounds=[(-100, 100), (-20, 20), (-7, 4)],
                                options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-7})
        if fit.success and np.isfinite(fit.fun):
            fits.append(fit)
    if not fits:
        raise RuntimeError("No skew-normal optimization converged")
    best = min(fits, key=lambda fit: fit.fun)
    return {"shape": float(best.x[0]), "loc": float(center + scale * best.x[1]),
            "scale": float(scale * np.exp(best.x[2])), "converged": True,
            "successful_starts": len(fits), "attempted_starts": 6,
            "optimizer_message": str(best.message)}


def fit_mixture(y: np.ndarray) -> dict:
    """Multistart EM with an explicit small component-scale floor.

    The normal-mixture likelihood is unbounded without a scale restriction.
    The floor is 2% of training SD (at least 0.0001 log-pace units); the winning
    fit records whether it is active. Components are ordered after fitting.
    """
    center, scale = float(np.mean(y)), float(np.std(y))
    z = (y - center) / scale
    floor = max(1e-4 / scale, 0.02)
    candidates = []
    starts = ((0.15, 0.55, 1.2), (0.3, 0.7, 1.1), (0.5, 0.8, 0.8),
              (0.75, 1.1, 0.5), (0.1, 0.9, 1.8), (0.4, 0.4, 1.4))
    for weight, sd0, sd1 in starts:
        mu0, mu1 = float(np.quantile(z, .3)), float(np.quantile(z, .85))
        previous = -np.inf
        converged = False
        for iteration in range(1, 2001):
            l0 = np.log1p(-weight) - np.log(sd0) - .5 * ((z - mu0) / sd0) ** 2
            l1 = np.log(weight) - np.log(sd1) - .5 * ((z - mu1) / sd1) ** 2
            total = np.logaddexp(l0, l1)
            ll = float(np.mean(total))
            if iteration > 1 and abs(ll - previous) < 1e-9:
                converged = True
                break
            r = np.exp(l1 - total)
            weight = float(np.clip(r.mean(), 1e-6, 1 - 1e-6))
            mu0 = float(np.sum((1 - r) * z) / np.sum(1 - r))
            mu1 = float(np.sum(r * z) / np.sum(r))
            sd0 = max(floor, float(np.sqrt(np.sum((1 - r) * (z - mu0) ** 2) / np.sum(1 - r))))
            sd1 = max(floor, float(np.sqrt(np.sum(r * (z - mu1) ** 2) / np.sum(r))))
            previous = ll
        if converged:
            if mu0 > mu1:
                mu0, mu1, sd0, sd1, weight = mu1, mu0, sd1, sd0, 1 - weight
            candidates.append((ll, {"mu_lower": center + scale * mu0, "mu_upper": center + scale * mu1,
                                    "sigma_lower": scale * sd0, "sigma_upper": scale * sd1,
                                    "weight_upper": weight, "iterations": iteration}))
    if not candidates:
        raise RuntimeError("No mixture EM start converged within 2000 iterations")
    _, p = max(candidates, key=lambda candidate: candidate[0])
    p.update(converged=True, successful_starts=len(candidates), attempted_starts=len(starts),
             mean_log_likelihood=float(np.mean(log_density("mixture", y, p))),
             converged_start_objective_range=float(max(v for v, _ in candidates) - min(v for v, _ in candidates)),
             scale_floor=scale * floor,
             floor_active=min(p["sigma_lower"], p["sigma_upper"]) <= scale * floor * 1.0001)
    p["density_modes"] = count_modes(p)
    # A wider upper component dominates both sufficiently extreme tails. This
    # is a distributional fact, not an observed count of physiological events.
    p["upper_component_dominates_both_extreme_tails"] = p["sigma_upper"] > p["sigma_lower"]
    return p


def count_modes(p: dict) -> int:
    """Count local maxima on a dense grid covering both component centers."""
    lo = min(p["mu_lower"] - 8 * p["sigma_lower"], p["mu_upper"] - 8 * p["sigma_upper"])
    hi = max(p["mu_lower"] + 8 * p["sigma_lower"], p["mu_upper"] + 8 * p["sigma_upper"])
    grid = np.linspace(lo, hi, 32769)
    maxima, _ = signal.find_peaks(np.exp(log_density("mixture", grid, p)))
    return int(len(maxima))


def mixture_crossover_diagnostics(p: dict) -> dict:
    """Solve weighted component equality, not equality of unweighted PDFs.

    The upper component has posterior responsibility above one half exactly
    where this quadratic log odds is positive. When it is wider and has two
    crossings, this holds below the left root and above the right root. A
    negative reference-centered response is not necessarily a raw speedup.
    """
    s0, s1 = p["sigma_lower"], p["sigma_upper"]
    m0, m1, w = p["mu_lower"], p["mu_upper"], p["weight_upper"]
    a = .5 * (s0 ** -2 - s1 ** -2)
    b = m1 / s1 ** 2 - m0 / s0 ** 2
    c = np.log(w) - np.log1p(-w) + np.log(s0 / s1) + .5 * (m0 ** 2 / s0 ** 2 - m1 ** 2 / s1 ** 2)
    roots = np.roots([a, b, c]) if a != 0 else np.roots([b, c])
    real_roots = sorted(float(z.real) for z in roots if abs(z.imag) < 1e-10)
    two_roots = len(real_roots) == 2
    return {"sd_ratio": float(s1 / s0), "variance_ratio": float((s1 / s0) ** 2),
            "log_odds_quadratic": float(a), "log_odds_linear": float(b), "log_odds_constant": float(c),
            "n_real_crossovers": len(real_roots),
            "left_crossover_y": real_roots[0] if two_roots else None,
            "right_crossover_y": real_roots[1] if two_roots else None,
            "upper_component_dominates_both_extreme_tails": bool(a > 0)}


def training_gain_thresholds(frame, train, reference) -> dict:
    """Fix the bulk/tail boundaries from training responses within each group."""
    training = frame.loc[train].copy()
    training["y"] = training["raw_ratio"].to_numpy() - reference.predict(training["ability"])
    return {group: tuple(float(v) for v in np.quantile(sub["y"], GAIN_QUANTILES))
            for group, sub in training.groupby("group", observed=True)}


def score_gain_partition(frame, train, scores, reference) -> pd.DataFrame:
    """An exhaustive decomposition of the primary mixture-minus-skew score gain.

    Boundaries are each group's training response 5th/95th percentiles. The
    central interval includes its endpoints. Percentages use the *global* net
    gain as denominator, so negative contributions are retained, not clipped.
    This descriptive localization does not select or refit either model.
    """
    thresholds = training_gain_thresholds(frame, train, reference)
    total = float((scores["logp_mixture"] - scores["logp_skewnormal"]).sum())
    rows = []
    for group, sub in scores.groupby("group", observed=True):
        lower, upper = thresholds[group]
        values = sub["y"].to_numpy()
        masks = {"left": values < lower, "central": (values >= lower) & (values <= upper),
                 "right": values > upper}
        gains = (sub["logp_mixture"] - sub["logp_skewnormal"]).to_numpy()
        for region, mask in masks.items():
            gain = float(gains[mask].sum())
            rows.append({"group": group, "region": region, "n_test": int(mask.sum()),
                         "gain_total_nats": gain, "percent_of_total_gain": 100 * gain / total,
                         "training_q05": lower, "training_q95": upper})
    groups = pd.DataFrame(rows)
    for region in ("left", "central", "right"):
        sub = groups.loc[groups["region"] == region]
        gain = float(sub["gain_total_nats"].sum())
        rows.append({"group": "all", "region": region, "n_test": int(sub["n_test"].sum()),
                     "gain_total_nats": gain, "percent_of_total_gain": 100 * gain / total,
                     "training_q05": None, "training_q95": None})
    return pd.DataFrame(rows)


def training_year_composition(frame, train) -> pd.DataFrame:
    """Descriptive year mix near the plotted kink; not an estimate of censoring."""
    training = frame.loc[train]
    rows = []
    for lower, upper in ((5., 5.5), (5.5, 6.), (6., 6.5)):
        in_bin = training.loc[(training["ability"] >= lower) & (training["ability"] < upper)]
        for year in sorted(frame["year"].unique()):
            sub = in_bin.loc[in_bin["year"] == year, "raw_ratio"]
            rows.append({"baseline_pace_lower": lower, "baseline_pace_upper": upper, "year": int(year),
                         "n_train": len(sub), "share_of_bin": float(len(sub) / len(in_bin)) if len(in_bin) else None,
                         "raw_ratio_q25": float(sub.quantile(.25)) if len(sub) else None,
                         "raw_ratio_median": float(sub.median()) if len(sub) else None})
    return pd.DataFrame(rows)


def score_comparisons(scores: pd.DataFrame, variant: str) -> list[dict]:
    rows = []
    subsets = [("all", scores)]
    if "year" in scores.columns and (scores["year"] == 2014).any():
        subsets.append(("2014 only", scores.loc[scores["year"] == 2014]))
    subsets.extend((group, scores.loc[scores["group"] == group]) for group in sorted(scores["group"].unique()))
    for group, sub in subsets:
        for other in ("gaussian", "skewnormal"):
            differences = (sub["logp_mixture"] - sub[f"logp_{other}"]).to_numpy()
            n = len(differences)
            # The split fixes each stratum's size. Sum within-stratum variances
            # rather than treating the strata as an additional random sample.
            variance_sum = sum(len(s) * float(np.var(s["logp_mixture"] - s[f"logp_{other}"], ddof=1))
                               for _, s in sub.groupby("group", observed=True))
            se_mean = float(np.sqrt(variance_sum) / n)
            rows.append({"variant": variant, "group": group, "comparison": f"mixture_minus_{other}",
                         "n_test": n, "gain_total_nats": float(differences.sum()),
                         "se_total_nats": se_mean * n, "gain_per_runner_nats": float(differences.mean()),
                         "se_per_runner_nats": se_mean})
    return rows


def fit_variant(frame: pd.DataFrame, train: np.ndarray, quantile: float | None):
    reference = None
    y = frame["raw_ratio"].to_numpy().copy()
    if quantile is not None:
        reference = TerrainReference.fit(frame.loc[train, "raw_ratio"], frame.loc[train, "ability"],
                                         quantile=quantile, n_bins=REFERENCE_BINS,
                                         min_per_bin=REFERENCE_MIN_PER_BIN)
        y -= reference.predict(frame["ability"])
    fitted = {}
    scores = frame.loc[~train, ["row_id", "year", "sex", "bib", "group"]].copy()
    scores["y"] = y[~train]
    for name in MODEL_NAMES:
        scores[f"logp_{name}"] = np.nan
        scores[f"pit_{name}"] = np.nan
    for group in sorted(frame["group"].unique()):
        member = (frame["group"] == group).to_numpy()
        training = y[train & member]
        params = {"gaussian": fit_gaussian(training), "skewnormal": fit_skewnormal(training),
                  "mixture": fit_mixture(training)}
        fitted[group] = {"n_train": int(np.sum(train & member)), "n_test": int(np.sum(~train & member)),
                         "parameters": params}
        heldout = (scores["group"] == group).to_numpy()
        values = scores.loc[heldout, "y"].to_numpy()
        for name, p in params.items():
            scores.loc[heldout, f"logp_{name}"] = log_density(name, values, p)
            scores.loc[heldout, f"pit_{name}"] = density_cdf(name, values, p)
        p = params["mixture"]
        log_upper = np.log(p["weight_upper"]) + stats.norm.logpdf(values, p["mu_upper"], p["sigma_upper"])
        scores.loc[heldout, "responsibility_upper"] = np.exp(log_upper - log_density("mixture", values, p))
    if not np.isfinite(scores[[f"logp_{name}" for name in MODEL_NAMES]].to_numpy()).all():
        raise RuntimeError("Nonfinite held-out log score")
    return fitted, scores, reference


def fit_year_variant(frame: pd.DataFrame, train: np.ndarray, year: int, quantile: float = .25):
    """Refit both reference and models within a year, preserving the split.

    Keep row_id from the pooled cohort so the independent verifier can trace
    every held-out record back to the original split. No records are resplit.
    """
    member = (frame["year"] == year).to_numpy()
    subset = frame.loc[member].copy().reset_index(drop=True)
    return fit_variant(subset, train[member], quantile)


def make_figure(frame, train, fitted, scores, reference, output):
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
    fig, axs = plt.subplots(2, 2, figsize=(6.5, 3.4), constrained_layout=True)
    ax = axs[0, 0]
    training = frame.loc[train].copy()
    training["bin"] = pd.qcut(training["ability"], 25, labels=False, duplicates="drop")
    bins = training.groupby("bin", observed=True)
    x = bins["ability"].median()
    quantiles = bins["raw_ratio"].quantile([.1, .5, .9]).unstack()
    ax.fill_between(x, quantiles[.1], quantiles[.9], color="#dce9ef", label="10–90% in training bins")
    ax.plot(x, quantiles[.5], color="#236c8c", label="Training median")
    ax.plot(reference.ability_knots, reference.delta_knots, "--", color="#a35731", label="Training 25% reference")
    ax.set(xlabel="Baseline pace, 10–25 km (min/km)", ylabel="Raw log pace ratio",
           title="(a) Slowing varies with baseline pace")
    ax.legend(fontsize=7, frameon=False)
    for ax, sex, panel in ((axs[0, 1], "F", "b"), (axs[1, 0], "M", "c")):
        sub = scores.loc[scores["sex"] == sex]
        lower, upper = np.quantile(sub["y"], [.002, .995])
        grid = np.linspace(min(-.1, lower), upper, 800)
        # Full-data histogram normalization; zooming does not renormalize the tails away.
        edges = np.linspace(sub["y"].min(), sub["y"].max(), 110)
        ax.hist(sub["y"], bins=edges, density=True, color="#dce9ef", edgecolor="none", label="Held-out records")
        group_counts = sub.groupby("group").size()
        for name in MODEL_NAMES:
            density = sum(n / len(sub) * np.exp(log_density(name, grid, fitted[g]["parameters"][name]))
                          for g, n in group_counts.items())
            ax.plot(grid, density, color=COLORS[name], label=name.replace("skewnormal", "skew-normal"))
        ax.set(xlim=(grid.min(), grid.max()), xlabel="Reference-centered log pace ratio", ylabel="Density",
               title=f"({panel}) Held-out {sex} records (n={len(sub):,})")
        ax.legend(fontsize=7, frameon=False)
    ax = axs[1, 1]
    u_grid = np.linspace(0, 1, 301)
    for name in MODEL_NAMES:
        pit = np.sort(scores[f"pit_{name}"].to_numpy())
        departure = np.searchsorted(pit, u_grid, side="right") / len(pit) - u_grid
        ax.plot(u_grid, departure, color=COLORS[name], label=name.replace("skewnormal", "skew-normal"))
    ax.axhline(0, color="black", lw=.7)
    ax.set(xlabel="Held-out predictive CDF value u", ylabel="Empirical CDF(u) − u",
           title="(d) Held-out calibration (zero is ideal)")
    # Model colors share the explicit legends in panels (b) and (c).
    # Avoid placing a third legend over the calibration curves.
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def make_macros(frame, train, comparisons, fits, hygiene, mixture_diagnostics, gain_partition, output):
    table = comparisons.loc[comparisons["group"] == "all"].set_index(["variant", "comparison"])
    primary_skew = table.loc[("q25", "mixture_minus_skewnormal")]
    primary_gauss = table.loc[("q25", "mixture_minus_gaussian")]
    macros = {"auditN": f"{len(frame):,}", "auditTrainN": f"{train.sum():,}", "auditTestN": f"{(~train).sum():,}",
              "auditGainSkew": f"{primary_skew['gain_total_nats']:.1f}",
              "auditSeSkew": f"{primary_skew['se_total_nats']:.1f}",
              "auditGainSkewPerRunner": f"{primary_skew['gain_per_runner_nats']:.4f}",
              "auditSeSkewPerRunner": f"{primary_skew['se_per_runner_nats']:.4f}",
              "auditGainGauss": f"{primary_gauss['gain_total_nats']:.1f}",
              "auditSeGauss": f"{primary_gauss['se_total_nats']:.1f}"}
    raw_rows = hygiene.loc[hygiene["step"] == "rows in raw file"]
    raw_n = int(raw_rows["n_before"].sum())
    macros.update(auditRawN=f"{raw_n:,}", auditExcludedN=f"{raw_n - len(frame):,}",
                  auditSeed=str(SEED), auditReferenceBins=str(REFERENCE_BINS))
    for year, suffix in ((2013, "Thirteen"), (2014, "Fourteen")):
        macros[f"auditRawYear{suffix}"] = f"{int(raw_rows.loc[raw_rows['race'] == f'llimllib-{year}', 'n_before'].sum()):,}"
    for suffix, step in (("Wheelchair", "drop wheelchair / handcycle / push-rim"),
                         ("Duplicate", "drop duplicate bibs"),
                         ("PaceExcluded", "drop segment pace outside"),
                         ("EndpointExcluded", "require real positive metric endpoints")):
        macros[f"audit{suffix}N"] = f"{int(hygiene.loc[hygiene['step'].str.startswith(step), 'dropped'].sum()):,}"
    macros["auditNonmonotoneN"] = f"{int(hygiene.groupby('race')['blanked_mats'].first().sum()):,}"
    for variant, suffix in (("raw", "Raw"), ("q10", "Ten"), ("q50", "Fifty")):
        row = table.loc[(variant, "mixture_minus_skewnormal")]
        macros[f"auditGain{suffix}"] = f"{row['gain_total_nats']:.1f}"
        macros[f"auditSe{suffix}"] = f"{row['se_total_nats']:.1f}"
    year_row = comparisons.loc[(comparisons["variant"] == "q25")
                               & (comparisons["group"] == "2014 only")
                               & (comparisons["comparison"] == "mixture_minus_skewnormal")].iloc[0]
    macros["auditGainYearFourteen"] = f"{year_row['gain_total_nats']:.1f}"
    macros["auditSeYearFourteen"] = f"{year_row['se_total_nats']:.1f}"
    separate = table.loc[("q25_2014", "mixture_minus_skewnormal")]
    macros["auditGainYearFourteenSeparate"] = f"{separate['gain_total_nats']:.1f}"
    macros["auditSeYearFourteenSeparate"] = f"{separate['se_total_nats']:.1f}"
    for field, suffix in (("sd_ratio", "SdRatio"), ("variance_ratio", "VarianceRatio"),
                          ("left_crossover_y", "LeftCrossover")):
        macros[f"audit{suffix}Min"] = f"{mixture_diagnostics[field].min():.3f}"
        macros[f"audit{suffix}Max"] = f"{mixture_diagnostics[field].max():.3f}"
    for group, suffix in (("F x 2013", "FemaleThirteen"), ("F x 2014", "FemaleFourteen"),
                          ("M x 2013", "MaleThirteen"), ("M x 2014", "MaleFourteen")):
        value = mixture_diagnostics.loc[mixture_diagnostics["group"] == group, "left_crossover_y"].iloc[0]
        macros[f"auditLeftCrossover{suffix}"] = f"{value:.3f}"
    for region, suffix in (("central", "Bulk"), ("right", "RightTail"), ("left", "LeftTail")):
        row = gain_partition.loc[(gain_partition["group"] == "all") & (gain_partition["region"] == region)].iloc[0]
        macros[f"auditGain{suffix}"] = f"{row['gain_total_nats']:.1f}"
        macros[f"auditGain{suffix}Percent"] = f"{row['percent_of_total_gain']:.1f}"
        macros[f"audit{suffix}N"] = f"{int(row['n_test']):,}"
    macros["auditBulkLowerPercentile"] = str(round(100 * GAIN_QUANTILES[0]))
    macros["auditBulkUpperPercentile"] = str(round(100 * GAIN_QUANTILES[1]))
    modes = [g["parameters"]["mixture"]["density_modes"] for g in fits["q25"].values()]
    macros["auditUnimodalGroups"] = str(sum(mode == 1 for mode in modes))
    macros["auditGroups"] = str(len(modes))
    output.write_text("% Generated by audited_analysis.py; MLE held-out results.\n" + "\n".join(
        f"\\newcommand{{\\{key}}}{{{value.replace(',', '{,}')}}}" for key, value in macros.items()) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data" / "boston")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--fig-dir", type=Path, default=Path(__file__).resolve().parents[1] / "figures")
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    sources = {year: (args.data_dir / f"results_{year}.csv", SCHEMAS["llimllib"])
               for year in (2013, 2014)}
    tidy, hygiene = load_races(sources)
    frame = observed_metric(tidy)
    train = stratified_split(frame)
    endpoint_steps = []
    for year in sorted(frame["year"].unique()):
        before = int((tidy["year"] == year).sum())
        after = int((frame["year"] == year).sum())
        endpoint_steps.append({"race": f"llimllib-{year}", "step": "require real positive metric endpoints",
                               "n_before": before, "n_after": after, "dropped": before - after})
    hygiene = pd.concat([hygiene, pd.DataFrame(endpoint_steps)], ignore_index=True)
    hygiene.to_csv(args.out_dir / "hygiene.csv", index=False)
    frame[["row_id", "year", "sex", "bib", "group", "ability", "raw_ratio"]].assign(
        partition=np.where(train, "train", "test")).to_csv(args.out_dir / "cohort.csv", index=False)
    fitted_variants, score_frames, comparison_rows, references = {}, [], [], {}
    for variant, quantile in (("q25", .25), ("raw", None), ("q10", .1), ("q50", .5)):
        print(f"Fitting {variant} ...", flush=True)
        fitted, scores, reference = fit_variant(frame, train, quantile)
        fitted_variants[variant] = fitted
        score_frames.append(scores.assign(variant=variant))
        comparison_rows.extend(score_comparisons(scores, variant))
        if reference is not None:
            reference.to_frame().to_csv(args.out_dir / f"reference_{variant}.csv", index=False)
            references[variant] = reference
        if variant == "q25":
            make_figure(frame, train, fitted, scores, reference, args.fig_dir / "audit_wall.pdf")
    print("Fitting q25_2014 (2014-only reference and models, original split) ...", flush=True)
    fitted, scores, reference = fit_year_variant(frame, train, 2014)
    fitted_variants["q25_2014"] = fitted
    score_frames.append(scores.assign(variant="q25_2014"))
    comparison_rows.extend(score_comparisons(scores, "q25_2014"))
    reference.to_frame().to_csv(args.out_dir / "reference_q25_2014.csv", index=False)
    references["q25_2014"] = reference
    comparisons = pd.DataFrame(comparison_rows)
    comparisons.to_csv(args.out_dir / "comparisons.csv", index=False)
    all_scores = pd.concat(score_frames, ignore_index=True)
    all_scores.to_csv(args.out_dir / "heldout_scores.csv", index=False)
    mixture_diagnostics = pd.DataFrame([
        {"group": group, **mixture_crossover_diagnostics(fit["parameters"]["mixture"])}
        for group, fit in fitted_variants["q25"].items()])
    mixture_diagnostics.to_csv(args.out_dir / "mixture_diagnostics.csv", index=False)
    primary_scores = all_scores.loc[all_scores["variant"] == "q25"]
    gain_partition = score_gain_partition(frame, train, primary_scores, references["q25"])
    gain_partition.to_csv(args.out_dir / "score_gain_partition.csv", index=False)
    composition = training_year_composition(frame, train)
    composition.to_csv(args.out_dir / "training_year_composition.csv", index=False)
    diagnostics = {
        "variant": "q25", "comparison": "mixture_minus_skewnormal",
        "component_definition": "Lower and upper refer to fitted means, not physiological states.",
        "crossover_definition": "Roots of log[w*Normal(y;mu_upper,sigma_upper)/((1-w)*Normal(y;mu_lower,sigma_lower))]; responsibility_upper > 0.5 where log odds > 0.",
        "response_interpretation": "y < 0 means less slowing than the fitted baseline-pace reference; actual speeding up requires raw_ratio < 0.",
        "sd_ratio_definition": "sigma_upper / sigma_lower; the variance ratio is its square.",
        "gain_partition_definition": {
            "training_quantiles": list(GAIN_QUANTILES), "thresholds": "Separate within each primary sex/year training group.",
            "regions": {"left": "y < training q05", "central": "training q05 <= y <= training q95", "right": "y > training q95"},
            "percentage_denominator": "The total primary held-out mixture-minus-skewnormal net gain, pooled across groups.",
            "interpretation": "An exhaustive descriptive localization of net score gain; negative regional contributions are retained. It does not prove that mode height caused the gain."},
        "mixture": mixture_diagnostics.to_dict(orient="records"),
        "score_gain_partition": gain_partition.astype(object).where(pd.notna(gain_partition), None).to_dict(orient="records"),
        "year_composition_definition": "Training counts and raw log-ratio summaries in fixed baseline-pace intervals [5,5.5), [5.5,6), [6,6.5); these cannot identify the causal effect of race interruption.",
        "training_year_composition": composition.to_dict(orient="records")}
    (args.out_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    raw_n = int(hygiene.loc[hygiene["step"] == "rows in raw file", "n_before"].sum())
    metadata = {"seed": SEED, "train_fraction": .8, "n_raw": raw_n, "n_excluded": raw_n - len(frame),
                "n": len(frame), "n_train": int(train.sum()),
                "n_test": int((~train).sum()), "method": "maximum likelihood, per sex/year group",
                "primary_variant": "q25", "metric": "log(pace 30-40km / pace 10-25km)",
                "reference": f"{REFERENCE_BINS}-bin conditional quantile, fit only on training records",
                "reference_n_bins": REFERENCE_BINS, "reference_min_per_bin": REFERENCE_MIN_PER_BIN,
                "variant_training": {
                    variant: {"reference_quantile": quantile, "fit_years": years,
                              "reference_years": years if quantile is not None else []}
                    for variant, quantile, years in (("q25", .25, [2013, 2014]), ("raw", None, [2013, 2014]),
                                                     ("q10", .1, [2013, 2014]), ("q50", .5, [2013, 2014]),
                                                     ("q25_2014", .25, [2014]))},
                "uncertainty": "paired held-out log-score SE, conditional on the trained fits and treating records as independent",
                "limitations": ["Finisher-only selection; 2013 race interruption", "No inference about biological states or wall prevalence",
                                "Random holdout measures within-year prediction, not a future race", "Repeat runners across years are not identified",
                                "Within q25, the 2014-only score subset retains the pooled training reference; q25_2014 separately refits the reference and models using only original 2014 training records",
                                "The existing all-data draft informed this audit; the holdout is an internal refitting diagnostic, not pristine confirmation",
                                "Sensitivity variants share the same heldout set and are not independent confirmations",
                                "One split does not quantify training-set or model-selection uncertainty"],
                "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                             "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
                "inputs_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path, _ in sources.values()},
                "fits": fitted_variants}
    (args.out_dir / "results.json").write_text(json.dumps(metadata, indent=2) + "\n")
    make_macros(frame, train, comparisons, fitted_variants, hygiene, mixture_diagnostics,
                gain_partition, args.fig_dir / "audit_numbers.tex")
    print(comparisons.loc[comparisons["group"] == "all"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
