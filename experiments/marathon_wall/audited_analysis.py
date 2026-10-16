"""A reproducible, held-out density comparison for HW1 Question 3.

Run from the repository root with::

    python -m experiments.marathon_wall.audited_analysis --data-dir data/boston

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
                                         quantile=quantile, n_bins=40)
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


def make_macros(frame, train, comparisons, fits, output):
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
    for variant, suffix in (("raw", "Raw"), ("q10", "Ten"), ("q50", "Fifty")):
        row = table.loc[(variant, "mixture_minus_skewnormal")]
        macros[f"auditGain{suffix}"] = f"{row['gain_total_nats']:.1f}"
        macros[f"auditSe{suffix}"] = f"{row['se_total_nats']:.1f}"
    year_row = comparisons.loc[(comparisons["variant"] == "q25")
                               & (comparisons["group"] == "2014 only")
                               & (comparisons["comparison"] == "mixture_minus_skewnormal")].iloc[0]
    macros["auditGainYearFourteen"] = f"{year_row['gain_total_nats']:.1f}"
    macros["auditSeYearFourteen"] = f"{year_row['se_total_nats']:.1f}"
    modes = [g["parameters"]["mixture"]["density_modes"] for g in fits["q25"].values()]
    macros["auditUnimodalGroups"] = str(sum(mode == 1 for mode in modes))
    macros["auditGroups"] = str(len(modes))
    output.write_text("% Generated by audited_analysis.py; MLE held-out results.\n" + "\n".join(
        f"\\newcommand{{\\{key}}}{{{value.replace(',', '{,}')}}}" for key, value in macros.items()) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/boston"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/marathon_wall/audit_outputs"))
    parser.add_argument("--fig-dir", type=Path, default=Path("homeworks/figures"))
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
    comparisons = pd.DataFrame(comparison_rows)
    comparisons.to_csv(args.out_dir / "comparisons.csv", index=False)
    all_scores = pd.concat(score_frames, ignore_index=True)
    all_scores.to_csv(args.out_dir / "heldout_scores.csv", index=False)
    raw_n = int(hygiene.loc[hygiene["step"] == "rows in raw file", "n_before"].sum())
    metadata = {"seed": SEED, "train_fraction": .8, "n_raw": raw_n, "n_excluded": raw_n - len(frame),
                "n": len(frame), "n_train": int(train.sum()),
                "n_test": int((~train).sum()), "method": "maximum likelihood, per sex/year group",
                "primary_variant": "q25", "metric": "log(pace 30-40km / pace 10-25km)",
                "reference": "40-bin conditional quantile, fit only on training records",
                "uncertainty": "paired held-out log-score SE, conditional on the trained fits and treating records as independent",
                "limitations": ["Finisher-only selection; 2013 race interruption", "No inference about biological states or wall prevalence",
                                "Random holdout measures within-year prediction, not a future race", "Repeat runners across years are not identified",
                                "2014-only score rows retain the training reference estimated using both years",
                                "The existing all-data draft informed this audit; the holdout is an internal refitting diagnostic, not pristine confirmation",
                                "Sensitivity variants share the same heldout set and are not independent confirmations",
                                "One split does not quantify training-set or model-selection uncertainty"],
                "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                             "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
                "inputs_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path, _ in sources.values()},
                "fits": fitted_variants}
    (args.out_dir / "results.json").write_text(json.dumps(metadata, indent=2) + "\n")
    make_macros(frame, train, comparisons, fitted_variants, args.fig_dir / "audit_numbers.tex")
    print(comparisons.loc[comparisons["group"] == "all"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
