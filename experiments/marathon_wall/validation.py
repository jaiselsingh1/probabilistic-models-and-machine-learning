"""Archived exploratory checks for the Bayesian pacing models.

The current submission uses :mod:`.audited_analysis`.  These routines preserve
the original simulations and descriptive comparisons.  Predictive superiority
over a skew-normal does not establish a distinct physiological state; one
simulated null dataset does not calibrate a test across all unimodal densities.
Synthetic year effects check recovery under the chosen generating model, not
the causal effect of weather or the power of a general test.  Pace-matched
trajectory differences are associations subject to selection, measurement error,
and conditioning effects, even when segments do not overlap the metric windows.

The legacy ``smyth_rule`` name is retained for compatibility, but that rule does
NOT implement Smyth (2021).  Historical outputs using it are misattributed and
are excluded from the current submission's evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence
import warnings

import arviz as az
import numpy as np
import pandas as pd

from .data import CHECKPOINT_KM
from .diagnostics import compare_models, posterior_responsibility
from .metric import MetricWindows, raw_log_ratio
from .models import GroupIndex, build_model, sample_model

__all__ = [
    "wall_fraction_by_stratum",
    "smyth_rule",
    "compare_to_smyth_rule",
    "segment_is_disjoint",
    "trajectory_invariance",
    "early_shape_test",
    "simulate_year_shock",
    "simulate_skewnormal_null",
    "SMYTH_2021_WALL_RATE",
]

# Smyth (2021), PLoS ONE 16(5): e0251513,
# https://doi.org/10.1371/journal.pone.0251513, reports 28% of men and 17% of
# women in the main repeat-runner analysis.  Those are not universal benchmarks
# or rates for every record in the original 4.1M-record dataset.  His actual
# rule is >=25% segment slowdown relative to 5K-20K baseline over >=5 km after
# 20K.  The next two legacy constants preserve a different 20% window-average
# rule for compatibility with archived outputs; their names are historical.
SMYTH_2021_WALL_RATE: Mapping[str, float] = {"M": 0.28, "F": 0.17}
SMYTH_2021_SLOWDOWN: float = 0.20
SMYTH_2021_WINDOWS: MetricWindows = MetricWindows(baseline=("5k", "20k"), late=("25k", "finish"))


def wall_fraction_by_stratum(
    idata: az.InferenceData,
    groups: GroupIndex,
    *,
    hdi_prob: float = 0.94,
) -> pd.DataFrame:
    """Posterior summary of :math:`\\pi_g` for every group, with group sizes."""
    pi = idata.posterior["pi"].stack(sample=("chain", "draw")).transpose("sample", "group").to_numpy()
    lower, upper = np.quantile(pi, [(1 - hdi_prob) / 2, 1 - (1 - hdi_prob) / 2], axis=0)
    return pd.DataFrame(
        {
            "group": list(groups.labels),
            "n": groups.counts().to_numpy(),
            "pi_mean": pi.mean(axis=0),
            "pi_lo": lower,
            "pi_hi": upper,
        }
    )


def segment_is_disjoint(segment: str, windows: MetricWindows = MetricWindows()) -> bool:
    """Is this segment free of overlap with either window that defines ``y``?

    Segments *inside* a window are mechanically tied to the metric: conditional
    on the baseline pace, running 10K-20K relatively fast forces Half-25K
    relatively slow, so a large standardised difference there says nothing about
    physiology.  Disjoint segments avoid this direct window overlap, but they
    can still share endpoint timing errors and conditioning-induced associations.
    """
    start, end = segment.split("->")
    lo, hi = CHECKPOINT_KM[start], CHECKPOINT_KM[end]
    for window_start, window_end in (windows.baseline, windows.late):
        if lo < CHECKPOINT_KM[window_end] and CHECKPOINT_KM[window_start] < hi:
            return False
    return True


def trajectory_invariance(
    frame: pd.DataFrame,
    responsibility: np.ndarray,
    segment_paces: pd.DataFrame,
    *,
    wall_threshold: float = 0.8,
    clean_threshold: float = 0.2,
    n_ability_bins: int = 10,
) -> pd.DataFrame:
    """Ability-matched standardised pace difference, segment by segment.

    Within each baseline-pace decile the segment pace is divided by the athlete's
    own baseline pace.  The reported
    statistic is the decile-size-weighted standardised mean difference between
    the putative wall class and the clean class.

    ``disjoint`` marks absence of distance overlap with the metric windows;
    it does not establish statistical independence.  Baseline pace is measured
    during the race, may already reflect fatigue, and is matched only in coarse
    bins.  Differences can describe pacing associations but cannot identify
    glycogen depletion, fitness, or the causal effect of an aggressive start.
    """
    ability_bin = pd.qcut(frame["ability"], n_ability_bins, labels=False, duplicates="drop")
    is_wall = responsibility >= wall_threshold
    is_clean = responsibility <= clean_threshold

    rows: list[dict[str, object]] = []
    for segment in segment_paces.columns:
        normalised = (segment_paces[segment] / frame["ability"]).to_numpy(dtype=float)
        numerator = 0.0
        denominator = 0.0
        for b in np.unique(ability_bin.dropna()):
            member = (ability_bin == b).to_numpy()
            a = normalised[member & is_wall]
            c = normalised[member & is_clean]
            a = a[np.isfinite(a)]
            c = c[np.isfinite(c)]
            if a.size < 20 or c.size < 20:
                continue
            pooled = np.sqrt(0.5 * (a.var(ddof=1) + c.var(ddof=1)))
            if pooled <= 0.0:
                continue
            weight = float(a.size + c.size)
            numerator += weight * (a.mean() - c.mean()) / pooled
            denominator += weight
        rows.append(
            {
                "segment": segment,
                "smd": numerator / denominator if denominator > 0 else np.nan,
                "disjoint": segment_is_disjoint(str(segment)),
                "n_wall": int(is_wall.sum()),
                "n_clean": int(is_clean.sum()),
            }
        )
    return pd.DataFrame(rows)


def _fit_pair(y: np.ndarray, groups: GroupIndex, **sampler_kwargs: object) -> dict[str, az.InferenceData]:
    fitted: dict[str, az.InferenceData] = {}
    for name in ("mixture", "skewnormal"):
        fitted[name] = sample_model(build_model(name, y, groups), **sampler_kwargs)
    return fitted


def simulate_year_shock(
    *,
    pi_by_year: Mapping[int, float] = {2015: 0.18, 2016: 0.20, 2017: 0.33},
    n_per_year: int = 8000,
    mu0: float = 0.0,
    gap: float = 0.17,
    sigma0: float = 0.055,
    sigma1: float = 0.115,
    seed: int = 20170417,
    **sampler_kwargs: object,
) -> pd.DataFrame:
    """Check recovery of a synthetic year-level component-weight difference.

    Generates three synthetic races whose only difference is :math:`\\pi`, fits
    the hierarchical mixture with year as the grouping, and reports recovery.
    A single simulation checks this chosen parameter setting.  It does not
    estimate repeated-sampling coverage or show that a real year difference is
    caused by weather.
    """
    rng = np.random.default_rng(seed)
    years: list[int] = []
    draws: list[float] = []
    for year, pi in pi_by_year.items():
        z = rng.random(n_per_year) < pi
        values = np.where(
            z,
            rng.normal(mu0 + gap, sigma1, n_per_year),
            rng.normal(mu0, sigma0, n_per_year),
        )
        years.extend([year] * n_per_year)
        draws.extend(values.tolist())

    frame = pd.DataFrame({"year": years, "y": draws})
    groups = GroupIndex.from_columns(frame, ["year"])
    idata = sample_model(build_model("mixture", frame["y"].to_numpy(), groups), **sampler_kwargs)

    summary = wall_fraction_by_stratum(idata, groups)
    summary["pi_true"] = [pi_by_year[int(label)] for label in summary["group"]]
    summary["covered"] = (summary["pi_lo"] <= summary["pi_true"]) & (summary["pi_true"] <= summary["pi_hi"])
    return summary


def simulate_skewnormal_null(
    *,
    n: int = 20000,
    xi: float = -0.02,
    omega: float = 0.13,
    alpha: float = 4.0,
    seed: int = 20150420,
    **sampler_kwargs: object,
) -> pd.DataFrame:
    """Compare the models on one synthetic skew-normal dataset.

    Draws from a single skew-normal -- no latent regime anywhere -- and runs the
    same LOO comparison.  This is a diagnostic for the selected generating
    distribution and seed, not evidence that model selection distinguishes
    physiological regimes or is calibrated for every unimodal alternative.
    """
    from scipy import stats

    rng = np.random.default_rng(seed)
    y = stats.skewnorm.rvs(a=alpha, loc=xi, scale=omega, size=n, random_state=rng)
    frame = pd.DataFrame({"stratum": np.repeat(["a", "b"], n // 2)[: y.size]})
    groups = GroupIndex.from_columns(frame, ["stratum"])
    fitted = _fit_pair(np.asarray(y, dtype=float), groups, **sampler_kwargs)
    table = compare_models(fitted)
    table.index.name = "model"
    table.insert(0, "truth", "skew-normal (unimodal)")
    return table.reset_index()


def early_shape_test(
    frame: pd.DataFrame,
    responsibility: np.ndarray,
    segment_paces: pd.DataFrame,
    *,
    wall_threshold: float = 0.8,
    clean_threshold: float = 0.2,
    n_ability_bins: int = 10,
) -> pd.DataFrame:
    """Did the two classes already differ over the Hopkinton descent?

    Uses ``e_i = log(p_{5-10K} / p_{0-5K})``.  Its distance window does not
    overlap either metric window, although the 10K endpoint and conditioning on
    baseline pace can still induce dependence.  A positive SMD means the
    higher-responsibility group has a larger early pace ratio; this observation
    does not identify muscle damage or glycogen depletion.
    """
    if "start->5k" not in segment_paces.columns or "5k->10k" not in segment_paces.columns:
        return pd.DataFrame(columns=["statistic", "smd", "n_wall", "n_clean"])

    shape = np.log(segment_paces["5k->10k"].to_numpy(dtype=float)
                   / segment_paces["start->5k"].to_numpy(dtype=float))
    ability_bin = pd.qcut(frame["ability"], n_ability_bins, labels=False, duplicates="drop")
    is_wall = responsibility >= wall_threshold
    is_clean = responsibility <= clean_threshold

    numerator = 0.0
    denominator = 0.0
    for b in np.unique(ability_bin.dropna()):
        member = (ability_bin == b).to_numpy()
        a = shape[member & is_wall]
        c = shape[member & is_clean]
        a, c = a[np.isfinite(a)], c[np.isfinite(c)]
        if a.size < 20 or c.size < 20:
            continue
        pooled = np.sqrt(0.5 * (a.var(ddof=1) + c.var(ddof=1)))
        if pooled <= 0.0:
            continue
        weight = float(a.size + c.size)
        numerator += weight * (a.mean() - c.mean()) / pooled
        denominator += weight
    return pd.DataFrame(
        [
            {
                "statistic": "log(pace 5-10K / pace 0-5K)",
                "smd": numerator / denominator if denominator > 0 else np.nan,
                "n_wall": int(is_wall.sum()),
                "n_clean": int(is_clean.sum()),
            }
        ]
    )


def smyth_rule(
    frame: pd.DataFrame,
    *,
    slowdown: float = SMYTH_2021_SLOWDOWN,
    windows: MetricWindows = SMYTH_2021_WINDOWS,
) -> pd.Series:
    """Archived 20% window-average slowdown rule, NOT Smyth's published rule.

    The default compares 25K-finish average pace with 5K-20K average pace.
    Smyth instead uses >=25% slowdown over >=5 km of segments after 20K.
    The function name, defaults, and return type are retained so archived
    outputs remain reproducible; do not report them as a replication of Smyth.

    Returns nullable booleans; missing endpoints produce undecidable calls.
    """
    warnings.warn(
        "smyth_rule is an archived 20% window-average rule, not Smyth (2021)'s "
        "published 25% segment/distance rule; do not use it as that benchmark.",
        UserWarning,
        stacklevel=2,
    )
    ratio = raw_log_ratio(frame, windows)
    called = ratio >= np.log1p(slowdown)
    return called.where(ratio.notna()).astype("boolean")


def compare_to_smyth_rule(
    frame: pd.DataFrame,
    responsibility: np.ndarray,
    *,
    loss_ratio: float = 1.0,
) -> pd.DataFrame:
    """Compare the archived window-average rule and mixture calls by sex.

    Reports each rule's incidence, how often they agree, and Cohen's kappa -- a
    high incidence match with a low kappa would mean the two rules select
    equally many athletes but not the same ones, which is the failure mode worth
    knowing about.  Legacy output column names are retained for compatibility;
    ``smyth_rule_rate_here`` is NOT an implementation of the published rule and
    cannot validate agreement with Smyth's reported prevalence.
    """
    from .diagnostics import bayes_decision

    rule = smyth_rule(frame)
    model = bayes_decision(responsibility, loss_ratio=loss_ratio)
    usable = np.isfinite(responsibility) & rule.notna().to_numpy()

    rows: list[dict[str, object]] = []
    for sex in sorted(pd.unique(frame["sex"].astype("string").dropna())):
        member = (frame["sex"] == sex).to_numpy() & usable
        if member.sum() == 0:
            continue
        a = rule.to_numpy(dtype="object")[member].astype(bool)
        b = model[member].astype(bool)
        agree = float(np.mean(a == b))
        p_a, p_b = float(a.mean()), float(b.mean())
        chance = p_a * p_b + (1 - p_a) * (1 - p_b)
        rows.append(
            {
                "sex": str(sex),
                "n": int(member.sum()),
                "smyth_rule_rate_here": p_a,
                "smyth_published_rate": SMYTH_2021_WALL_RATE.get(str(sex), np.nan),
                "bayes_rate": p_b,
                "agreement": agree,
                "cohens_kappa": (agree - chance) / (1 - chance) if chance < 1 else np.nan,
            }
        )
    return pd.DataFrame(rows)
