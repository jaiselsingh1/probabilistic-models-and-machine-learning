"""External validation: is the second component a physiological state or an artefact?

Three checks, in increasing order of how hard they are to fake.

(a) *Predictive*.  PSIS-LOO Delta-ELPD against the unimodal skew-normal control,
    plus LOO-PIT calibration.  This only establishes that two components fit
    better than one skewed component -- necessary, nowhere near sufficient.

(b) *Exogenous shock*.  The model never sees weather.  If the fitted wall
    fraction nevertheless tracks a year that was physiologically harder, the
    component is responding to something real.  :func:`wall_fraction_by_stratum`
    extracts the fractions; :func:`simulate_year_shock` establishes that the
    estimator *can* recover such a spike at this sample size and effect size,
    so a null result would be informative rather than merely underpowered.

(c) *Trajectory invariance*.  A glycogen-depletion wall is a late event.
    Conditional on ability, athletes the model assigns to the wall component
    should be indistinguishable from the rest *before* 25K, and separate sharply
    after.  If instead they are slower or more erratic from the gun, the
    "component" is just a label for bad pacing or low fitness.

:func:`simulate_skewnormal_null` is the guard that makes (a) meaningful: data
generated from a genuinely unimodal skewed law must *not* produce a decisive
Delta-ELPD in favour of the mixture.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

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

# Smyth (2021), "How recreational marathon runners hit the wall", PLoS ONE
# 16(5): e0251513.  Over 4.1M performances he calls a wall when 25K-to-finish
# pace is >= 20% slower than 5K-20K pace, and reports an incidence of 28% of men
# and 17% of women.  Both the threshold and the windows are reproduced here so
# the rule can be run on this field rather than merely cited at it.
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
    physiology.  Only disjoint segments carry evidence, and the tied ones are
    retained purely as a positive control.
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

    Within each ability decile the segment pace is divided by the athlete's own
    baseline pace, so the comparison is of *shape*, not speed.  The reported
    statistic is the decile-size-weighted standardised mean difference between
    the putative wall class and the clean class.

    Matching on ability is what makes the early segments interpretable: within a
    stratum of equal 10K-25K pace, class membership is driven entirely by the
    30K-40K pace, so a difference over the first 10 km cannot have been induced
    by the metric.  ``disjoint`` marks the segments for which that argument
    holds; the rest are a positive control only.  Under a pure late-onset
    depletion hypothesis the disjoint early segments should show SMD ~ 0.
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
    """Can a year-level wall-fraction spike be recovered without weather covariates?

    Generates three synthetic races whose only difference is :math:`\\pi`, fits
    the hierarchical mixture with year as the grouping, and reports recovery.
    This calibrates check (b): it says what a real spike would have to look like
    before the estimator could see it.
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
    """Does the mixture win even when the truth is unimodal-but-skewed?

    Draws from a single skew-normal -- no latent regime anywhere -- and runs the
    same LOO comparison.  If the mixture still wins decisively here, a decisive
    Delta-ELPD on the real data would be worthless.
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

    Uses only the first 10 km -- the one statistic that is disjoint from both
    windows of ``y`` under any conditioning: ``e_i = log(p_{5-10K} / p_{0-5K})``,
    the degree to which an athlete eased off after the 140 m drop out of
    Hopkinton.  A positive SMD means the wall class attacked the descent harder
    and then eased more, which would implicate eccentric muscle damage rather
    than (or as well as) glycogen depletion.
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
    """Smyth's (2021) rule applied verbatim to this field.

    Deliberately uses his windows and his *unadjusted* ratio rather than the
    de-confounded metric: the point of the comparison is to see what a fixed
    threshold does on a course whose topography it never accounted for.

    Returns a nullable boolean: an athlete missing one of *his* four mats is
    undecidable by his rule, and must not be silently counted as a non-wall.
    """
    ratio = raw_log_ratio(frame, windows)
    called = ratio >= np.log1p(slowdown)
    return called.where(ratio.notna()).astype("boolean")


def compare_to_smyth_rule(
    frame: pd.DataFrame,
    responsibility: np.ndarray,
    *,
    loss_ratio: float = 1.0,
) -> pd.DataFrame:
    """Rule-based versus model-based wall calls, by sex.

    Reports each rule's incidence, how often they agree, and Cohen's kappa -- a
    high incidence match with a low kappa would mean the two rules select
    equally many athletes but not the same ones, which is the failure mode worth
    knowing about.
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
