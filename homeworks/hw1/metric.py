"""Empirically centered late-to-baseline log pace ratios.

The default response is log[pace(30K--40K) / pace(10K--25K)] minus
an estimated conditional quantile given baseline pace. This is descriptive
centering, not an identified correction for terrain or physiology: baseline
pace and late pace also reflect fatigue, strategy, weather, and selection.
The late window includes part of the Newton Hills.

For predictive evaluation, fit TerrainReference on training records only and
pass that frozen reference when transforming validation records. A location
shift preserves shape at exactly fixed baseline pace, but different shifts
across observations can alter the pooled shape and number of modes. There is
no guarantee that this transformation is conservative for mixture detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Mapping, Sequence

import numpy as np
import pandas as pd

from .data import CHECKPOINT_KM

__all__ = [
    "MetricWindows",
    "TerrainReference",
    "minetti_cost_ratio",
    "MinettiProfile",
    "segment_pace",
    "raw_log_ratio",
    "build_metric",
    "segment_pace_table",
]

# Ordered checkpoint grid used for the per-segment views (Task 6 and check (c)).
_GRID: Final[tuple[str, ...]] = ("start", "5k", "10k", "15k", "20k", "half", "25k", "30k", "35k", "40k", "finish")


@dataclass(frozen=True)
class MetricWindows:
    """Checkpoints bounding the baseline and late-race windows."""

    baseline: tuple[str, str] = ("10k", "25k")
    late: tuple[str, str] = ("30k", "40k")

    @property
    def required(self) -> tuple[str, ...]:
        return (*self.baseline, *self.late)

    def __str__(self) -> str:
        b0, b1 = self.baseline
        l0, l1 = self.late
        return f"log[pace({l0}-{l1}) / pace({b0}-{b1})]"


def segment_pace(tidy: pd.DataFrame, start: str, end: str) -> pd.Series:
    """Mean pace (min/km) over ``start`` -> ``end``, as a cumulative-time difference."""
    distance = CHECKPOINT_KM[end] - CHECKPOINT_KM[start]
    if distance <= 0.0:
        raise ValueError(f"{end!r} does not come after {start!r}")
    return (tidy[f"t_{end}"] - tidy[f"t_{start}"]) / distance


def raw_log_ratio(tidy: pd.DataFrame, windows: MetricWindows = MetricWindows()) -> pd.Series:
    """Unadjusted late/baseline log pace ratio."""
    late = segment_pace(tidy, *windows.late)
    base = segment_pace(tidy, *windows.baseline)
    return np.log(late / base)


def minetti_cost_ratio(grade: np.ndarray | float) -> np.ndarray:
    """Energy cost of running at ``grade`` relative to level ground.

    Minetti et al. (2002), *J. Appl. Physiol.* 93:1039-1046, fit the metabolic
    cost of running Cr (J/kg/m) as a quintic in the gradient i = rise/run over
    -0.45 <= i <= 0.45.  Dividing by Cr(0) = 3.6 gives the factor by which a
    gradient inflates (or deflates) the pace an athlete holding constant
    metabolic power will run.
    """
    i = np.clip(np.asarray(grade, dtype=float), -0.45, 0.45)
    cost = 155.4 * i**5 - 30.4 * i**4 - 43.3 * i**3 + 46.3 * i**2 + 19.5 * i + 3.6
    return cost / 3.6


@dataclass(frozen=True)
class MinettiProfile:
    """Optional physiological grade adjustment from a surveyed elevation profile.

    ``elevation_m`` maps checkpoint names to metres above sea level.  Supply a
    profile measured off the certified course (or a GPX track); none is shipped
    with this module, because a wrong elevation table is worse than none at all.
    """

    elevation_m: Mapping[str, float]

    def window_factor(self, start: str, end: str) -> float:
        """Minetti cost ratio for the mean gradient of a window."""
        rise = self.elevation_m[end] - self.elevation_m[start]
        run = (CHECKPOINT_KM[end] - CHECKPOINT_KM[start]) * 1000.0
        return float(minetti_cost_ratio(rise / run))

    def log_offset(self, windows: MetricWindows) -> float:
        """Additive offset on the log-ratio scale implied by the two windows."""
        return float(np.log(self.window_factor(*windows.late) / self.window_factor(*windows.baseline)))


@dataclass
class TerrainReference:
    """Empirical baseline-pace reference :math:`\\hat\\delta_q(a)`.

    Estimated as the ``quantile``-th conditional quantile of the raw log ratio
    within ability bins, then linearly interpolated between bin centres and held
    flat beyond the outermost centres.
    """

    ability_knots: np.ndarray
    delta_knots: np.ndarray
    quantile: float
    n_bins: int

    @classmethod
    def fit(
        cls,
        raw_ratio: pd.Series,
        ability: pd.Series,
        *,
        quantile: float = 0.25,
        n_bins: int = 40,
        min_per_bin: int = 50,
    ) -> "TerrainReference":
        usable = raw_ratio.notna() & ability.notna()
        ratio_values = raw_ratio[usable].to_numpy(dtype=float)
        ability_values = ability[usable].to_numpy(dtype=float)

        edges = np.quantile(ability_values, np.linspace(0.0, 1.0, n_bins + 1))
        edges = np.unique(edges)
        index = np.clip(np.searchsorted(edges, ability_values, side="right") - 1, 0, len(edges) - 2)

        knots_a: list[float] = []
        knots_d: list[float] = []
        for b in range(len(edges) - 1):
            member = index == b
            if int(member.sum()) < min_per_bin:
                continue
            knots_a.append(float(np.median(ability_values[member])))
            knots_d.append(float(np.quantile(ratio_values[member], quantile)))
        if len(knots_a) < 2:
            raise ValueError("not enough populated ability bins to fit a terrain reference")

        order = np.argsort(knots_a)
        return cls(
            ability_knots=np.asarray(knots_a)[order],
            delta_knots=np.asarray(knots_d)[order],
            quantile=quantile,
            n_bins=n_bins,
        )

    def predict(self, ability: pd.Series | np.ndarray) -> np.ndarray:
        values = np.asarray(ability, dtype=float)
        return np.interp(values, self.ability_knots, self.delta_knots)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"ability_min_per_km": self.ability_knots, "delta": self.delta_knots})


def build_metric(
    tidy: pd.DataFrame,
    *,
    windows: MetricWindows = MetricWindows(),
    quantile: float = 0.25,
    n_bins: int = 40,
    require_observed: bool = True,
    reference: TerrainReference | None = None,
    minetti: MinettiProfile | None = None,
) -> tuple[pd.DataFrame, TerrainReference]:
    """Attach the empirically centered deceleration metric ``y`` to the tidy frame.

    ``require_observed`` insists that the four checkpoints bounding the two
    windows came off a real mat rather than the interior-gap interpolation in
    :mod:`.data`, so that no part of ``y`` is an artefact of the repair step.
    """
    frame = tidy.copy()
    needed = windows.required

    complete = np.ones(len(frame), dtype=bool)
    for checkpoint in needed:
        complete &= frame[f"t_{checkpoint}"].notna().to_numpy()
        if require_observed:
            complete &= ~frame[f"imp_{checkpoint}"].to_numpy(dtype=bool)
    frame = frame.loc[complete].reset_index(drop=True)

    frame["ability"] = segment_pace(frame, *windows.baseline)
    frame["late_pace"] = segment_pace(frame, *windows.late)
    frame["y_raw"] = np.log(frame["late_pace"] / frame["ability"])

    if minetti is not None:
        frame["y_raw"] = frame["y_raw"] - minetti.log_offset(windows)

    if reference is None:
        reference = TerrainReference.fit(frame["y_raw"], frame["ability"], quantile=quantile, n_bins=n_bins)
    frame["delta_hat"] = reference.predict(frame["ability"])
    frame["y"] = frame["y_raw"] - frame["delta_hat"]
    return frame, reference


def segment_pace_table(tidy: pd.DataFrame, grid: Sequence[str] | None = None) -> pd.DataFrame:
    """Per-segment pace (min/km) for every consecutive pair on the checkpoint grid."""
    present = [c for c in (grid or _GRID) if f"t_{c}" in tidy.columns]
    out = pd.DataFrame(index=tidy.index)
    for start, end in zip(present[:-1], present[1:]):
        out[f"{start}->{end}"] = segment_pace(tidy, start, end)
    return out
