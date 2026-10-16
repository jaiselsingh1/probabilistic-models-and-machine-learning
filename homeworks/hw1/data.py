"""Loading and hygiene for Boston Marathon 5K-checkpoint split files.

Two public BAA-derived schemas are supported:

``rojour``
    The Kaggle "Boston Marathon 2015/2016/2017" dumps.  Cumulative splits are
    ``HH:MM:SS`` strings at 5K, 10K, 15K, 20K, Half, 25K, 30K, 35K, 40K.

``llimllib``
    The ``llimllib/bostonmarathon`` scrape of baa.org.  Cumulative splits are
    decimal minutes and the 15K mat is absent.

Both encode an unread timing mat as ``"-"``.  Everything downstream consumes the
tidy frame produced by :func:`load_races`, so adding a third schema means adding
one :class:`SplitSchema` and nothing else.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "CHECKPOINT_KM",
    "SplitSchema",
    "SCHEMAS",
    "HygieneReport",
    "parse_clock_series",
    "load_race",
    "load_races",
]

# Cumulative distance (km) of each timing mat.  ``finish`` is the certified
# marathon distance; ``half`` is the certified half-marathon point, which is why
# it is not exactly 21.0.
CHECKPOINT_KM: Final[Mapping[str, float]] = {
    "start": 0.0,
    "5k": 5.0,
    "10k": 10.0,
    "15k": 15.0,
    "20k": 20.0,
    "half": 21.0975,
    "25k": 25.0,
    "30k": 30.0,
    "35k": 35.0,
    "40k": 40.0,
    "finish": 42.195,
}

# Broad quality-control bounds, not physiological impossibility claims.
# In particular, a slow segment can reflect a real stop or medical problem;
# this filter changes the target to records passing these bounds.
_MIN_SEGMENT_PACE: Final[float] = 2.0
_MAX_SEGMENT_PACE: Final[float] = 20.0

# Bib prefixes used by the BAA for non-open-division athletes.
_NON_OPEN_BIB_PREFIXES: Final[tuple[str, ...]] = ("W", "H", "P")


@dataclass(frozen=True)
class SplitSchema:
    """How one raw results file maps onto the canonical checkpoint grid."""

    name: str
    columns: Mapping[str, str]
    time_format: str  # "clock" (HH:MM:SS) or "minutes" (decimal)
    sex_column: str
    age_column: str
    bib_column: str
    missing_tokens: tuple[str, ...] = ("-", "", "–", "—", "NA", "nan")

    @property
    def checkpoints(self) -> tuple[str, ...]:
        """Canonical checkpoint names present in this schema, ordered by distance."""
        return tuple(sorted(self.columns, key=lambda c: CHECKPOINT_KM[c]))


SCHEMAS: Final[Mapping[str, SplitSchema]] = {
    "rojour": SplitSchema(
        name="rojour",
        columns={
            "5k": "5K",
            "10k": "10K",
            "15k": "15K",
            "20k": "20K",
            "half": "Half",
            "25k": "25K",
            "30k": "30K",
            "35k": "35K",
            "40k": "40K",
            "finish": "Official Time",
        },
        time_format="clock",
        sex_column="M/F",
        age_column="Age",
        bib_column="Bib",
    ),
    "llimllib": SplitSchema(
        name="llimllib",
        columns={
            "5k": "5k",
            "10k": "10k",
            "20k": "20k",
            "half": "half",
            "25k": "25k",
            "30k": "30k",
            "35k": "35k",
            "40k": "40k",
            "finish": "official",
        },
        time_format="minutes",
        sex_column="gender",
        age_column="age",
        bib_column="bib",
    ),
}


@dataclass
class HygieneReport:
    """Audit trail: how many athletes each hygiene step removed, and why."""

    race: str
    steps: list[tuple[str, int, int]] = field(default_factory=list)
    blanked_mats: int = 0
    repaired_mats: int = 0

    def record(self, label: str, before: int, after: int) -> None:
        self.steps.append((label, before, after))

    def to_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame(self.steps, columns=["step", "n_before", "n_after"])
        frame.insert(0, "race", self.race)
        frame["dropped"] = frame["n_before"] - frame["n_after"]
        frame["blanked_mats"] = self.blanked_mats
        frame["repaired_mats"] = self.repaired_mats
        return frame

    def __str__(self) -> str:
        lines = [f"[{self.race}] hygiene pipeline"]
        for label, before, after in self.steps:
            lines.append(f"  {label:<44s} {before:6d} -> {after:6d}  (-{before - after})")
        lines.append(f"  mats blanked for non-monotone cumulative time: {self.blanked_mats}")
        lines.append(f"  isolated interior mats repaired by interpolation: {self.repaired_mats}")
        return "\n".join(lines)


def parse_clock_series(values: pd.Series, time_format: str, missing_tokens: Sequence[str]) -> pd.Series:
    """Parse a raw split column into cumulative elapsed minutes.

    Handles ``HH:MM:SS``, ``MM:SS``, decimal minutes, the BAA's ``"-"`` sentinel
    for an unread mat, and the ``0`` that some exports use for the same thing.
    Anything unparseable becomes ``NaN`` rather than raising: a single malformed
    cell should cost one checkpoint, not one athlete.
    """
    text = values.astype("string").str.strip()
    text = text.mask(text.isin(list(missing_tokens)))

    if time_format == "minutes":
        parsed = pd.to_numeric(text, errors="coerce")
    elif time_format == "clock":
        # Parse by each row's format, so mixed MM:SS / HH:MM:SS / decimal
        # columns work and an all-missing column cannot index absent fields.
        parsed = pd.to_numeric(text, errors="coerce")
        for pattern, hours in ((r"^(\d+):(\d{1,2}(?:\.\d+)?)$", False),
                               (r"^(\d+):(\d{1,2}):(\d{1,2}(?:\.\d+)?)$", True)):
            parts = text.str.extract(pattern).apply(pd.to_numeric, errors="coerce")
            seconds = parts[2] if hours else parts[1]
            valid = seconds.ge(0) & seconds.lt(60)
            if hours:
                valid &= parts[1].ge(0) & parts[1].lt(60)
                value = 60 * parts[0] + parts[1] + seconds / 60
            else:
                value = parts[0] + seconds / 60
            parsed = parsed.where(~valid.fillna(False), value)
    else:  # pragma: no cover - guarded by SplitSchema construction
        raise ValueError(f"unknown time_format {time_format!r}")

    parsed = pd.to_numeric(parsed, errors="coerce").astype(float)
    return parsed.mask(parsed <= 0.0)


def _derive_wave(bib_numeric: pd.Series, n_waves: int = 4) -> pd.Series:
    """Assign a start wave from the bib number.

    The BAA seeds bibs by qualifying time and issues them in contiguous blocks of
    roughly 8,000, one per wave.  The wave is used only as a *stratification*
    factor: every split in these files is a net (chip) time, so wave membership
    never needs to be subtracted off a clock.
    """
    edges = [0.0] + [8000.0 * (i + 1) for i in range(n_waves - 1)] + [np.inf]
    labels = [f"wave{i + 1}" for i in range(n_waves)]
    wave = pd.cut(bib_numeric, bins=edges, labels=labels, right=False)
    return wave.astype("string").fillna("unknown")


def _enforce_monotone(cumulative: pd.DataFrame, report: HygieneReport) -> pd.DataFrame:
    """Blank out mats that violate strict monotonicity of cumulative time.

    A non-increasing cumulative split is internally inconsistent. Its cause
    and the location of the original error are unknown; this conservative
    left-to-right rule may blank later reads after an erroneously large one.
    """
    values = cumulative.to_numpy(dtype=float, copy=True)
    n_rows, n_cols = values.shape
    for j in range(1, n_cols):
        previous = np.fmax.accumulate(np.nan_to_num(values[:, :j], nan=-np.inf), axis=1)[:, -1]
        bad = np.isfinite(previous) & np.isfinite(values[:, j]) & (values[:, j] <= previous)
        values[bad, j] = np.nan
        report.blanked_mats += int(bad.sum())
    return pd.DataFrame(values, index=cumulative.index, columns=cumulative.columns)


def _interpolate_interior(cumulative: pd.DataFrame, distances: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Repair *isolated* interior mats by linear interpolation in distance.

    Only a single missing checkpoint flanked by two observed ones is repaired; a
    run of two or more consecutive misses is left missing, because interpolating
    across 10 km would manufacture exactly the smooth late-race trajectory the
    analysis is trying to detect.  The returned mask marks every repaired cell so
    the primary analysis can insist on *observed* mats.
    """
    values = cumulative.to_numpy(dtype=float, copy=True)
    observed = np.isfinite(values)
    repaired = np.zeros_like(observed)

    for j in range(1, values.shape[1] - 1):
        isolated = ~observed[:, j] & observed[:, j - 1] & observed[:, j + 1]
        if not isolated.any():
            continue
        span = distances[j + 1] - distances[j - 1]
        weight = (distances[j] - distances[j - 1]) / span
        values[isolated, j] = (
            values[isolated, j - 1] + weight * (values[isolated, j + 1] - values[isolated, j - 1])
        )
        repaired[isolated, j] = True

    frame = pd.DataFrame(values, index=cumulative.index, columns=cumulative.columns)
    mask = pd.DataFrame(repaired, index=cumulative.index, columns=cumulative.columns)
    return frame, mask, int(repaired.sum())


def load_race(
    path: str | Path,
    schema: SplitSchema,
    year: int,
    *,
    report: HygieneReport | None = None,
) -> tuple[pd.DataFrame, HygieneReport]:
    """Read one results file and run the full hygiene pipeline.

    Returns a tidy frame with one row per surviving athlete: cumulative
    checkpoint times in minutes (``t_5k`` ... ``t_finish``), an ``imp_*`` flag
    per checkpoint, and ``sex``/``age``/``bib``/``wave``/``year`` metadata.
    """
    path = Path(path)
    report = report or HygieneReport(race=f"{schema.name}-{year}")
    raw = pd.read_csv(path, dtype="string", low_memory=False)
    report.record("rows in raw file", len(raw), len(raw))

    checkpoints = schema.checkpoints
    distances = np.array([CHECKPOINT_KM[c] for c in checkpoints], dtype=float)

    # --- bib / division metadata -------------------------------------------------
    bib_text = raw[schema.bib_column].astype("string").str.strip().str.upper()
    prefix = bib_text.str.extract(r"^([A-Z]*)", expand=False).fillna("")
    is_open = ~prefix.isin(_NON_OPEN_BIB_PREFIXES)
    before = len(raw)
    raw = raw.loc[is_open]
    bib_text = bib_text.loc[is_open]
    report.record("drop wheelchair / handcycle / push-rim", before, len(raw))

    before = len(raw)
    duplicated = bib_text.duplicated(keep="first")
    raw = raw.loc[~duplicated]
    bib_text = bib_text.loc[~duplicated]
    report.record("drop duplicate bibs", before, len(raw))

    # --- splits ------------------------------------------------------------------
    cumulative = pd.DataFrame(
        {
            name: parse_clock_series(raw[schema.columns[name]], schema.time_format, schema.missing_tokens)
            for name in checkpoints
        },
        index=raw.index,
    )[list(checkpoints)]

    cumulative = _enforce_monotone(cumulative, report)
    cumulative, imputed, n_repaired = _interpolate_interior(cumulative, distances)
    report.repaired_mats = n_repaired

    # --- plausibility ------------------------------------------------------------
    # Include the start-to-first-checkpoint segment in the same check.
    with_start = np.column_stack([np.zeros(len(cumulative)), cumulative.to_numpy(dtype=float)])
    segment_times = np.diff(with_start, axis=1)
    segment_pace = segment_times / np.diff(np.r_[0.0, distances])
    with warnings.catch_warnings():
        # An athlete with no usable mat at all yields an all-NaN slice; they are
        # kept here and removed later by the metric's completeness requirement.
        warnings.simplefilter("ignore", RuntimeWarning)
        slowest = np.nanmax(segment_pace, axis=1)
        fastest = np.nanmin(segment_pace, axis=1)
    implausible = (fastest < _MIN_SEGMENT_PACE) | (slowest > _MAX_SEGMENT_PACE)

    before = len(raw)
    keep = ~implausible
    raw, cumulative, imputed, bib_text = raw.loc[keep], cumulative.loc[keep], imputed.loc[keep], bib_text.loc[keep]
    report.record(f"drop segment pace outside [{_MIN_SEGMENT_PACE}, {_MAX_SEGMENT_PACE}] min/km", before, len(raw))

    # --- assemble ----------------------------------------------------------------
    sex = raw[schema.sex_column].astype("string").str.strip().str.upper().str[0]
    before = len(raw)
    valid_sex = sex.isin(["M", "F"])
    raw, cumulative, imputed, bib_text, sex = (
        raw.loc[valid_sex], cumulative.loc[valid_sex], imputed.loc[valid_sex],
        bib_text.loc[valid_sex], sex.loc[valid_sex],
    )
    report.record("drop missing or unsupported recorded sex field", before, len(raw))

    bib_numeric = pd.to_numeric(bib_text.str.extract(r"(\d+)", expand=False), errors="coerce")
    tidy = pd.DataFrame(index=raw.index)
    tidy["t_start"] = 0.0
    tidy["imp_start"] = False
    for name in checkpoints:
        tidy[f"t_{name}"] = cumulative[name]
        tidy[f"imp_{name}"] = imputed[name].to_numpy(dtype=bool)
    tidy["sex"] = sex.astype("string")
    tidy["age"] = pd.to_numeric(raw[schema.age_column], errors="coerce")
    tidy["bib"] = bib_numeric
    tidy["wave"] = _derive_wave(bib_numeric)
    tidy["year"] = year
    return tidy.reset_index(drop=True), report


def load_races(
    sources: Mapping[int, tuple[str | Path, SplitSchema]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and concatenate several races; returns ``(tidy, hygiene_table)``."""
    frames: list[pd.DataFrame] = []
    reports: list[pd.DataFrame] = []
    for year, (path, schema) in sorted(sources.items()):
        tidy, report = load_race(path, schema, year)
        print(report)
        frames.append(tidy)
        reports.append(report.to_frame())
    combined = pd.concat(frames, ignore_index=True, sort=False)
    return combined, pd.concat(reports, ignore_index=True)
