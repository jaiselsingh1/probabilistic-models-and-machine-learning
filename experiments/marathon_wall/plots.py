"""Figures for the Box's-loop write-up.

Colours come from the three validated categorical slots (blue / orange / aqua);
every series is direct-labelled or legended, never identified by colour alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import arviz as az
import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from .diagnostics import _flat, loo_pit

__all__ = [
    "SERIES",
    "use_report_style",
    "pooled_predictive_density",
    "figure_data",
    "figure_model",
    "figure_invariance",
]

SERIES: Mapping[str, str] = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a"}
_INK = "#0b0b0b"
_MUTED = "#52514e"
_GRID = "#d9d8d4"
_HILL = "#f0ece2"


def use_report_style() -> None:
    """Recessive grid, thin marks, serif text to match the LaTeX body."""
    mpl.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 160,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 7.5,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.5,
            "axes.edgecolor": _MUTED,
            "axes.linewidth": 0.6,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": _GRID,
            "grid.linewidth": 0.4,
            "legend.frameon": False,
            "legend.fontsize": 6.8,
            "xtick.color": _MUTED,
            "ytick.color": _MUTED,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "text.color": _INK,
            "axes.labelcolor": _INK,
            "lines.linewidth": 1.3,
        }
    )


def _normal_pdf(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    z = (x - mu) / sigma
    return np.exp(-0.5 * z * z) / (sigma * np.sqrt(2.0 * np.pi))


def pooled_predictive_density(
    idata: az.InferenceData,
    model_name: str,
    grid: np.ndarray,
    group_weights: np.ndarray,
    *,
    max_draws: int = 400,
) -> np.ndarray:
    """Posterior-mean predictive density, pooled over groups by group size.

    Evaluated analytically rather than by sampling replicates: with N ~ 3e4 the
    Monte-Carlo noise of a sampled PPC would dominate the very feature -- a
    shoulder at the right of the mode -- the figure exists to show.
    """
    from scipy import stats

    weights = np.asarray(group_weights, dtype=float)
    weights = weights / weights.sum()
    x = grid[None, :, None]  # (1, grid, group)

    if model_name == "gaussian":
        mu = _flat(idata, "mu")[:max_draws, None, :]
        sigma = _flat(idata, "sigma")[:max_draws, None, :]
        density = _normal_pdf(x, mu, sigma)
    elif model_name == "mixture":
        pi = _flat(idata, "pi")[:max_draws, None, :]
        mu0 = _flat(idata, "mu0_")[:max_draws, None, :]
        mu1 = _flat(idata, "mu1")[:max_draws, None, :]
        s0 = _flat(idata, "sigma0")[:max_draws, None, :]
        s1 = _flat(idata, "sigma1")[:max_draws, None, :]
        density = (1.0 - pi) * _normal_pdf(x, mu0, s0) + pi * _normal_pdf(x, mu1, s1)
    elif model_name == "skewnormal":
        xi = _flat(idata, "xi")[:max_draws, None, :]
        omega = _flat(idata, "omega")[:max_draws, None, :]
        alpha = _flat(idata, "alpha")[:max_draws, None, :]
        density = stats.skewnorm.pdf(x, a=alpha, loc=xi, scale=omega)
    else:  # pragma: no cover
        raise ValueError(f"unknown model {model_name!r}")

    return np.asarray((density * weights[None, None, :]).sum(axis=2).mean(axis=0), dtype=float)


def figure_data(
    segment_profile: pd.Series,
    reference_frame: pd.DataFrame,
    ability_quantiles: pd.DataFrame,
    metric: pd.DataFrame,
    path: Path,
) -> Figure:
    """Task 2: the confound, the adjustment, and the adjusted metric."""
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 1.48), constrained_layout=True)

    ax = axes[0]
    labels = list(segment_profile.index)
    centres = np.arange(len(labels))
    ax.axvspan(4.5, 6.5, color=_HILL, zorder=0)
    ax.plot(centres, segment_profile.to_numpy(), color=SERIES["blue"], marker="o", markersize=2.6)
    ax.text(5.5 / (len(labels) - 1), 0.95, "Newton\nHills", transform=ax.transAxes,
            color=_MUTED, fontsize=6.2, ha="center", va="top")
    ax.text(0.03, 0.80, "Hopkinton\ndescent", transform=ax.transAxes,
            color=_MUTED, fontsize=6.2, ha="left", va="top")
    ax.set_xticks(centres)
    ax.set_xticklabels([l.replace("->", "–").replace("start", "0") for l in labels],
                       rotation=55, ha="right", fontsize=5.6)
    ax.set_ylabel("field median pace (min/km)")
    ax.set_title("(a) the course is the confound", loc="left")

    ax = axes[1]
    for q, colour in zip(("q10", "q50", "q90"), ("aqua", "blue", "orange")):
        ax.plot(ability_quantiles["ability"], ability_quantiles[q], color=SERIES[colour], linewidth=1.1)
        ax.annotate(q, xy=(ability_quantiles["ability"].iloc[-1], ability_quantiles[q].iloc[-1]),
                    xytext=(2.5, 0), textcoords="offset points", color=SERIES[colour], fontsize=6.2,
                    va="center")
    ax.plot(reference_frame["ability_min_per_km"], reference_frame["delta"],
            color=_INK, linewidth=1.4, linestyle=(0, (3, 1.4)))
    mid = len(reference_frame) // 4
    ax.annotate(r"$\hat{\delta}_q$", xy=(reference_frame["ability_min_per_km"].iloc[mid],
                reference_frame["delta"].iloc[mid]),
                xytext=(-13, 4), textcoords="offset points", color=_INK, fontsize=7)
    ax.set_xlabel("baseline 10K–25K pace (min/km)")
    ax.set_ylabel(r"raw $\log(p^{late}/p^{base})$")
    ax.set_title("(b) fade depends on ability", loc="left")

    ax = axes[2]
    bins = np.linspace(-0.3, 0.6, 130)
    for sex, colour in (("M", "blue"), ("F", "orange")):
        values = metric.loc[metric["sex"] == sex, "y"].to_numpy()
        ax.hist(values, bins=bins, density=True, histtype="step", linewidth=1.1, color=SERIES[colour])
        peak = np.histogram(values, bins=bins, density=True)[0]
        centre = 0.5 * (bins[int(np.argmax(peak))] + bins[int(np.argmax(peak)) + 1])
        ax.annotate(sex, xy=(centre, peak.max()), xytext=(9 if sex == "M" else -11, -4),
                    textcoords="offset points", color=SERIES[colour], fontsize=7)
    ax.margins(y=0.12)
    ax.set_xlabel(r"de-confounded deceleration $y$")
    ax.set_ylabel("density")
    ax.set_title("(c) a mode and a shoulder", loc="left")

    fig.savefig(path, bbox_inches="tight")
    return fig


def figure_model(
    metric: pd.DataFrame,
    group_codes: np.ndarray,
    grid: np.ndarray,
    densities: Mapping[str, np.ndarray],
    idatas: Mapping[str, az.InferenceData],
    pi_frame: pd.DataFrame,
    smyth: Mapping[str, float],
    path: Path,
) -> Figure:
    """Tasks 4-5: which model, is it calibrated, and does the component behave."""
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 1.48), constrained_layout=True)

    ax = axes[0]
    bins = np.linspace(-0.25, 0.55, 150)
    ax.hist(metric["y"].to_numpy(), bins=bins, density=True, color="#e6e5e0", edgecolor="none",
            label="observed")
    styles = {"mixture": ("blue", "-"), "skewnormal": ("orange", (0, (3, 1.3))), "gaussian": ("aqua", (0, (1, 1.4)))}
    for name, (colour, dash) in styles.items():
        if name in densities:
            ax.plot(grid, densities[name], color=SERIES[colour], linestyle=dash, linewidth=1.2,
                    label={"mixture": "2-component", "skewnormal": "skew-normal", "gaussian": "Gaussian"}[name])
    ax.set_xlim(-0.2, 0.52)
    ax.set_xlabel(r"$y$")
    ax.set_ylabel("density")
    ax.legend(loc="upper right", handlelength=1.6)
    ax.set_title("(d) posterior predictive fit", loc="left")

    ax = axes[1]
    for name, colour in (("mixture", "blue"), ("skewnormal", "orange")):
        if name not in idatas:
            continue
        pit = loo_pit(idatas[name], name, metric["y"].to_numpy(dtype=float), group_codes)
        pit = np.sort(pit[np.isfinite(pit)])
        ecdf = np.arange(1, pit.size + 1) / pit.size
        ax.plot(pit, ecdf - pit, color=SERIES[colour], linewidth=1.2,
                label={"mixture": "2-component", "skewnormal": "skew-normal"}[name])
    ax.axhline(0.0, color=_MUTED, linewidth=0.6)
    ax.set_xlabel("LOO-PIT")
    ax.set_ylabel("ECDF − uniform")
    ax.legend(loc="upper right", handlelength=1.6)
    ax.set_title("(e) calibration of the two fits", loc="left")

    ax = axes[2]
    order = np.argsort(pi_frame["group"].to_numpy())
    positions = np.arange(len(pi_frame))
    colours = [SERIES["blue"] if str(g).startswith("M") else SERIES["orange"] for g in pi_frame["group"].iloc[order]]
    ax.errorbar(
        pi_frame["pi_mean"].to_numpy()[order], positions,
        xerr=np.vstack([
            (pi_frame["pi_mean"] - pi_frame["pi_lo"]).to_numpy()[order],
            (pi_frame["pi_hi"] - pi_frame["pi_mean"]).to_numpy()[order],
        ]),
        fmt="o", markersize=3.0, linewidth=1.0, ecolor=_MUTED, mfc="none", mew=1.0,
    )
    for pos, colour, value in zip(positions, colours, pi_frame["pi_mean"].to_numpy()[order]):
        ax.plot([value], [pos], marker="o", markersize=3.0, color=colour)
    for sex, colour in (("M", "blue"), ("F", "orange")):
        ax.axvline(smyth[sex], color=SERIES[colour], linestyle=(0, (2, 1.6)), linewidth=1.0)
        ax.text(smyth[sex], 0.985, f"Smyth {sex} {smyth[sex]:.0%}",
                transform=ax.get_xaxis_transform(), color=SERIES[colour], fontsize=5.8,
                rotation=90, ha="right", va="top")
    ax.set_yticks(positions)
    ax.set_yticklabels(pi_frame["group"].to_numpy()[order], fontsize=5.8)
    ax.margins(x=0.10)
    ax.set_xlabel(r"wall fraction $\pi_g$")
    ax.set_title("(f) $\\pi_g$ vs the rule-based rate", loc="left")

    fig.savefig(path, bbox_inches="tight")
    return fig


def figure_invariance(invariance: pd.DataFrame, path: Path) -> Figure:
    """Check (c): ability-matched pace difference between the latent classes."""
    fig, ax = plt.subplots(figsize=(3.4, 1.85), constrained_layout=True)
    labels = [str(s).replace("->", "\u2013").replace("start", "0") for s in invariance["segment"]]
    positions = np.arange(len(labels))
    disjoint = invariance["disjoint"].to_numpy(dtype=bool)
    late = np.array([_segment_starts_after(str(s), 25.0) for s in invariance["segment"]])
    colours = [SERIES["orange"] if flag else SERIES["blue"] for flag in late]
    ax.axhline(0.0, color=_MUTED, linewidth=0.6)
    # Hollow bars are mechanically tied to the metric and carry no evidence.
    ax.bar(positions[disjoint], invariance["smd"].to_numpy()[disjoint],
           color=[c for c, d in zip(colours, disjoint) if d], width=0.68)
    ax.bar(positions[~disjoint], invariance["smd"].to_numpy()[~disjoint], width=0.68,
           facecolor="none", edgecolor=_MUTED, linewidth=0.7, hatch="////")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=5.6)
    ax.set_ylabel("ability-matched SMD")
    ax.text(0.03, 0.96, "before 25K", transform=ax.transAxes, color=SERIES["blue"],
            fontsize=6.2, ha="left", va="top")
    ax.text(0.97, 0.96, "after 25K", transform=ax.transAxes, color=SERIES["orange"],
            fontsize=6.2, ha="right", va="top")
    ax.set_xlabel("hatched = inside a window of $y$, positive control only", fontsize=5.8,
                  color=_MUTED)
    fig.savefig(path, bbox_inches="tight")
    return fig


def _segment_starts_after(segment: str, km: float) -> bool:
    from .data import CHECKPOINT_KM

    return CHECKPOINT_KM[segment.split("->")[0]] >= km
