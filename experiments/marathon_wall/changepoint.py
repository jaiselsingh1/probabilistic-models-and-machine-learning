"""Archived exploratory hierarchical change-point model over segment paces.

The current submission uses :mod:`homeworks.hw1.audited_analysis`.  This extension asks when
pacing changes appear, which a scalar late/baseline ratio cannot identify.
Neither a fitted change-point nor a mixture component identifies glycogen
depletion: its timing depends on intensity, initial stores, fueling, and other
unobserved factors.  There is no universal elapsed-time or distance threshold.

So promote the latent label to a latent *location*.  Write the athlete's log
pace on segment :math:`s` with midpoint distance :math:`d_s` as

.. math::

    \\ell_{is} = \\alpha_i + \\delta_s + \\beta_i \\, (d_s - \\kappa_i)_+ + \\varepsilon_{is},
    \\qquad \\varepsilon_{is} \\sim \\mathcal{N}(0, \\sigma_\\varepsilon^2),

with :math:`\\alpha_i` the athlete's level, :math:`\\delta_s` a sum-to-zero
segment profile shared by the whole field, :math:`\\kappa_i` the change-point and
:math:`\\beta_i > 0` the post-break fade rate.
The shared profile can absorb both course effects and common fatigue; it does
not identify a purely topographic correction.

The regime indicator survives as a mixture over whole trajectories:

.. math::

    p(\\ell_{i\\cdot}) = (1 - \\pi_g) \\prod_s \\mathcal{N}(\\ell_{is} \\mid \\alpha_i + \\delta_s, \\sigma_\\varepsilon^2)
    + \\pi_g \\prod_s \\mathcal{N}(\\ell_{is} \\mid \\alpha_i + \\delta_s + \\beta_i (d_s - \\kappa_i)_+, \\sigma_\\varepsilon^2),

The prior already favours change-points near 30 km, so concentration there is
not independent validation.  As :math:`\\beta_i` approaches zero, the increasing
trajectory approaches the flat branch, weakening identification of component
membership and the change-point.  Weather is not included in this model, and
year differences alone would not identify its causal effect.

The hinge is replaced by a softplus of width ``hinge_width`` so that NUTS sees a
differentiable likelihood; as the width goes to zero it recovers the exact
piecewise-linear form.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from homeworks.hw1.data import CHECKPOINT_KM
from .models import GroupIndex

__all__ = ["SegmentPanel", "build_changepoint_model", "summarise_changepoints"]


@dataclass(frozen=True)
class SegmentPanel:
    """Athlete-by-segment log-pace panel plus the segment midpoint distances."""

    log_pace: np.ndarray  # (n_athletes, n_segments)
    midpoints_km: np.ndarray  # (n_segments,)
    segment_names: tuple[str, ...]
    group_codes: np.ndarray  # (n_athletes,)
    group_labels: tuple[str, ...]

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        segment_paces: pd.DataFrame,
        groups: GroupIndex,
        *,
        max_athletes: int | None = None,
        seed: int = 11,
    ) -> "SegmentPanel":
        complete = segment_paces.notna().all(axis=1).to_numpy()
        index = np.flatnonzero(complete)
        if max_athletes is not None and index.size > max_athletes:
            index = np.random.default_rng(seed).choice(index, size=max_athletes, replace=False)
            index.sort()

        names = tuple(str(c) for c in segment_paces.columns)
        midpoints = np.array(
            [0.5 * (CHECKPOINT_KM[name.split("->")[0]] + CHECKPOINT_KM[name.split("->")[1]]) for name in names],
            dtype=float,
        )
        return cls(
            log_pace=np.log(segment_paces.to_numpy(dtype=float)[index]),
            midpoints_km=midpoints,
            segment_names=names,
            group_codes=np.asarray(groups.codes, dtype=int)[index],
            group_labels=groups.labels,
        )


def build_changepoint_model(
    panel: SegmentPanel,
    *,
    hinge_width: float = 1.5,
    kappa_bounds: tuple[float, float] = (8.0, 40.0),
) -> pm.Model:
    """Hierarchical piecewise-linear change-point model with a regime mixture."""
    n_athletes, n_segments = panel.log_pace.shape
    n_groups = len(panel.group_labels)
    distances = panel.midpoints_km

    coords = {
        "athlete": np.arange(n_athletes),
        "segment": list(panel.segment_names),
        "group": list(panel.group_labels),
    }
    with pm.Model(coords=coords) as model:
        observed = pm.Data("log_pace", panel.log_pace, dims=("athlete", "segment"))
        codes = pm.Data("group_code", panel.group_codes, dims="athlete")

        # Shared segment profile, sum-to-zero so that alpha carries the level.
        delta_raw = pm.Normal("delta_raw", mu=0.0, sigma=0.3, shape=n_segments - 1)
        delta = pm.Deterministic(
            "delta", pt.concatenate([delta_raw, -pt.sum(delta_raw, keepdims=True)]), dims="segment"
        )

        mu_alpha = pm.Normal("mu_alpha", mu=float(np.mean(panel.log_pace)), sigma=0.5)
        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=0.5)
        alpha = pm.Deterministic(
            "alpha", mu_alpha + sigma_alpha * pm.Normal("alpha_offset", 0.0, 1.0, dims="athlete"), dims="athlete"
        )

        # Change-point and post-break fade rate, partially pooled across athletes.
        mu_kappa = pm.TruncatedNormal("mu_kappa", mu=30.0, sigma=6.0, lower=kappa_bounds[0], upper=kappa_bounds[1])
        sigma_kappa = pm.HalfNormal("sigma_kappa", sigma=6.0)
        kappa = pm.TruncatedNormal(
            "kappa", mu=mu_kappa, sigma=sigma_kappa,
            lower=kappa_bounds[0], upper=kappa_bounds[1], dims="athlete",
        )
        mu_log_beta = pm.Normal("mu_log_beta", mu=np.log(0.01), sigma=1.0)
        sigma_log_beta = pm.HalfNormal("sigma_log_beta", sigma=1.0)
        beta = pm.Deterministic(
            "beta",
            pt.exp(mu_log_beta + sigma_log_beta * pm.Normal("log_beta_offset", 0.0, 1.0, dims="athlete")),
            dims="athlete",
        )

        sigma_eps = pm.HalfNormal("sigma_eps", sigma=0.1)
        logit_pi = pm.Normal("logit_pi", mu=-1.0, sigma=1.0, shape=n_groups)
        pi = pm.Deterministic("pi", pm.math.sigmoid(logit_pi), dims="group")

        # Softplus hinge: (d - kappa)_+ smoothed to width `hinge_width`.
        scaled = (distances[None, :] - kappa[:, None]) / hinge_width
        hinge = hinge_width * pt.softplus(scaled)

        flat = alpha[:, None] + delta[None, :]
        resid_clean = observed - flat
        resid_wall = observed - (flat + beta[:, None] * hinge)

        def _traj_logp(resid: pt.TensorVariable) -> pt.TensorVariable:
            per_cell = -0.5 * (resid / sigma_eps) ** 2 - pt.log(sigma_eps) - 0.5 * np.log(2.0 * np.pi)
            return pt.sum(per_cell, axis=1)

        log_clean = pt.log1p(-pi[codes]) + _traj_logp(resid_clean)
        log_wall = pt.log(pi[codes]) + _traj_logp(resid_wall)
        pm.Deterministic("log_r_wall", log_wall - pt.logaddexp(log_clean, log_wall), dims="athlete")
        pm.Potential("trajectory_likelihood", pt.sum(pt.logaddexp(log_clean, log_wall)))

    return model


def summarise_changepoints(idata: az.InferenceData, responsibility_floor: float = 0.8) -> pd.DataFrame:
    """Summarise athlete posterior means above a component-probability threshold.

    Selection uses the posterior arithmetic mean responsibility.  The reported
    ``kappa_q10/q50/q90`` are quantiles across selected athletes' posterior mean
    locations, not posterior credible intervals.  Legacy ``wall`` labels refer
    only to the increasing-trajectory component.  Archived summaries generated
    before the responsibility correction are not recomputed by this function.
    """
    posterior = idata.posterior
    kappa = posterior["kappa"].mean(dim=("chain", "draw")).to_numpy()
    r = np.exp(posterior["log_r_wall"]).mean(dim=("chain", "draw")).to_numpy()
    wall = r >= responsibility_floor
    return pd.DataFrame(
        {
            "statistic": ["n_wall", "kappa_mean", "kappa_q10", "kappa_q50", "kappa_q90"],
            "value": [
                float(wall.sum()),
                float(np.mean(kappa[wall])) if wall.any() else np.nan,
                float(np.quantile(kappa[wall], 0.10)) if wall.any() else np.nan,
                float(np.quantile(kappa[wall], 0.50)) if wall.any() else np.nan,
                float(np.quantile(kappa[wall], 0.90)) if wall.any() else np.nan,
            ],
        }
    )
