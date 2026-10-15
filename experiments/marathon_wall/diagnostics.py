"""Model comparison, calibration, and the latent-class decision rule.

Nothing here is specific to marathons: given fitted ``InferenceData`` for the
three models of :mod:`.models` it produces the PSIS-LOO table, the LOO-PIT
calibration curve, the per-athlete posterior responsibility
:math:`P(z_i = 1 \\mid y_i)`, and the Bayes-risk decision rule that turns that
responsibility into a binary "hit the wall" call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr

__all__ = [
    "compare_models",
    "predictive_cdf",
    "loo_table",
    "posterior_responsibility",
    "responsibility_curve",
    "bayes_decision",
    "calibrate_loss_ratio",
    "decision_boundary",
    "loo_pit",
]

_LOG_SQRT_2PI: float = 0.5 * np.log(2.0 * np.pi)


def _flat(idata: az.InferenceData, name: str) -> np.ndarray:
    """Posterior draws for ``name`` flattened to ``(draw, ...)``."""
    array = idata.posterior[name]
    return array.stack(sample=("chain", "draw")).transpose("sample", ...).to_numpy()


def compare_models(idatas: Mapping[str, az.InferenceData]) -> pd.DataFrame:
    """PSIS-LOO comparison table, ranked, with the Delta-ELPD against the winner."""
    table = az.compare(dict(idatas), ic="loo", method="stacking")
    return table


def loo_table(idatas: Mapping[str, az.InferenceData]) -> pd.DataFrame:
    """Per-model ELPD, effective parameters, and the worst Pareto-k."""
    rows: list[dict[str, object]] = []
    for name, idata in idatas.items():
        result = az.loo(idata, pointwise=True)
        khat = np.asarray(result.pareto_k.to_numpy(), dtype=float)
        rows.append(
            {
                "model": name,
                "elpd_loo": float(result.elpd_loo),
                "se": float(result.se),
                "p_loo": float(result.p_loo),
                "max_khat": float(np.max(khat)),
                "frac_khat_gt_0.7": float(np.mean(khat > 0.7)),
            }
        )
    return pd.DataFrame(rows).sort_values("elpd_loo", ascending=False).reset_index(drop=True)


def _normal_logpdf(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    z = (y - mu) / sigma
    return -0.5 * z * z - np.log(sigma) - _LOG_SQRT_2PI


def posterior_responsibility(
    idata: az.InferenceData,
    y: np.ndarray,
    group_codes: np.ndarray,
    *,
    chunk: int = 4000,
) -> np.ndarray:
    """Posterior mean responsibility :math:`E[P(z_i = 1 \\mid y_i, \\theta)]`.

    The latent indicator was marginalised out for sampling; it is recovered here
    by evaluating the component posterior at each draw and averaging, which is
    the Rao-Blackwellised estimator of :math:`P(z_i = 1 \\mid y)`.
    """
    pi = _flat(idata, "pi")
    mu0 = _flat(idata, "mu0_")
    mu1 = _flat(idata, "mu1")
    sigma0 = _flat(idata, "sigma0")
    sigma1 = _flat(idata, "sigma1")

    y = np.asarray(y, dtype=float)
    codes = np.asarray(group_codes, dtype=int)
    out = np.empty(y.size, dtype=float)

    for start in range(0, y.size, chunk):
        stop = min(start + chunk, y.size)
        block = y[start:stop][None, :]
        c = codes[start:stop]
        log0 = np.log1p(-pi[:, c]) + _normal_logpdf(block, mu0[:, c], sigma0[:, c])
        log1 = np.log(pi[:, c]) + _normal_logpdf(block, mu1[:, c], sigma1[:, c])
        peak = np.maximum(log0, log1)
        r = np.exp(log1 - peak) / (np.exp(log0 - peak) + np.exp(log1 - peak))
        out[start:stop] = r.mean(axis=0)
    return out


def responsibility_curve(
    idata: az.InferenceData,
    group_code: int,
    y_grid: np.ndarray,
) -> np.ndarray:
    """Posterior mean responsibility as a function of ``y`` for one group."""
    codes = np.full(y_grid.size, group_code, dtype=int)
    return posterior_responsibility(idata, y_grid, codes)


def bayes_decision(responsibility: np.ndarray, loss_ratio: float = 1.0) -> np.ndarray:
    """Minimum-Bayes-risk classification of each athlete.

    Under 0-1-style losses with ``loss_ratio`` = C(false negative) / C(false
    positive), the risk-minimising rule calls a wall whenever
    :math:`P(z_i = 1 \\mid y_i) > 1 / (1 + \\text{loss\\_ratio})`.
    """
    if loss_ratio <= 0.0:
        raise ValueError("loss_ratio must be positive")
    return responsibility > 1.0 / (1.0 + loss_ratio)


def calibrate_loss_ratio(responsibility: np.ndarray, target_rate: float) -> tuple[float, float]:
    """Find the loss ratio whose Bayes rule reproduces ``target_rate`` prevalence.

    Returns ``(loss_ratio, threshold)``.  A value near 1 means the model's
    equal-cost decision boundary already agrees with the external rule being
    compared against; a value far from 1 quantifies the implicit asymmetry in
    that external rule.
    """
    if not 0.0 < target_rate < 1.0:
        raise ValueError("target_rate must lie strictly inside (0, 1)")
    threshold = float(np.quantile(responsibility, 1.0 - target_rate))
    threshold = float(np.clip(threshold, 1e-9, 1.0 - 1e-9))
    return (1.0 - threshold) / threshold, threshold


def decision_boundary(
    idata: az.InferenceData,
    group_code: int,
    *,
    threshold: float = 0.5,
    side: str = "upper",
    lo: float = -0.6,
    hi: float = 1.2,
    n_grid: int = 2401,
) -> float:
    """Value of ``y`` at which a group's responsibility crosses ``threshold``.

    The wall component is the *wider* of the two, so its responsibility exceeds
    the threshold in **both** tails: a runner with a huge negative split is
    flagged just as readily as one who collapsed.  That is a defect of the
    two-Gaussian parameterisation rather than a physiological claim, so the
    decision rule must say which crossing it means.  ``side="upper"`` gives the
    deceleration boundary; ``side="lower"`` exposes the spurious one, which is
    worth reporting precisely because it is spurious.
    """
    grid = np.linspace(lo, hi, n_grid)
    curve = responsibility_curve(idata, group_code, grid)
    crossings = np.flatnonzero(np.diff(np.sign(curve - threshold)) != 0)
    if crossings.size == 0:
        return float("nan")
    if side == "upper":
        i = int(crossings[-1])
    elif side == "lower":
        i = int(crossings[0])
    else:  # pragma: no cover
        raise ValueError("side must be 'upper' or 'lower'")
    x0, x1 = grid[i], grid[i + 1]
    y0, y1 = curve[i] - threshold, curve[i + 1] - threshold
    return float(x0 - y0 * (x1 - x0) / (y1 - y0))


def predictive_cdf(
    idata: az.InferenceData,
    model_name: str,
    y: np.ndarray,
    group_codes: np.ndarray,
    *,
    draw_slice: slice = slice(None),
) -> np.ndarray:
    """Posterior predictive CDF ``F(y_i | theta^s)`` with shape ``(draw, obs)``."""
    from scipy import stats

    y = np.asarray(y, dtype=float)[None, :]
    c = np.asarray(group_codes, dtype=int)

    if model_name == "gaussian":
        mu, sigma = _flat(idata, "mu")[draw_slice], _flat(idata, "sigma")[draw_slice]
        return stats.norm.cdf(y, loc=mu[:, c], scale=sigma[:, c])
    if model_name == "mixture":
        pi = _flat(idata, "pi")[draw_slice][:, c]
        lower = stats.norm.cdf(y, loc=_flat(idata, "mu0_")[draw_slice][:, c],
                               scale=_flat(idata, "sigma0")[draw_slice][:, c])
        upper = stats.norm.cdf(y, loc=_flat(idata, "mu1")[draw_slice][:, c],
                               scale=_flat(idata, "sigma1")[draw_slice][:, c])
        return (1.0 - pi) * lower + pi * upper
    if model_name == "skewnormal":
        return stats.skewnorm.cdf(
            y,
            a=_flat(idata, "alpha")[draw_slice][:, c],
            loc=_flat(idata, "xi")[draw_slice][:, c],
            scale=_flat(idata, "omega")[draw_slice][:, c],
        )
    raise ValueError(f"unknown model {model_name!r}")


def loo_pit(
    idata: az.InferenceData,
    model_name: str,
    y: np.ndarray,
    group_codes: np.ndarray,
    *,
    chunk: int = 4000,
) -> np.ndarray:
    """Leave-one-out probability-integral transform, computed from the CDF.

    Rather than drawing and ranking posterior-predictive replicates -- which at
    N ~ 5e4 would cost hundreds of megabytes and add Monte-Carlo noise to the
    very tail being judged -- this evaluates the exact predictive CDF at each
    observation and reweights the draws by their Pareto-smoothed importance
    weights, which is the definition the sampled version approximates:

    .. math::  \mathrm{PIT}_i = \sum_s w_i^{(s)} F(y_i \mid \theta^{(s)}),
       \qquad w_i^{(s)} \propto 1 / p(y_i \mid \theta^{(s)}).

    Under a calibrated predictive these are uniform on [0, 1]; a U shape means
    the predictive is too narrow, an inverted U too wide, and a tilt means it is
    systematically mislocated.
    """
    # The log-likelihood was stored on a thinned slice of the posterior; the CDF
    # must be evaluated on exactly those draws or the weights will not align.
    # Select by the stored draw coordinates rather than re-deriving a thinning
    # factor, which integer division gets wrong whenever the draw count is not
    # an exact multiple of the stride.
    posterior_slice = idata.sel(draw=idata.log_likelihood["draw"].to_numpy())

    log_lik = idata.log_likelihood["y_obs"].stack(sample=("chain", "draw"))
    log_lik = log_lik.transpose("sample", ...).to_numpy()
    n_draws, n_obs = log_lik.shape
    if n_obs != y.size:  # pragma: no cover - guards a mis-aligned idata
        raise ValueError(f"log-likelihood has {n_obs} observations but y has {y.size}")

    out = np.empty(n_obs, dtype=float)
    for start in range(0, n_obs, chunk):
        stop = min(start + chunk, n_obs)
        block_loglik = log_lik[:, start:stop]
        log_weights, _ = az.psislw(-block_loglik.T, reff=1.0)  # (obs, draw)
        weights = np.exp(log_weights)
        weights /= weights.sum(axis=1, keepdims=True)
        cdf = predictive_cdf(posterior_slice, model_name, y[start:stop], group_codes[start:stop])
        out[start:stop] = np.einsum("od,do->o", weights, cdf)
    return out
