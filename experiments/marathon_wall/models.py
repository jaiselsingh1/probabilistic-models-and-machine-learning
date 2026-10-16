"""Archived Bayesian density models for the deceleration metric.

The current submission uses :mod:`homeworks.hw1.audited_analysis`.  This module preserves
the exploratory fits; their component labels are statistical, not diagnosed
physiological states.

All three are hierarchical over a grouping ``g`` (here sex x start wave, or sex
x year), and all three are fitted to the same vector ``y``:

``gaussian``
    A single Gaussian reference density,
    :math:`y_i \\sim \\mathcal{N}(\\mu_g, \\sigma_g^2)`.

``mixture``
    A two-component density, marginalised over the latent indicator so that
    NUTS sees a smooth posterior:

    .. math::
        p(y_i \\mid \\theta) = (1 - \\pi_g)\\,\\mathcal{N}(y_i \\mid \\mu_{0,g}, \\sigma_{0,g}^2)
        + \\pi_g\\,\\mathcal{N}(y_i \\mid \\mu_{1,g}, \\sigma_{1,g}^2).

    Label switching is removed structurally rather than by post-hoc relabelling:
    the higher-mean component's mean is built as
    :math:`\\mu_{1,g} = \\mu_{0,g} + \\exp(\\lambda_g)`, so
    :math:`\\mu_{0,g} < \\mu_{1,g}` holds on every draw and the hierarchical
    prior on :math:`\\lambda_g` is still unconstrained.

``skewnormal``
    A unimodal asymmetric reference density.  A mixture can improve prediction
    by fitting tail weight or other shape features this family cannot express.
    Beating this reference does not establish bimodality or distinct states.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, Sequence

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

__all__ = ["GroupIndex", "ModelName", "build_model", "sample_model", "SAMPLER_DEFAULTS"]

ModelName = Literal["gaussian", "mixture", "skewnormal"]

SAMPLER_DEFAULTS: Final[dict[str, object]] = {
    "draws": 1000,
    "tune": 1500,
    "chains": 4,
    "target_accept": 0.9,
    "random_seed": 20260415,
}


@dataclass(frozen=True)
class GroupIndex:
    """Integer codes for a categorical grouping, plus their labels."""

    codes: np.ndarray
    labels: tuple[str, ...]

    @classmethod
    def from_columns(cls, frame: pd.DataFrame, columns: Sequence[str]) -> "GroupIndex":
        joined = frame[list(columns)].astype("string").agg(" x ".join, axis=1)
        categorical = pd.Categorical(joined)
        return cls(codes=np.asarray(categorical.codes, dtype="int32"), labels=tuple(categorical.categories))

    @property
    def n_groups(self) -> int:
        return len(self.labels)

    def counts(self) -> pd.Series:
        return pd.Series(np.bincount(self.codes, minlength=self.n_groups), index=list(self.labels), name="n")


def _hierarchical_scalar(
    name: str,
    n_groups: int,
    *,
    mu_prior: tuple[float, float],
    tau_prior: float,
    centered: bool = True,
    dims: str = "group",
) -> pt.TensorVariable:
    """Hierarchical normal for one group-level scalar.

    The archived implementation defaults to a centred parameterisation.
    Its suitability depends on the likelihood and group information; sample
    size alone does not establish good geometry, especially for mixtures.
    Convergence and sampler diagnostics must be checked for each fit.
    """
    mu = pm.Normal(f"{name}_mu", mu=mu_prior[0], sigma=mu_prior[1])
    tau = pm.HalfNormal(f"{name}_tau", sigma=tau_prior)
    if centered:
        return pm.Normal(name, mu=mu, sigma=tau, dims=dims)
    offset = pm.Normal(f"{name}_offset", mu=0.0, sigma=1.0, dims=dims)
    return pm.Deterministic(name, mu + tau * offset, dims=dims)


def build_model(
    name: ModelName,
    y: np.ndarray,
    groups: GroupIndex,
    *,
    scale_hint: float | None = None,
) -> pm.Model:
    """Construct one of the three models.

    ``scale_hint`` sets the prior scale; it defaults to the empirical standard
    deviation of ``y``, which keeps the priors weakly informative on whatever
    units the metric happens to carry rather than hard-coding Boston's.
    This default uses the supplied observations to set prior scales, so it is
    an empirical prior choice; predictive evaluation must account for that.
    """
    y = np.asarray(y, dtype=float)
    if y.shape != groups.codes.shape:
        raise ValueError(f"y has shape {y.shape} but the group index has shape {groups.codes.shape}")
    s = float(scale_hint if scale_hint is not None else np.std(y))
    g = groups.n_groups

    coords = {"group": list(groups.labels), "obs": np.arange(y.size)}
    with pm.Model(coords=coords) as model:
        index = pm.Data("group_code", groups.codes, dims="obs")
        observed = pm.Data("y", y, dims="obs")

        if name == "gaussian":
            mu = _hierarchical_scalar("mu", g, mu_prior=(0.0, 2.0 * s), tau_prior=s)
            log_sigma = _hierarchical_scalar("log_sigma_", g, mu_prior=(np.log(s), 0.5), tau_prior=0.5)
            sigma = pm.Deterministic("sigma", pt.exp(log_sigma), dims="group")
            pm.Normal("y_obs", mu=mu[index], sigma=sigma[index], observed=observed, dims="obs")

        elif name == "mixture":
            mu0 = _hierarchical_scalar("mu0_", g, mu_prior=(0.0, 2.0 * s), tau_prior=s)
            # Positive increment => mu0 < mu1 on every draw, no label switching.
            log_gap = _hierarchical_scalar("log_gap", g, mu_prior=(np.log(1.5 * s), 0.7), tau_prior=0.5)
            mu1 = pm.Deterministic("mu1", mu0 + pt.exp(log_gap), dims="group")
            pm.Deterministic("gap", pt.exp(log_gap), dims="group")

            log_sigma0 = _hierarchical_scalar("log_sigma0_", g, mu_prior=(np.log(0.6 * s), 0.5), tau_prior=0.5)
            log_sigma1 = _hierarchical_scalar("log_sigma1_", g, mu_prior=(np.log(1.0 * s), 0.5), tau_prior=0.5)
            sigma0 = pm.Deterministic("sigma0", pt.exp(log_sigma0), dims="group")
            sigma1 = pm.Deterministic("sigma1", pt.exp(log_sigma1), dims="group")

            logit_pi = _hierarchical_scalar("logit_pi", g, mu_prior=(-1.4, 1.0), tau_prior=0.7)
            pi = pm.Deterministic("pi", pm.math.sigmoid(logit_pi), dims="group")

            weights = pt.stack([1.0 - pi[index], pi[index]], axis=1)
            components = [
                pm.Normal.dist(mu=mu0[index], sigma=sigma0[index]),
                pm.Normal.dist(mu=mu1[index], sigma=sigma1[index]),
            ]
            pm.Mixture("y_obs", w=weights, comp_dists=components, observed=observed, dims="obs")

        elif name == "skewnormal":
            xi = _hierarchical_scalar("xi", g, mu_prior=(0.0, 2.0 * s), tau_prior=s)
            log_omega = _hierarchical_scalar("log_omega_", g, mu_prior=(np.log(s), 0.5), tau_prior=0.5)
            omega = pm.Deterministic("omega", pt.exp(log_omega), dims="group")
            alpha = _hierarchical_scalar("alpha", g, mu_prior=(2.0, 3.0), tau_prior=1.5)
            pm.SkewNormal("y_obs", mu=xi[index], sigma=omega[index], alpha=alpha[index],
                          observed=observed, dims="obs")

        else:  # pragma: no cover - exhaustive over ModelName
            raise ValueError(f"unknown model {name!r}")

    return model


def sample_model(
    model: pm.Model,
    *,
    log_likelihood_thin: int = 4,
    compute_log_likelihood: bool = True,
    **sampler_kwargs: object,
) -> az.InferenceData:
    """Sample, then attach a *thinned* pointwise log-likelihood.

    With N ~ 3e4 athletes a full 4000-draw log-likelihood array is ~1 GB per
    model; thinning to every fourth draw reduces memory use, at the cost of
    fewer draws for PSIS-LOO.  Its Monte Carlo accuracy still needs checking.
    Pass ``compute_log_likelihood=False`` for a model whose likelihood
    is expressed as a :class:`pm.Potential` (the change-point model), where PyMC
    has no observed variable to evaluate pointwise.
    """
    options = {**SAMPLER_DEFAULTS, **sampler_kwargs}
    with model:
        idata = pm.sample(progressbar=False, idata_kwargs={"log_likelihood": False}, **options)
        if compute_log_likelihood:
            thinned = idata.sel(draw=slice(None, None, log_likelihood_thin))
            pm.compute_log_likelihood(thinned, model=model, progressbar=False, extend_inferencedata=True)
            idata.add_groups({"log_likelihood": thinned.log_likelihood})
    return idata
