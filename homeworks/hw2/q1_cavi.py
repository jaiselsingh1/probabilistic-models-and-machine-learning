"""Full-covariance mean-field CAVI for Homework 2, Question 1(d).

Run: python q1_cavi.py
Writes results/q1_results.json and figures/q1_numbers.tex beside this script.
Requires NumPy only. All generated data come from the stated model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def update_factor(ratings, other_means, other_covs, sigma2, k):
    """One exact coordinate maximizer; an empty ratings array gives the prior."""
    precision = np.eye(k) + (
        other_covs.sum(axis=0) + other_means.T @ other_means
    ) / sigma2
    linear = other_means.T @ ratings / sigma2
    covariance = np.linalg.solve(precision, np.eye(k))
    mean = np.linalg.solve(precision, linear)
    return mean, covariance


def elbo(ratings, observed, mu, su, mv, sv, sigma2):
    ii, jj = np.nonzero(observed)
    au = su + np.einsum("ik,il->ikl", mu, mu)
    av = sv + np.einsum("jk,jl->jkl", mv, mv)
    predicted = np.einsum("ik,ik->i", mu[ii], mv[jj])
    second = np.einsum("ikl,ilk->i", au[ii], av[jj])
    error2 = ratings[ii, jj] ** 2 - 2 * ratings[ii, jj] * predicted + second
    likelihood = -.5 * (len(ii) * np.log(2 * np.pi * sigma2) + error2.sum() / sigma2)
    divergence = 0.
    for means, covs in ((mu, su), (mv, sv)):
        signs, logdet = np.linalg.slogdet(covs)
        assert np.all(signs > 0)
        divergence += .5 * (np.trace(covs, axis1=1, axis2=2).sum()
                             + (means ** 2).sum() - means.size - logdet.sum())
    return float(likelihood - divergence)


def fit(ratings, observed, k=3, sigma2=.25, seed=0, max_iter=1000, tol=1e-9):
    rng = np.random.default_rng(seed)
    n_users, n_items = ratings.shape
    mu = rng.normal(0, .1, size=(n_users, k))
    mv = rng.normal(0, .1, size=(n_items, k))
    su = np.repeat(np.eye(k)[None], n_users, axis=0)
    sv = np.repeat(np.eye(k)[None], n_items, axis=0)
    history = [elbo(ratings, observed, mu, su, mv, sv, sigma2)]
    for iteration in range(max_iter):
        for i in range(n_users):
            selected = observed[i]
            mu[i], su[i] = update_factor(ratings[i, selected], mv[selected], sv[selected], sigma2, k)
        for j in range(n_items):
            selected = observed[:, j]
            mv[j], sv[j] = update_factor(ratings[selected, j], mu[selected], su[selected], sigma2, k)
        current = elbo(ratings, observed, mu, su, mv, sv, sigma2)
        assert current >= history[-1] - 1e-7, "CAVI lowered the ELBO"
        history.append(current)
        if current - history[-2] < tol * (1 + abs(current)):
            break
    else:
        raise RuntimeError("CAVI did not converge within max_iter")
    return mu, su, mv, sv, np.asarray(history)


def predictive_moments(mu, su, mv, sv, sigma2):
    mean = mu @ mv.T
    variance = (sigma2 + np.einsum("ikl,jlk->ij", su, sv)
                + np.einsum("ik,jkl,il->ij", mu, sv, mu)
                + np.einsum("jk,ikl,jl->ij", mv, su, mv))
    return mean, variance


def verify_moments(seed=20261112):
    rng = np.random.default_rng(seed)
    k, n, sigma2 = 3, 500_000, .25
    mu, mv = rng.normal(size=(2, k))
    a, b = rng.normal(size=(2, k, k))
    su, sv = a @ a.T / k + .1 * np.eye(k), b @ b.T / k + .1 * np.eye(k)
    mean, variance = predictive_moments(mu[None], su[None], mv[None], sv[None], sigma2)
    draws = (rng.multivariate_normal(mu, su, n) * rng.multivariate_normal(mv, sv, n)).sum(axis=1)
    draws += rng.normal(0, np.sqrt(sigma2), n)
    mean, variance = float(mean[0, 0]), float(variance[0, 0])
    mean_se = np.sqrt(variance / n)
    variance_se = np.std((draws - mean) ** 2, ddof=1) / np.sqrt(n)
    mean_z = abs(draws.mean() - mean) / mean_se
    variance_z = abs(draws.var(ddof=1) - variance) / variance_se
    assert mean_z < 5 and variance_z < 5
    cold_mean, cold_cov = update_factor(np.empty(0), np.empty((0, k)), np.empty((0, k, k)), sigma2, k)
    np.testing.assert_array_equal(cold_mean, np.zeros(k))
    np.testing.assert_array_equal(cold_cov, np.eye(k))
    return {"predictive_mean_mc_z": float(mean_z), "predictive_variance_mc_z": float(variance_z),
            "cold_start_prior_check": "passed", "moment_mc_draws": n}


def interval_coverage(ratings, heldout, mu, su, mv, sv, sigma2, seed, draws=2000):
    """Marginal predictive intervals from q, without a Gaussian approximation."""
    rng = np.random.default_rng(seed)
    ii, jj = np.nonzero(heldout)
    lu, lv = np.linalg.cholesky(su), np.linalg.cholesky(sv)
    covered, widths = 0, []
    for start in range(0, len(ii), 64):
        i, j = ii[start:start + 64], jj[start:start + 64]
        u = mu[i, None, :] + np.einsum("bkl,bdl->bdk", lu[i], rng.normal(size=(len(i), draws, mu.shape[1])))
        v = mv[j, None, :] + np.einsum("bkl,bdl->bdk", lv[j], rng.normal(size=(len(i), draws, mv.shape[1])))
        responses = (u * v).sum(axis=2) + rng.normal(0, np.sqrt(sigma2), size=(len(i), draws))
        lower, upper = np.quantile(responses, [.025, .975], axis=1)
        covered += np.count_nonzero((ratings[i, j] >= lower) & (ratings[i, j] <= upper))
        widths.extend(upper - lower)
    return {"nominal_coverage": .95, "empirical_test_coverage": covered / len(ii),
            "mean_interval_width": float(np.mean(widths)), "mc_draws_per_pair": draws}


def experiment():
    seed, n_users, n_items, k, sigma2 = 20261111, 80, 60, 3, .25
    rng = np.random.default_rng(seed)
    true_u = rng.normal(size=(n_users, k))
    true_v = rng.normal(size=(n_items, k))
    latent = true_u @ true_v.T
    ratings = latent + rng.normal(0, np.sqrt(sigma2), size=latent.shape)
    observed = rng.uniform(size=ratings.shape) < .4
    fits = [fit(ratings, observed, k, sigma2, seed + restart + 1) for restart in range(3)]
    best = max(fits, key=lambda fitted: fitted[-1][-1])
    mu, su, mv, sv, history = best
    predicted, variance = predictive_moments(mu, su, mv, sv, sigma2)
    heldout = ~observed
    assert np.all(variance >= sigma2)
    assert np.all(np.linalg.eigvalsh(su) > 0) and np.all(np.linalg.eigvalsh(sv) > 0)
    assert np.max(np.linalg.eigvalsh(su)) <= 1 + 1e-10
    assert np.max(np.linalg.eigvalsh(sv)) <= 1 + 1e-10
    # Rotation-invariant validation: assess predictions, not coordinatewise U/V recovery.
    result = {"seed": seed, "n_users": n_users, "n_items": n_items, "k": k, "noise_sd": np.sqrt(sigma2),
              "train_n": int(observed.sum()), "test_n": int(heldout.sum()), "restarts": len(fits),
              "best_elbo": float(history[-1]), "sweeps": len(history) - 1,
              "all_elbo_minimum_increases": [float(np.diff(f[-1]).min()) for f in fits],
              "test_rmse": float(np.sqrt(np.mean((ratings[heldout] - predicted[heldout]) ** 2))),
              "zero_predictor_test_rmse": float(np.sqrt(np.mean(ratings[heldout] ** 2))),
              "oracle_conditional_mean_test_rmse": float(np.sqrt(np.mean((ratings[heldout] - latent[heldout]) ** 2))),
              "latent_signal_test_rmse": float(np.sqrt(np.mean((latent[heldout] - predicted[heldout]) ** 2))),
              "test_mean_predictive_sd": float(np.sqrt(variance[heldout]).mean()),
              "predictive_interval": interval_coverage(ratings, heldout, mu, su, mv, sv, sigma2, seed + 100),
              "checks": verify_moments()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument("--output", type=Path, default=root / "results" / "q1_results.json")
    parser.add_argument("--fig-dir", type=Path, default=root / "figures")
    args = parser.parse_args()
    result = experiment()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    numbers = {
        "qoneUsers": str(result["n_users"]),
        "qoneItems": str(result["n_items"]),
        "qoneK": str(result["k"]),
        "qoneNoise": f'{result["noise_sd"]:.1f}',
        "qoneTrain": str(result["train_n"]),
        "qoneTest": str(result["test_n"]),
        "qoneRMSE": f'{result["test_rmse"]:.3f}',
        "qoneBaseline": f'{result["zero_predictor_test_rmse"]:.3f}',
        "qoneOracle": f'{result["oracle_conditional_mean_test_rmse"]:.3f}',
        "qoneCoverage": f'{100 * result["predictive_interval"]["empirical_test_coverage"]:.1f}',
        "qoneSeed": str(result["seed"]),
    }
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    macros = "% Generated by q1_cavi.py; do not edit numerical values manually.\n"
    macros += "\n".join(r"\newcommand{\%s}{%s}" % (name, value) for name, value in numbers.items()) + "\n"
    (args.fig_dir / "q1_numbers.tex").write_text(macros)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
