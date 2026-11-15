"""Independent numerical checks for HW2 Q2; run python check_kl.py.

This checks the written identities and the deterministic grid objective. It
does not replace the assignment's Monte Carlo grid search in q2_kl.py, and
does not import that implementation. Assertions complement the written proofs.
"""

import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parent
A, S = 3.0, 1.0
LOG2 = np.log(2.0)
LOG2PI = np.log(2.0 * np.pi)


def reverse_kl_logcosh(m, sigma):
    """Use the exact Gaussian entropy and quadrature for E_q log cosh."""
    def integrand(eps):
        t = A * (m + sigma * eps) / S**2
        log_cosh = np.logaddexp(t, -t) - LOG2
        return log_cosh * np.exp(-eps**2 / 2.0) / np.sqrt(2.0 * np.pi)

    # At this finite grid's scales the omitted standard-normal tails are
    # negligible relative to the quadrature tolerance. Splitting at z=0
    # helps the adaptive integrator resolve the mixture's central region.
    value, _ = quad(integrand, -12.0, 12.0, points=[-m / sigma],
                    epsabs=2e-12, epsrel=2e-12)
    return (np.log(S / sigma) - 0.5
            + (m**2 + sigma**2 + A**2) / (2.0 * S**2) - value)


def direct_reverse_kl(m, sigma):
    """Independently integrate q(z) [log q(z)-log p(z)] on the real line."""
    def integrand(z):
        log_q = -np.log(sigma) - LOG2PI / 2.0 - ((z - m) / sigma)**2 / 2.0
        log_p = (logsumexp([-((z + A) / S)**2 / 2.0,
                            -((z - A) / S)**2 / 2.0])
                 - LOG2 - np.log(S) - LOG2PI / 2.0)
        return np.exp(log_q) * (log_q - log_p)

    return quad(integrand, -np.inf, np.inf, epsabs=1e-12, epsrel=1e-12)


def centered_difference(function, theta, step=1e-5):
    gradient = np.empty_like(theta)
    for j in range(theta.size):
        displacement = np.zeros_like(theta)
        displacement[j] = step
        gradient[j] = (function(theta + displacement)
                       - function(theta - displacement)) / (2.0 * step)
    return gradient


def continuous_fits(grid_minimizers, grid_minimum):
    """Refine each grid minimum in (mean, log standard deviation)."""
    def objective(parameters):
        return reverse_kl_logcosh(parameters[0], np.exp(parameters[1]))

    fits = []
    for initial_m, initial_sigma in grid_minimizers:
        fit = minimize(objective, [initial_m, np.log(initial_sigma)],
                       method="Nelder-Mead", options={"xatol": 1e-9, "fatol": 1e-13})
        if not fit.success:
            raise RuntimeError(fit.message)
        m, sigma = float(fit.x[0]), float(np.exp(fit.x[1]))
        kl, error = direct_reverse_kl(m, sigma)
        np.testing.assert_allclose(kl, fit.fun, atol=2e-11, rtol=0)
        np.testing.assert_allclose(centered_difference(objective, fit.x), 0, atol=1e-6)
        assert kl < grid_minimum and m * initial_m > 0
        fits.append({"m": m, "sigma": sigma, "kl": kl, "direct_quadrature_error": error})
    minus, plus = fits
    np.testing.assert_allclose([-minus["m"], minus["sigma"], minus["kl"]],
                               [plus["m"], plus["sigma"], plus["kl"]], rtol=0, atol=2e-7)
    return fits


def check_discrete_identities():
    """Check (d)-(f) with a strictly positive softmax model and zero masses."""
    theta = np.array([-1.1, 0.2, 0.8, -0.3])
    comparisons = [theta, -theta, np.zeros(4), np.array([0.4, -0.9, 0.1, 1.2])]
    observations = np.array([0, 2, 2, 3, 2, 3, 0, 2, 3, 2, 3])
    population = np.array([0.2, 0.0, 0.5, 0.3])
    empirical = np.bincount(observations, minlength=4) / observations.size
    checks = []

    for label, weights in [("population", population), ("empirical", empirical)]:
        positive = weights > 0.0  # Implements 0 log(0/q)=0 without log(0).
        entropy = -float(np.dot(weights[positive], np.log(weights[positive])))

        def log_likelihood(parameters):
            return float(np.dot(weights, parameters - logsumexp(parameters)))

        def divergence(parameters):
            log_q = parameters - logsumexp(parameters)
            return float(np.dot(weights[positive],
                                np.log(weights[positive]) - log_q[positive]))

        q = np.exp(theta - logsumexp(theta))
        exact_gradient = weights - q
        ell_gradient = centered_difference(log_likelihood, theta)
        kl_gradient = centered_difference(divergence, theta)
        np.testing.assert_allclose(ell_gradient, exact_gradient, atol=2e-10, rtol=0)
        np.testing.assert_allclose(kl_gradient, -exact_gradient, atol=2e-10, rtol=0)
        objective_errors = [abs(divergence(t) + entropy + log_likelihood(t))
                            for t in comparisons]
        assert max(objective_errors) < 2e-14
        for first in comparisons:
            for second in comparisons:
                assert ((log_likelihood(first) <= log_likelihood(second))
                        == (divergence(first) >= divergence(second)))

        if label == "empirical":
            sample_log_likelihood = float((theta - logsumexp(theta))[observations].mean())
            sample_gradient = (np.eye(4)[observations] - q).mean(axis=0)
            np.testing.assert_allclose(sample_log_likelihood, log_likelihood(theta), atol=1e-14)
            np.testing.assert_allclose(sample_gradient, exact_gradient, atol=1e-14)

        checks.append({
            "distribution": label,
            "probabilities": weights.tolist(),
            "zero_mass_categories": np.flatnonzero(~positive).tolist(),
            "maximum_objective_identity_error": max(objective_errors),
            "maximum_log_likelihood_gradient_error": float(np.max(abs(ell_gradient - exact_gradient))),
            "maximum_kl_gradient_error": float(np.max(abs(kl_gradient + exact_gradient))),
            "ordering_comparisons_passed": len(comparisons)**2,
        })
    return checks


def main():
    means = np.arange(-40, 41) / 10.0
    sigmas = np.arange(5, 41) / 10.0
    values = np.array([[reverse_kl_logcosh(m, s) for m in means] for s in sigmas])
    assert values.shape == (36, 81) and np.isfinite(values).all()
    symmetry_error = float(np.max(abs(values - values[:, ::-1])))
    assert symmetry_error < 2e-12
    minimum = float(values.min())
    indices = np.argwhere(np.isclose(values, minimum, rtol=0.0, atol=1e-11))
    minimizers = sorted((float(means[i]), float(sigmas[j])) for j, i in indices)
    assert minimizers == [(-3.0, 1.0), (3.0, 1.0)]
    continuous = continuous_fits(minimizers, minimum)

    direct_checks = []
    for m, sigma in [(-3.0, 1.0), (3.0, 1.0), (2.9, 1.0),
                     (3.0, 1.1), (0.0, np.sqrt(A**2 + S**2)), (1.5, 2.0)]:
        direct_value, error_estimate = direct_reverse_kl(m, sigma)
        formula_value = reverse_kl_logcosh(m, sigma)
        difference = abs(direct_value - formula_value)
        assert difference < 2e-11
        direct_checks.append({
            "m": m, "sigma": float(sigma),
            "logcosh_quadrature": formula_value,
            "direct_density_quadrature": direct_value,
            "absolute_difference": difference,
            "direct_quadrature_error_estimate": error_estimate,
        })

    result = {
        "status": "passed",
        "target": {"a": A, "s": S},
        "quadrature_grid_points": int(values.size),
        "grid_minimizers": [{"m": m, "sigma": s} for m, s in minimizers],
        "grid_minimum_reverse_kl": minimum,
        "continuous_numerical_minimizers": continuous,
        "maximum_grid_symmetry_error": symmetry_error,
        "independent_direct_density_checks": direct_checks,
        "discrete_objective_and_gradient_checks": check_discrete_identities(),
        "note": "Deterministic checks complement, rather than replace, the required Monte Carlo experiment.",
    }
    destination = ROOT / "results" / "q2_independent_checks.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    positive = continuous[1]
    macros = {
        "klContinuousAbsM": f'{positive["m"]:.3f}',
        "klContinuousSigma": f'{positive["sigma"]:.3f}',
        "klContinuousD": f'{positive["kl"]:.5f}',
    }
    macro_path = ROOT / "figures" / "q2_continuous.tex"
    macro_path.parent.mkdir(exist_ok=True)
    macro_path.write_text("% Generated by check_kl.py.\n" + "".join(
        f"\\newcommand{{\\{key}}}{{{value}}}\n" for key, value in macros.items()))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
