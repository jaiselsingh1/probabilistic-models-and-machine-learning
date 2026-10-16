#!/usr/bin/env python3
"""Independent numerical checks for Homework 1, Questions 1 and 2.

Run with ``python -m experiments.marathon_wall.check_derivations``.
Uses only the standard library. Integrates the *prior times likelihood*, then
compares its evidence and posterior moments with the derived expressions.
These finite numerical examples complement, rather than replace, the proofs.
"""

from __future__ import annotations

import json
import math
from typing import Callable


def _close(actual: float, expected: float, *, tolerance: float = 2e-8) -> None:
    if not math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise AssertionError(f"{actual:.16g} differs from {expected:.16g}")


def _simpson(f: Callable[[float], float], lo: float, hi: float, n: int) -> float:
    if n <= 0 or n % 2:
        raise ValueError("Simpson integration needs a positive, even interval count")
    step = (hi - lo) / n
    return step / 3 * math.fsum(
        (1 if k in (0, n) else 4 if k % 2 else 2) * f(lo + k * step)
        for k in range(n + 1)
    )


def _normal_pdf(x: float, mean: float, variance: float) -> float:
    return math.exp(-(x - mean) ** 2 / (2 * variance)) / math.sqrt(2 * math.pi * variance)


def _posterior(tau2: float, beta: float, x1: float) -> tuple[float, float]:
    denominator = beta + (1 - beta) * tau2
    return math.sqrt(1 - beta) * tau2 * x1 / denominator, beta * tau2 / denominator


def check_q1() -> dict[str, object]:
    examples = []
    for tau2, beta, x1 in ((2.5, 0.3, 1.2), (0.2, 0.9, -0.75), (1.0, 0.01, 2.0)):
        def joint(x0: float) -> float:
            return _normal_pdf(x0, 0, tau2) * _normal_pdf(x1, math.sqrt(1 - beta) * x0, beta)

        # Fixed prior-scale bounds, chosen independently of the posterior formula.
        extent = 12 * math.sqrt(tau2)
        evidence = _simpson(joint, -extent, extent, 20_000)
        mean = _simpson(lambda x: x * joint(x), -extent, extent, 20_000) / evidence
        variance = _simpson(lambda x: (x - mean) ** 2 * joint(x), -extent, extent, 20_000) / evidence
        expected_mean, expected_variance = _posterior(tau2, beta, x1)
        expected_evidence = _normal_pdf(x1, 0, (1 - beta) * tau2 + beta)
        _close(evidence, expected_evidence, tolerance=1e-10)
        _close(mean, expected_mean, tolerance=1e-10)
        _close(variance, expected_variance, tolerance=1e-10)
        examples.append({
            "tau_squared": tau2, "beta": beta, "x1": x1,
            "posterior_mean": mean, "posterior_variance": variance,
            "maximum_absolute_error": max(abs(evidence - expected_evidence),
                                          abs(mean - expected_mean), abs(variance - expected_variance)),
        })

    for tau2 in (0.2, 1.0, 2.5):
        x1 = 0.75
        _close(_posterior(tau2, 0, x1)[0], x1)
        _close(_posterior(tau2, 0, x1)[1], 0)
        _close(_posterior(tau2, 1, x1)[0], 0)
        _close(_posterior(tau2, 1, x1)[1], tau2)
        near_zero = _posterior(tau2, 1e-10, x1)
        near_one = _posterior(tau2, 1 - 1e-12, x1)
        _close(near_zero[0], x1)
        _close(near_zero[1], 0)
        _close(near_one[0], 0, tolerance=2e-6)
        _close(near_one[1], tau2)
        variances = [_posterior(tau2, k / 100, x1)[1] for k in range(101)]
        if any(v1 >= v2 for v1, v2 in zip(variances, variances[1:])):
            raise AssertionError("Posterior variance must increase strictly with beta")
    return {"quadrature_examples": examples, "endpoint_and_monotonicity_checks": "passed"}


def _beta(parameters: tuple[float, ...]) -> float:
    return math.exp(math.fsum(math.lgamma(a) for a in parameters) - math.lgamma(sum(parameters)))


def _simplex_integrals(alpha: tuple[float, ...], counts: tuple[int, ...], n: int = 400) -> list[float]:
    """Integrate joint density, first moments, and second moments on a triangle.

    pi=(s,(1-s)t,(1-s)(1-t)) maps [0,1]^2 to the simplex and has
    Jacobian 1-s. No posterior normalizer or posterior moments are used here.
    The chosen test priors have nonsingular, polynomial integrands.
    """
    row_totals: list[list[float]] = [[] for _ in range(7)]
    prior_normalizer = 1 / _beta(alpha)
    for i in range(n + 1):
        s = i / n
        wi = 1 if i in (0, n) else 4 if i % 2 else 2
        rows: list[list[float]] = [[] for _ in range(7)]
        for j in range(n + 1):
            t = j / n
            wj = 1 if j in (0, n) else 4 if j % 2 else 2
            pi = (s, (1 - s) * t, (1 - s) * (1 - t))
            prior = prior_normalizer * math.prod(p ** (a - 1) for p, a in zip(pi, alpha))
            likelihood = math.prod(p ** count for p, count in zip(pi, counts))
            weighted_joint = wi * wj * (1 - s) * prior * likelihood
            moments = (1.0, *pi, *(p * p for p in pi))
            for row, moment in zip(rows, moments):
                row.append(weighted_joint * moment)
        for total, row in zip(row_totals, rows):
            total.append(math.fsum(row))
    return [math.fsum(rows) / (9 * n * n) for rows in row_totals]


def check_q2() -> dict[str, object]:
    examples = []
    for alpha, counts in (((1.0, 2.0, 3.0), (3, 0, 2)),
                          ((2.0, 3.0, 2.0), (0, 0, 0)),
                          ((1.0, 1.0, 1.0), (0, 2, 1))):
        posterior = tuple(a + count for a, count in zip(alpha, counts))
        total = sum(posterior)
        integrals = _simplex_integrals(alpha, counts)
        evidence = integrals[0]
        expected_evidence = _beta(posterior) / _beta(alpha)
        # Equivalently, integrating the proposed normalized posterior must give 1.
        normalized_mass = evidence / expected_evidence
        _close(normalized_mass, 1)
        predictive = [m / evidence for m in integrals[1:4]]
        expected_predictive = [a / total for a in posterior]
        variances = [moment / evidence - mean ** 2 for moment, mean in zip(integrals[4:], predictive)]
        expected_variances = [a * (total - a) / (total ** 2 * (total + 1)) for a in posterior]
        for actual, expected in zip(predictive + variances, expected_predictive + expected_variances):
            _close(actual, expected)
        _close(sum(predictive), 1)
        if not all(p > 0 for p in predictive):
            raise AssertionError("Every predictive category must retain positive probability")
        examples.append({
            "alpha": alpha, "counts": counts, "posterior_parameters": posterior,
            "integrated_posterior_mass": normalized_mass, "predictive": predictive,
            "maximum_moment_error": max(abs(a - b) for a, b in
                                        zip(predictive + variances, expected_predictive + expected_variances)),
        })
    return {"simplex_quadrature_examples": examples, "zero_count_and_no_data_cases": "passed"}


def main() -> None:
    print(json.dumps({"status": "passed", "q1": check_q1(), "q2": check_q2()}, indent=2))


if __name__ == "__main__":
    main()
