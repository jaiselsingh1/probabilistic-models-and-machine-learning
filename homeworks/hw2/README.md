# Homework 2

This write-up covers Question 1(a) to (d), including its optional synthetic
CAVI experiment, Question 2(a) to (f), and Question 3, the final-project proposal.

The document uses the same 12-point article, Times-style fonts, one-inch margins,
and heading style as Homework 1. Question 1 occupies one page. Question 2 has
no stated page limit and occupies three pages. Question 3 occupies two pages,
including references.

- `hw2.pdf`: compiled six-page solution.
- `hw2.tex` and `preamble.tex`: editable LaTeX source.
- `q1_cavi.py`: full-covariance mean-field CAVI and synthetic validation.
- `q2_kl.py`: the required Monte Carlo grid search, figures, and numerical macros.
- `check_kl.py`: independent quadrature and checks of the population/empirical KL identities.
- `check_preferences.py`: numerical check for an earlier Question 3 draft on preference learning (commit `49143bd`). The current proposal does not use it.
- `figures/`: PDF figures and generated LaTeX macros. The heatmap mesh is rasterized to prevent seams.
- `results/`: complete grid estimates and numerical results.

## Reproduce

Run from this directory with Python 3.11:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python q1_cavi.py
python q2_kl.py
python check_kl.py
python check_preferences.py
latexmk -pdf -interaction=nonstopmode -halt-on-error hw2.tex
```

On Windows, replace the activation command with
`.venv\Scripts\Activate.ps1`. The PDF rebuild needs a TeX distribution with the
packages in `preamble.tex`, plus TikZ and `fix-cm`. Reading the supplied PDF does
not require Python or TeX. No external dataset is required.

## What the numerical checks establish

### Question 1

The synthetic matrix is generated from the assignment's model with seed
20261111, 80 users, 60 items, rank 3, and noise standard deviation 0.5. A random
40% observation mask yields 1,874 training and 2,926 held-out ratings. Three
nonzero random initializations are fit using only observed ratings; the highest
training ELBO selects the fit. Test ratings never select the initialization.
All-zero factor means are avoided because their CAVI updates remain zero.

The implementation checks that each sweep increases the ELBO, covariance
matrices remain positive definite, and an entity with no observations retains
the prior. It compares the predictive mean and variance formula with 500,000
independent draws. Its predictive intervals use 2,000 simulation draws per
held-out pair, not a Gaussian approximation to the bilinear predictive density.

The reported test RMSE is 0.558890, versus 1.751241 for zero prediction and
0.494990 for the known synthetic conditional mean. Nominal 95% predictive
intervals cover 94.3951% of test ratings in this one simulation. These are
descriptive checks, not a guarantee that mean-field intervals are calibrated
on every dataset. Factor coordinates are not compared directly with the true
coordinates because simultaneous orthogonal rotations leave predictions unchanged.

### Question 2

The required grid has 81 mean values and 36 standard deviations, with spacing
0.1 and inclusive endpoints. Each of its 2,916 points receives 10,000 newly
generated standard-normal draws transformed into samples from that point's
Gaussian. The loop visits increasing mean first, then increasing standard
deviation, using NumPy `default_rng` (PCG64) with seed 20261111.

The sampled minimum is `(m, sigma) = (3.0, 1.0)` with estimated reverse KL
0.6892963181 nats. The independent checker integrates the entire grid and
finds the symmetric minima `(±3, 1)`, with KL 0.6892979330. Refining those two
points gives `(±2.984306, 1.023369)`, with KL 0.688768996. It also compares
the log-cosh and direct-density integrals and tests the objective, gradient,
and ordering identities on finite population and empirical distributions,
including a zero-mass category. The Monte Carlo standard error reported in
the write-up is a pointwise diagnostic; it does not account for selecting the
smallest estimate across the grid.

The theoretical solutions do not depend on the random seed. The particular
Monte Carlo grid winner can depend on it. Minor floating-point differences
between platforms are possible. The independent quadrature calculation is a
verification step, not a substitute for the assignment's sampling method.

### Question 3

Question 3 proposes a probabilistic account of distilling a sampling-based
planner (MPPI) into a policy with guided policy search. It contains no new
computation in this folder: the pilot figures it quotes come from the separate
research project and are marked as unaudited in the text. The earlier draft of
Question 3, on preference learning, is in commit `49143bd`; `check_preferences.py`,
`figures/q3_numbers.tex` and `results/q3_checks.json` belong to that draft and are
no longer used by `hw2.tex`.
