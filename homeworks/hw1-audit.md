# Homework 1 audit — October 16, 2026

## Outcome

Q1 and Q2 are mathematically correct. Q3 now makes a narrower, defensible claim: a two-Gaussian mixture improves predictive density relative to the fitted Gaussian and skew-normal, but its fitted densities are unimodal and its components do not identify physiological wall states. The final homework is four pages, respecting the assignment's 1/1/2-page limits.

The starting branch was `hw1-q3-marathon-wall`, commit `2149a2094257a3b117bb42273ae444b90ded2eb3`. Revised work was reviewed on `codex/hw1-verified-analysis` and is now merged into `main`. The submitted code and reproduction instructions are in [hw1/](hw1/README.md). Historical experiments remain available and are explicitly distinguished from the submitted analysis.

## The deeper task requested

A professional statistical review would independently derive the conjugate results, reconstruct the analysis from original observations, define the population and estimand, and challenge whether predictive fit establishes the proposed scientific interpretation. It would also verify selection, preprocessing leakage, uncertainty, numerical fitting, and the submitted document itself.

The least-digested inputs are the assignment equations and requirements, repository source and branch history, original BAA-derived checkpoint CSVs, the scraper/conversion code, and primary scientific sources. Existing prose, summary tables, and plots are claims to check against those inputs.

Methods/tools: independent algebra and numerical integration; pinned GitHub sources and SHA-256 verification; Python/NumPy/pandas/SciPy/Matplotlib; stratified held-out likelihood; training-only preprocessing; multistart EM and numerical optimization; sensitivity checks; density-mode and predictive-CDF diagnostics; regression tests; strict LaTeX compilation and visual inspection of every page.

External evidence of correctness: normalization and posterior-moment identities; original-source definitions and provenance; reproducible numerical outputs from verified CSV bytes; withheld-record scores; explicit failure-oriented tests; agreement between generated numerical macros, code, figures, and PDF. Predictive performance alone does not establish physiological truth.

Rewritten prompt executed: “Audit Homework 1 from first principles. Verify every part of Questions 1 and 2, correcting only what is needed. On the existing Q3 branch, trace the data to its source, sharpen the question to what the data can support, reproduce and check the probabilistic analysis, and complete a clear two-page account of Box's loop. Preserve useful work, document consequential choices and limitations, and deliver the revised source, code, figures, and checked PDF with concrete evidence for the conclusions.”

## Q1 and Q2

Q1's posterior mean is `sqrt(1-beta)*tau²*x1 / [beta+(1-beta)*tau²]`; its variance is `beta*tau² / [beta+(1-beta)*tau²]`. Completing the square and conditioning a joint Gaussian independently give the same expressions. The beta→0 point mass and beta→1 prior limits are correct, as is the stated variance derivative. The shrinkage interpretation correctly refers to the inverted observation.

The plate has correct dependencies and shading. The revision explicitly says N=1 for the stated one-pair problem. “Likelihood independent of x0” clarifies the beta→1 limit without changing the calculation.

Q2 correctly uses the ordered categorical-data likelihood without an unnecessary multinomial factor. The posterior is Dirichlet(alpha+n), and the predictive probability is `(alpha_v+n_v)/(sum(alpha)+n)`. The beta-function derivation, sufficient-statistic statement, normalization, and smoothing interpretation are correct. The optional empirical-frequency decomposition now states n>0. A redundant intermediate integral was removed to fix an overfull display.

Independent quadrature checks normalize the original prior×likelihood and compare posterior moments. Q1 errors are at most 5.6e-17 in the checked examples. Q2 posterior mass differs from one by at most 1.33e-9, and checked predictive/moment errors are at most 4.3e-10, including zero-count and no-observation cases. These supplement, rather than replace, the algebraic proof.

## Consequential Q3 findings and corrections

1. **Better mixture fit is not proof of two states.** A two-component mixture can itself have one peak. All four fitted primary group densities, and all 16 group/specification fits, are numerically unimodal. The broader higher-mean component dominates both extreme tails. Neither mixture weight nor its classification threshold estimates physiological wall prevalence.
2. **The empirical reference does not identify terrain effects.** Baseline pace and slowing also reflect strategy, fatigue, and selection. The 30–40 km window includes Heartbreak Hill. A fixed conditional location shift preserves shape at exactly fixed baseline pace, but shifting different observations can change the pooled shape; the old “conservative/cannot create bimodality” guarantee was false.
3. **Preprocessing was outside the old validation.** The old reference used all observations before PSIS-LOO. The new 80/20 stratified split precedes reference estimation, and held-out observations cannot affect the training reference or fitted parameters. The original exploratory analysis had already seen these data, so the revised holdout is an internal diagnostic, not pristine confirmation.
4. **The literature benchmark was misimplemented.** Smyth's published definition uses at least 25% slowing relative to 5–20 km over at least 5 km after 20 km. The old implementation used a 20% average slowdown from 25 km to the finish. Its “verbatim” comparison, agreement statistics, and calibrated loss interpretation were removed from the submission. Archived code warns about the historical misattribution. The published 28%/17% rates came from a selected repeat-runner analysis, not a target truth for Boston.
5. **The dataset is an incomplete scrape.** It excludes nonfinishers and some finishers. The interrupted 2013 race is especially selected. Informative selection alone does not imply a lower bound on wall prevalence. Rows are race performances, potentially repeated athletes across years.
6. **Exclusion accounting was incomplete.** 48,148 raw rows minus 105 wheelchair records, seven duplicate bibs, 41 pace-screen exclusions, and 304 records without observed metric endpoints leaves 47,691. The old “153 dropped” omitted the last 304. The final text reports all 457 exclusions.
7. **The old diagnostics did not establish everything claimed.** Skew-normal PSIS-LOO had max Pareto-k=0.7249 and a warning, while the prose emphasized the mixture's 0.407 value. One planted dataset with three covering intervals is not a coverage study. The old sampler draws/convergence summaries were not committed, so the audit does not claim to have independently reproduced those historical Bayesian estimates.
8. **The change-point extension was not an answer.** The old fit was unidentified. Also, its responsibility summary used exp(mean(log r)), a geometric mean, instead of mean(exp(log r)); the code now uses the arithmetic posterior mean. Across-athlete quantiles of estimated change points are not posterior credible intervals. This extension remains exploratory, outside the submission.
9. **Document generation was broken.** Macros such as `wallPiFx2013` contain digits that TeX does not treat as part of an ordinary command name, producing numeric garbage before the title. The old generator now uses letter-only names and validates them; the submitted PDF uses separately generated valid audit macros.

## Reproduced analysis and evidence

The primary model is deliberately manageable within course material: a Gaussian likelihood and two-component latent mixture, fitted by likelihood optimization/EM, plus a skew-normal shape comparison. The week 3–5 reading reports cover optimization, MCMC, and latent/mixed-membership models. The earlier hierarchical model and simulations were retained rather than required for the two-page answer.

There are 38,152 training records and 9,539 test records, split within recorded-sex/year groups using seed 20261016. The reference uses 40 bins and only training observations. Models are separately fitted within each group; there is no hierarchical prior in the revised analysis. Mixture likelihoods require a scale constraint because an unconstrained component can collapse on one observation. The selected fits are away from the floor, all six EM starts converge, and their mean objective values agree to less than 7.3e-10 for the primary fits. This supports numerical stability but is not a mathematical global-optimum proof.

| Comparison | Total held-out log-score gain | Conditional standard error |
|---|---:|---:|
| Mixture minus skew-normal, 25th-percentile reference | 244.9 | 48.1 |
| Mixture minus Gaussian, same reference | 1151.5 | 72.7 |
| Mixture minus skew-normal, raw ratio | 108.6 | 42.7 |
| Mixture minus skew-normal, 10th-percentile reference | 194.6 | 46.4 |
| Mixture minus skew-normal, median reference | 280.8 | 47.8 |
| Mixture minus skew-normal, 2014 held-out subset | 179.9 | 42.1 |

The primary per-record gain is 0.0257 nats, with conditional SE 0.0050. The 2014 subset retains the shared reference estimated using training records from both years; it is not a full 2014-only refit. Sensitivity rows share the same test set and do not provide independent replications. Not every individual sex/year comparison is decisive.

The paired-score SE is `sqrt(sum_g n_g * sample_variance(d_g))`, respecting the fixed group counts. It treats records as independent, conditions on the fitted training data, and excludes training-set uncertainty, possible repeat-runner dependence, and variation across races. No Bayesian posterior interval or population prevalence is reported.

Independent recomputation from saved data, without importing the fit functions, checks all 38,156 specification×heldout rows and all 48 comparison rows. Maximum log-density discrepancy is 3.6e-14. A separate normal-mixture derivative calculation finds one stationary maximum in all 16 fits, agreeing with the plotted/grid-based assessment. See [the verification results](hw1/results/verification.json) and [the independent verifier](hw1/verify_results.py).

Eleven regression tests cover mixed-format and missing clocks, observed-endpoint selection, isolated versus consecutive imputation, nonmonotone timing, the first segment's quality filter, density normalization/modes, recovery on separated synthetic components, invariance of training fits to altered held-out values, fixed-stratum SE, valid LaTeX macro names, and full exclusion counts. Strict LaTeX compilation completes with no overfull boxes, unresolved references, or warnings. All four rendered pages were visually inspected.

## What to be able to explain

- Why two mixture components can give a one-peaked density, and why predictive fit does not establish biological classes.
- Why the reference must be trained without test outcomes, and why it still is not a causal terrain correction.
- Why a large sample gives precise conditional estimates without resolving selection bias or unidentified physiology.
- What the standard error includes and omits, and why the next useful observation is an external measure of exhaustion onset.

## Primary sources checked

- [Pinned dataset README](https://github.com/llimllib/bostonmarathon/blob/f49b44950cdd13fdeaf1661ba921edbe3dfabff7/README.md), [scraper](https://github.com/llimllib/bostonmarathon/blob/f49b44950cdd13fdeaf1661ba921edbe3dfabff7/dl.py), and [CSV converter](https://github.com/llimllib/bostonmarathon/blob/f49b44950cdd13fdeaf1661ba921edbe3dfabff7/makecsv.py): collection, omissions, fields, and minute rounding.
- [Smyth (2021)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0251513): the actual segment-based wall definition and study population.
- [BAA course map](https://www.baa.org/wp-content/uploads/docs/2025-03/129thBM_SpectatorGuide.pdf): Heartbreak Hill lies inside the late window.
- [Rapoport (2010)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000960): depletion depends on runner characteristics, intensity, stores, and fueling, rather than a universal 90–120-minute switch.
- [Harvard account of BAA's interrupted-race estimation project](https://news.harvard.edu/gazette/story/newsplus/researchers-help-boston-marathon-organizers-plan-for-2014-race/): 2013's interrupted outcomes.

The assignment PDF supplies the requirements; its submission instructions were treated as assignment context, not authorization to submit through Gradescope. The reviewed work was subsequently merged to `main` at the user's request. No Gradescope submission has been made.
