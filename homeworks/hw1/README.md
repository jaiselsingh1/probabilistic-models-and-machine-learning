# Homework 1: code and reproduction

This directory contains the code used for the [submitted PDF](../hw1.pdf).
Questions 1 and 2 have numerical checks of the written derivations. Question 3
fits and checks Gaussian, skew-normal, and two-Gaussian mixture densities for
Boston Marathon pacing records. The implementation, tests, and compact results
are included here; earlier exploratory models remain in
[`experiments/marathon_wall/`](../../experiments/marathon_wall/) and are not needed
to reproduce the submission.

For the GitHub submission form, select the repository
`jaiselsingh1/probabilistic-models-and-machine-learning` and branch **`main`**.
That submits the selected repository branch; the homework PDF is
`homeworks/hw1.pdf` and its code is `homeworks/hw1/`.

## Set up

Use Python **3.11**. The requirements file pins the analysis environment and its
dependencies; no Bayesian sampling packages are required. From a terminal:

```bash
git clone --branch main https://github.com/jaiselsingh1/probabilistic-models-and-machine-learning.git
cd probabilistic-models-and-machine-learning
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r homeworks/hw1/requirements-test.txt
```

If the repository is already cloned, start from its root on `main` and run the
environment commands. On Windows, activate the environment with
`.venv\Scripts\Activate.ps1` in PowerShell instead of `source`.

## Reproduce the results

Run these commands from the repository root with the environment activated:

```bash
python -m homeworks.hw1.fetch_data
python -m homeworks.hw1.audited_analysis
python -m homeworks.hw1.check_derivations
python -m homeworks.hw1.verify_results
python -m pytest homeworks/hw1 -q
```

Allow a few minutes for the analysis and checks; runtime varies by machine.
Installation and download time depend on the connection. The downloader retrieves the two
public 2013/2014 CSV files, verifies their SHA-256 hashes, and records their
source in `homeworks/hw1/data/boston/source_manifest.json`. It also verifies existing local
copies, so repeating the command does not require downloading them again.

The analysis uses seed `20261016` and a fixed 80/20 split within recorded-sex/year
groups. It learns the pace reference from training records only, fits all three
models, and regenerates the primary comparison, sensitivity results, PDF figure,
and numerical LaTeX macros. The `q25_2014` sensitivity keeps the original 2014
train/test assignment but learns both the reference and models from 2014
training records alone. The independent verifier then reconstructs the
held-out responses, densities, score differences, standard errors, and mixture
mode checks from saved outputs without importing the fitting code. The Q1/Q2
checks use numerical integration as a complement to the proofs in the PDF.

Defaults are resolved from the code's location:

| Output | Location relative to the repository root |
| --- | --- |
| Downloaded race files and source manifest | `homeworks/hw1/data/boston/` |
| Fitted parameters, input hashes, environment versions | `homeworks/hw1/results/results.json` |
| Held-out comparisons and standard errors | `homeworks/hw1/results/comparisons.csv` |
| Exclusions and training reference curves | `homeworks/hw1/results/hygiene.csv`, `reference_*.csv` |
| Component scale ratios and weighted responsibility crossovers | `homeworks/hw1/results/mixture_diagnostics.csv` |
| Signed bulk/tail log-score contributions | `homeworks/hw1/results/score_gain_partition.csv` |
| Training year composition near the reference kink | `homeworks/hw1/results/training_year_composition.csv` |
| Diagnostic definitions and full-precision values | `homeworks/hw1/results/diagnostics.json` |
| Eligible records and detailed held-out scores | `homeworks/hw1/results/cohort.csv`, `heldout_scores.csv` |
| Independent verification report | `homeworks/hw1/results/verification.json` |
| Figure and numbers used by the write-up | `homeworks/figures/audit_wall.pdf`, `audit_numbers.tex` |

The raw race files and detailed pointwise CSVs are regenerated locally and
ignored by Git. The compact results, figure, and macros are committed. Run the
analysis before the independent verifier, because the verifier needs those
detailed CSVs. To use other locations, the downloader accepts `--data-dir`, the
analysis accepts `--data-dir`, `--out-dir`, and `--fig-dir`, and the verifier
accepts `--out-dir` and `--fig-dir`. For analysis without pytest, install
`requirements.txt`; `requirements-test.txt` adds the pinned test dependencies.

## Expected checks

- Cohort: **48,148** raw records, **457** excluded, **47,691** eligible.
- Split: **38,152** training and **9,539** held-out records.
- Primary mixture-minus-skew-normal gain: **244.9 ± 48.1 nats** total,
  or **0.0257 ± 0.0050 nats per record**.
- Primary mixture-minus-Gaussian gain: **1,151.5 ± 72.7 nats** total.
- With both reference and models fit on 2014 training records alone:
  **169.8 ± 41.4 nats** over the skew-normal on the original 2014 holdout.
- Using each group's training 5th/95th percentiles as fixed boundaries, the
  primary gain decomposes into **−28.0 nats** in the central interval,
  **+196.0** in the right tail, and **+76.9** in the left tail. These signed
  contributions sum to the net gain; the visual mode-height difference does
  not imply a positive net gain throughout the bulk.
- Component SD ratios range **2.117–2.463** (variance ratios **4.481–6.065**).
  Weighted responsibility crosses 50% on the left at centered responses
  **−0.159 to −0.135** across groups. Negative centered response means less
  slowing than the reference; actual speeding up requires raw log ratio < 0.
- All **four** primary mixture densities are unimodal.
- The verifier independently checks every specification-by-held-out row,
  comparison, training reference, diagnostic, and generated numerical macro
  and reports `"status": "passed"`; Q1/Q2 checks report the same status,
  and pytest reports all tests passed.

The ± values are one paired held-out standard error, conditional on the fitted
models and fixed stratum sizes. They are not posterior intervals. Small
floating-point differences can occur across platforms. Mixture components
describe the observed distribution; their weights do not estimate the fraction
of runners experiencing a physiological state. See the PDF for the data and
interpretation limitations.

## Optional: rebuild the PDF

The committed PDF can be read without installing LaTeX. After regenerating the
analysis, rebuild it using a TeX distribution with `latexmk` and the packages
used by [`preamble.tex`](../preamble.tex), including newtx, mathalpha, thmtools,
and TikZ:

```bash
cd homeworks
latexmk -pdf -interaction=nonstopmode -halt-on-error hw1.tex
```

This writes the four-page `homeworks/hw1.pdf`. PDF compilation is separate from
running and verifying the Python code.
