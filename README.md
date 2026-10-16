# Probabilistic models

Coursework and reader reports for Fall 2026.

- [Course reading index](READINGS.md): all readings from the course page, organized by topic.
- [Reader reports](reader_reports/): LaTeX reports.
- [Homeworks](homeworks/): LaTeX problem sets.
- [Report template](template/report.tex): shared template and preamble.
- [Official reading list](https://www.cs.columbia.edu/~blei/fogm/2026F/readings.html).

To refresh the reading index from the course page:

```bash
python3 scripts/update_readings.py
```

The refresh script uses Python's standard library and needs no additional packages.
Keep reading notes and progress in separate files; `READINGS.md` is regenerated on refresh.

## Homework 1: audited marathon analysis

[The revised homework](homeworks/hw1.pdf) keeps the verified Q1/Q2 derivations and
uses a reproducible density comparison for Q3. The question is whether a
two-Gaussian mixture predicts late-race pace ratios better than a Gaussian or
skew-normal, and whether that supports a physiological interpretation. It does
predict better on the held-out sample, but all four fitted group mixture
densities are unimodal; component weights are not wall prevalence.

The primary implementation is
[`audited_analysis.py`](experiments/marathon_wall/audited_analysis.py).
It uses a fixed stratified 80/20 split, learns the empirical pace reference only
on training records, and fits all three models by maximum likelihood within
recorded-sex/year groups. The reference is descriptive centering, not an
identified terrain correction. The input archive is incomplete, especially in
2013, and the original draft already explored these data; the refit is an
internal diagnostic, not independent confirmation.

From the repository root:

```bash
uv sync --frozen
uv run python -m experiments.marathon_wall.fetch_data --data-dir data/boston
uv run python -m experiments.marathon_wall.audited_analysis --data-dir data/boston
uv run python -m experiments.marathon_wall.check_derivations
uv run python scripts/verify_hw1_results.py
uv run python -m unittest experiments.marathon_wall.test_data experiments.marathon_wall.test_audited_analysis experiments.marathon_wall.test_make_tex_macros -v
cd homeworks
latexmk -pdf -interaction=nonstopmode -halt-on-error hw1.tex
```

`uv.lock` records the full project environment. The audited analysis itself needs
only NumPy, pandas, SciPy and Matplotlib; the independent derivation check uses
the Python standard library. Compiling uses the existing LaTeX preamble and
requires its packages (including newtx, mathalpha, thmtools, and TikZ).

The downloader pins the source repository commit and verifies SHA-256 hashes,
then writes `data/boston/source_manifest.json`. Race files remain uncommitted.
The analysis writes cohort accounting, fitted parameters, versions, reference
curves and held-out comparisons to
[`audit_outputs/`](experiments/marathon_wall/audit_outputs/), plus the figure and
letter-only numeric macros used by the PDF. The detailed `cohort.csv` and
`heldout_scores.csv` are generated locally and included in the review bundle;
compact result summaries are committed. Every Q3 fitted result comes from these
outputs. The PDF is four pages: one each for Q1/Q2 and two for Q3.

The earlier hierarchical Bayesian, simulation, and change-point work is preserved
in the other experiment modules and
[`outputs/`](experiments/marathon_wall/outputs/README.md). Those historical results
are not used in the revised submission. In particular, the historical function
named `smyth_rule` does not implement Smyth's published rule, its LOO comparison
freezes an all-data reference, and its saved change-point fit is unidentified.
See [the audit](homeworks/hw1-audit.md) for findings, evidence, and interpretation
checks to understand before submitting.
