# Probabilistic models

Coursework and reader reports for Fall 2026.

- [Course reading index](READINGS.md): all readings from the course page, organized by topic.
- [Reader reports](reader_reports/): LaTeX reports.
- [Homeworks](homeworks/README.md): PDFs, source code, and reproduction instructions.
- [Report template](template/report.tex): shared template and preamble.
- [Official reading list](https://www.cs.columbia.edu/~blei/fogm/2026F/readings.html).

To refresh the reading index from the course page:

```bash
python3 scripts/update_readings.py
```

The refresh script uses Python's standard library and needs no additional packages.
Keep reading notes and progress in separate files; `READINGS.md` is regenerated on refresh.

## Homework 1

- [Written submission](homeworks/hw1.pdf) (four pages).
- [Submitted code and reproduction instructions](homeworks/hw1/README.md).
- [Saved result summaries](homeworks/hw1/results/).
- [Correctness audit](homeworks/hw1-audit.md).

All code used for the submitted analysis lives in `homeworks/hw1/`. Its README
provides a minimal pinned Python environment, data download, analysis, independent
checks, expected results, and optional PDF compilation. For the GitHub submission
option, select this repository and branch **main**.

Q3 compares Gaussian, skew-normal, and two-Gaussian mixture densities using a
fixed held-out sample and training-only preprocessing. Better predictive fit
does not identify physiological wall states: all four primary mixture densities
are unimodal. Earlier Bayesian and change-point experiments remain under
[`experiments/marathon_wall/`](experiments/marathon_wall/outputs/README.md) as
historical work; they do not supply the submitted results.
