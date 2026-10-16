# Historical exploratory outputs — not used in the revised submission

These committed tables came from the pre-audit hierarchical Bayesian analysis at commit `2149a2094257a3b117bb42273ae444b90ded2eb3`. They are retained to preserve the original work, not relabeled as new results. Sampler draws were not committed, and these historical numerical outputs were not independently regenerated during the audit.

Use `../audit_outputs/` and `../audited_analysis.py` for the revised Q3.

Known limitations of these historical outputs:

- `smyth_rule.csv` implements a 20% average slowdown across 25 km–finish relative to 5–20 km. This is NOT Smyth (2021)'s published 25% slowdown across at least 5 km after 20 km; its agreement statistics are not a literature replication.
- `decision_rule.csv` calibrates decision thresholds to external incidence rates. Those are different populations and definitions; the resulting loss ratios do not validate a physiological label.
- The reference was estimated on all records before PSIS-LOO. LOO does not validate this preprocessing step, and skew-normal `max_khat=0.7249` triggers a warning. No MCMC convergence summary was saved.
- Mixture weights are not identified physiological prevalence or a lower bound on it. Two mixture components need not give two density modes.
- Three intervals covering three planted values in one simulated dataset are not a coverage study; one skew-normal simulation does not validate against all unimodal laws.
- `changepoint_summary.csv` belongs to an unidentified exploratory model. Its saved responsibilities used a geometric mean; the source now fixes this arithmetic, but this historical file has not been regenerated. Its location quantiles summarize athlete posterior means, not posterior credible intervals.

The original figures `wall_data.pdf`, `wall_model.pdf`, `wall_invariance.pdf` and `wall_numbers.tex` are historical too. `wall_numbers.tex` has repaired LaTeX command names, with numerical values preserved. Only `audit_wall.pdf` and `audit_numbers.tex` feed the revised homework.
