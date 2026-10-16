"""Boston Marathon pacing analyses for one turn of Box's loop.

The current submission uses :mod:`.audited_analysis`.  The Bayesian models,
validation routines, and change-point extension are archived exploratory work;
their historical outputs are not the current submission's evidence.  Mixture
components describe a distribution and do not establish physiological states.
"""

from __future__ import annotations

__all__ = ["audited_analysis", "data", "metric", "models", "diagnostics", "validation", "changepoint", "plots"]
