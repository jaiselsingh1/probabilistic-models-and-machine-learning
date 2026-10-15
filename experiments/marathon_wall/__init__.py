"""Is the marathon wall a latent regime, or the tail of one fatigue distribution?

One turn of Box's loop on the Boston Marathon 5K checkpoint grid: build a
topographically de-confounded deceleration metric (:mod:`.metric`) from cleaned
chip splits (:mod:`.data`), fit a two-component mixture against a single
Gaussian and a unimodal skew-normal control (:mod:`.models`), criticise the fit
with PSIS-LOO and calibration (:mod:`.diagnostics`), probe whether the latent
component behaves like a physiological state (:mod:`.validation`), and hand the
next turn a change-point model over all nine splits (:mod:`.changepoint`).
"""

from __future__ import annotations

__all__ = ["data", "metric", "models", "diagnostics", "validation", "changepoint", "plots"]
