"""Temperature scaling of fused outputs, fitted per language bucket.

Calibrates the fusion probability before the conformal threshold is applied,
so ECE is low in every bucket (measured in T6). Fitted separately per bucket
(en / hi / te / cm).

The frozen ``cal`` split (locked §5, non-negotiable #6) is human-only by
construction (it exists to fit the conformal threshold from human-only
scores), so it carries no label variation to fit a temperature against.
Temperature is therefore fit on the TRAIN split's fused logits, per bucket --
the only split besides ``test`` with both classes. This is a documented
limitation (train-set-optimistic), not blocking: T6 evaluates ECE
before/after on the TEST split, which the fit never touches.

Serves docs/master-execution-plan.md Phase 3 §3.1.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

_GLOBAL_KEY = "_global"


def _sigmoid(x: "np.ndarray") -> "np.ndarray":
    return 1.0 / (1.0 + np.exp(-x))


class TemperatureScaler:
    """One temperature per bucket, fitted by NLL minimisation over logits."""

    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}
        self.temperatures: dict[str, float] = {}

    def fit(
        self,
        logits: "np.ndarray",
        labels: "np.ndarray",
        bucket: str | None = None,
    ) -> "TemperatureScaler":
        """Fit the temperature for one bucket by minimising NLL. Returns self."""
        from scipy.optimize import minimize_scalar

        logits = np.asarray(logits, dtype=float)
        labels = np.asarray(labels, dtype=float)
        if len(set(labels.tolist())) < 2:
            raise ValueError("temperature fitting needs both classes in `labels`")

        def nll(t: float) -> float:
            t = max(t, 1e-3)
            p = np.clip(_sigmoid(logits / t), 1e-7, 1 - 1e-7)
            return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))

        result = minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded")
        key = bucket if bucket is not None else _GLOBAL_KEY
        self.temperatures[key] = float(result.x)
        return self

    def transform(self, logits: "np.ndarray", bucket: str | None = None) -> "np.ndarray":
        """Apply the bucket's temperature and return calibrated probabilities."""
        key = bucket if bucket is not None else _GLOBAL_KEY
        if key not in self.temperatures:
            raise KeyError(f"no temperature fitted for bucket {key!r}")
        t = self.temperatures[key]
        return _sigmoid(np.asarray(logits, dtype=float) / t)

    def temperature(self, bucket: str | None = None) -> float:
        """Return the fitted temperature for a bucket."""
        key = bucket if bucket is not None else _GLOBAL_KEY
        return self.temperatures[key]
