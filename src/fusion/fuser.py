"""Fuser - logistic regression (or shallow GBM) over head scores.

Interpretability is a feature (locked §5): logistic regression first; GBM only
if it wins on calibration AUROC. Consumes per-head scores from Heads A/B(/C)
and emits a single machine-probability that the calibration layer consumes.

Contract: ``fit(head_scores, labels)`` / ``predict_proba(head_scores)`` where
``head_scores`` has shape ``(n_samples, n_heads)``.

Serves docs/master-execution-plan.md Phase 2 §2.2.6.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class Fuser:
    """Interpretable fusion model over head outputs."""

    def __init__(self, method: str = "logistic", config: dict | None = None) -> None:
        """Args:
        method: ``"logistic"`` (default) or ``"gbm"``.
        config: Loaded ``configs/default.yaml`` (``fusion`` block).
        """
        # TODO(phase-2 step-2.2.6): init logistic/GBM backend.
        raise NotImplementedError

    def fit(self, head_scores: "np.ndarray", labels: "np.ndarray") -> "Fuser":
        """Fit the fusion model.

        Args:
            head_scores: Shape ``(n_samples, n_heads)`` of per-head scores.
            labels: Shape ``(n_samples,)`` with 0 = human, 1 = machine.

        Returns:
            self.
        """
        # TODO(phase-2 step-2.2.6): fit fusion model; return self.
        raise NotImplementedError

    def predict_proba(self, head_scores: "np.ndarray") -> "np.ndarray":
        """Return machine-probability per sample, shape ``(n_samples,)``."""
        # TODO(phase-2 step-2.2.6): predict fused machine-probability.
        raise NotImplementedError

    def predict(self, head_scores: "np.ndarray") -> "np.ndarray":
        """Return hard 0/1 predictions (pre-abstention, for diagnostics only)."""
        # TODO(phase-2 step-2.2.6): threshold predict_proba at 0.5.
        raise NotImplementedError

    def coefficients(self) -> dict[str, float]:
        """Return per-head weights (interpretability; logistic method)."""
        # TODO(phase-2 step-2.2.6): expose learned per-head coefficients.
        raise NotImplementedError
