"""Split-conformal thresholds fitted PER LANGUAGE BUCKET on human-only text.

NON-NEGOTIABLE #3: per-language thresholds are the contribution; never collapse
to a global threshold. Thresholds are derived from human-only calibration sets
(>= 1000 texts per bucket, locked §5) so the empirical FPR is bounded by
alpha in every bucket (the distribution-free guarantee).

Contract: ``fit(scores_human_only, bucket)`` / ``threshold(bucket)``.

Serves docs/master-execution-plan.md Phase 3 §3.2.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class ConformalCalibrator:
    """Fits and stores a split-conformal threshold per language bucket."""

    def __init__(self, alpha: float = 0.05, config: dict | None = None) -> None:
        """Args:
        alpha: Target false-positive rate (locked options {0.01, 0.05}).
        config: Loaded ``configs/default.yaml`` (``calibration`` block).
        """
        # TODO(phase-3 step-3.2): init per-bucket threshold store with alpha.
        raise NotImplementedError

    def fit(self, scores_human_only: "np.ndarray", bucket: str) -> "ConformalCalibrator":
        """Fit the conformal threshold for one bucket from human-only scores.

        Args:
            scores_human_only: Machine-likelihood scores of HUMAN calibration
                texts for this bucket only.
            bucket: Language bucket (``en`` / ``hi`` / ``te`` / ``cm``).

        Returns:
            self.
        """
        # TODO(phase-3 step-3.2): compute the (1 - alpha) conformal quantile.
        raise NotImplementedError

    def threshold(self, bucket: str) -> float:
        """Return the fitted conformal threshold for a bucket."""
        # TODO(phase-3 step-3.2): return stored per-bucket threshold.
        raise NotImplementedError

    def is_calibrated(self, bucket: str) -> bool:
        """Return True if a threshold has been fitted for the bucket."""
        # TODO(phase-3 step-3.2): report calibration status.
        raise NotImplementedError

    def buckets(self) -> list[str]:
        """Return the buckets that have been calibrated."""
        # TODO(phase-3 step-3.2): list calibrated buckets.
        raise NotImplementedError
