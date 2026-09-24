"""Split-conformal thresholds fitted PER LANGUAGE BUCKET on human-only text.

NON-NEGOTIABLE #3: per-language thresholds are the contribution; never collapse
to a global threshold. Thresholds are derived from human-only calibration sets
(>= 1000 texts per bucket, locked §5) so the empirical FPR is bounded by
alpha in every bucket (the distribution-free guarantee).

Split-conformal quantile: for n human-only calibration scores and target
false-positive rate alpha, tau is the ``ceil((n+1)(1-alpha))``-th smallest
score. Any future human text's score exceeds tau with probability <= alpha
(exchangeability), which is exactly the FPR bound T5 measures. A global
(pooled-calibration) threshold is also fit, per bucket-vs-global comparison
in T5, but it is NEVER the one abstention.py uses (non-negotiable #3).

Contract: ``fit(scores_human_only, bucket)`` / ``threshold(bucket)``.

Serves docs/master-execution-plan.md Phase 3 §3.2.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

_GLOBAL_KEY = "_global"


def conformal_quantile(scores_human_only: "np.ndarray", alpha: float) -> float:
    """The ``ceil((n+1)(1-alpha))``-th smallest of ``scores_human_only``.

    Returns ``inf`` when n is too small for the target alpha (the quantile
    index would exceed n) -- the bound then can't be certified at that alpha,
    and no score should be classified MACHINE by this threshold.
    """
    scores = np.sort(np.asarray(scores_human_only, dtype=float))
    n = len(scores)
    if n == 0:
        return float("inf")
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        return float("inf")
    return float(scores[k - 1])


class ConformalCalibrator:
    """Fits and stores a split-conformal threshold per language bucket."""

    def __init__(self, alpha: float = 0.05, config: dict | None = None) -> None:
        """Args:
        alpha: Target false-positive rate (locked options {0.01, 0.05}).
        config: Loaded ``configs/default.yaml`` (``calibration`` block).
        """
        self.alpha = alpha
        self.config = config or {}
        self.thresholds: dict[str, float] = {}
        self.n_calibration: dict[str, int] = {}

    def fit(self, scores_human_only: "np.ndarray", bucket: str) -> "ConformalCalibrator":
        """Fit the conformal threshold for one bucket from human-only scores.

        Args:
            scores_human_only: Machine-likelihood scores of HUMAN calibration
                texts for this bucket only.
            bucket: Language bucket (``en`` / ``hi`` / ``te`` / ``cm``).

        Returns:
            self.
        """
        self.thresholds[bucket] = conformal_quantile(scores_human_only, self.alpha)
        self.n_calibration[bucket] = len(scores_human_only)
        return self

    def fit_global(self, scores_human_only_pooled: "np.ndarray") -> "ConformalCalibrator":
        """Fit ONE threshold on calibration scores pooled across all buckets.

        For T5's global-vs-per-bucket comparison only; abstention.py never
        uses this (non-negotiable #3 forbids a global operating threshold).
        """
        self.thresholds[_GLOBAL_KEY] = conformal_quantile(scores_human_only_pooled, self.alpha)
        self.n_calibration[_GLOBAL_KEY] = len(scores_human_only_pooled)
        return self

    def threshold(self, bucket: str) -> float:
        """Return the fitted conformal threshold for a bucket (or ``"_global"``)."""
        if bucket not in self.thresholds:
            raise KeyError(f"no conformal threshold fitted for bucket {bucket!r}")
        return self.thresholds[bucket]

    def is_calibrated(self, bucket: str) -> bool:
        """Return True if a threshold has been fitted for the bucket."""
        return bucket in self.thresholds

    def buckets(self) -> list[str]:
        """Return the buckets that have been calibrated (excludes the global key)."""
        return sorted(b for b in self.thresholds if b != _GLOBAL_KEY)
