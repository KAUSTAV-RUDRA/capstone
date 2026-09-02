"""Head A - stylometric feature extraction (docs/project-context-master.md §3).

Features: function words, burstiness, punctuation, TTR/MTLD, POS n-grams,
syntactic depth. Language-aware feature norms per bucket.

Implements the shared detector contract ``score(texts) -> np.ndarray`` and the
extractor contract ``fit`` / ``transform`` / ``feature_names``.

Serves docs/master-execution-plan.md Phase 1 (Day 2) and Phase 2 §2.2.2.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class StylometricExtractor:
    """Extracts and (optionally) scores stylometric features."""

    def __init__(self, language: str | None = None, config: dict | None = None) -> None:
        """Args:
        language: Optional bucket for language-aware norms (``en``/``hi``/``te``/``cm``).
        config: Loaded ``configs/default.yaml``.
        """
        # TODO(phase-2 step-2.2.2): init feature registry and per-bucket norms.
        raise NotImplementedError

    def fit(self, texts: list[str], languages: list[str] | None = None) -> "StylometricExtractor":
        """Fit feature normalisers/vocabularies on training texts.

        Args:
            texts: Training texts.
            languages: Optional parallel list of buckets for per-language norms.

        Returns:
            self.
        """
        # TODO(phase-2 step-2.2.2): fit norms/vocab; return self.
        raise NotImplementedError

    def transform(self, texts: list[str]) -> "np.ndarray":
        """Return a feature matrix of shape ``(len(texts), n_features)``."""
        # TODO(phase-2 step-2.2.2): compute stylometric feature matrix.
        raise NotImplementedError

    def fit_transform(self, texts: list[str], languages: list[str] | None = None) -> "np.ndarray":
        """Convenience: :meth:`fit` then :meth:`transform`."""
        # TODO(phase-2 step-2.2.2): fit then transform.
        raise NotImplementedError

    def feature_names(self) -> list[str]:
        """Return the ordered feature names matching :meth:`transform` columns."""
        # TODO(phase-2 step-2.2.2): return ordered feature names.
        raise NotImplementedError

    def score(self, texts: list[str]) -> "np.ndarray":
        """Return per-text machine-likelihood scores (higher = more machine-like).

        Shared head/baseline contract. Typically a light head classifier over
        :meth:`transform` features.
        """
        # TODO(phase-2 step-2.2.2): map features -> machine-likelihood score.
        raise NotImplementedError
