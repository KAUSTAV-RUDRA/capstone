"""Baseline: raw-perplexity threshold detector.

The failure-mode baseline (docs/project-context-master.md §2): raw perplexity
flags low-perplexity non-native writing as AI. Included precisely to show the
fairness gap our curvature+conformal method closes (motivates T5).

Shared contract: ``score(texts) -> np.ndarray``.

Serves docs/master-execution-plan.md Phase 2 §2.2.1.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class PerplexityThresholdDetector:
    """Scores texts by (negative) language-model perplexity."""

    def __init__(
        self,
        scorer_model: str = "ai-forever/mGPT",
        device: str = "cpu",
        config: dict | None = None,
    ) -> None:
        # TODO(phase-2 step-2.2.1): store config; defer model load.
        raise NotImplementedError

    def load(self) -> None:
        """Load the scorer LM + tokenizer."""
        # TODO(phase-2 step-2.2.1): load scorer model.
        raise NotImplementedError

    def score(self, texts: list[str]) -> "np.ndarray":
        """Return per-text machine-likelihood scores (higher = more machine-like)."""
        # TODO(phase-2 step-2.2.1): compute perplexity-based scores.
        raise NotImplementedError
