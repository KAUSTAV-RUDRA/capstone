"""Baseline: Binoculars (Hans, ICML 2024) - cross-perplexity ratio.

Scores the ratio of a text's perplexity under one LM to its cross-perplexity
under a second, closely related LM. Included as a strong zero-shot T1 baseline.

Shared contract: ``score(texts) -> np.ndarray``.

Serves docs/master-execution-plan.md Phase 2 §2.2.1.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class BinocularsDetector:
    """Cross-perplexity (observer/performer) detector."""

    def __init__(
        self,
        observer_model: str = "ai-forever/mGPT",
        performer_model: str = "Qwen/Qwen2.5-0.5B",
        device: str = "cpu",
        config: dict | None = None,
    ) -> None:
        # TODO(phase-2 step-2.2.1): store config; defer model loads.
        raise NotImplementedError

    def load(self) -> None:
        """Load observer + performer models."""
        # TODO(phase-2 step-2.2.1): load both models.
        raise NotImplementedError

    def score(self, texts: list[str]) -> "np.ndarray":
        """Return per-text machine-likelihood scores (higher = more machine-like)."""
        # TODO(phase-2 step-2.2.1): implement Binoculars ratio.
        raise NotImplementedError
