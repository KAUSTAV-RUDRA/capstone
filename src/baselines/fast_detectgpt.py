"""Baseline: Fast-DetectGPT (Bao, ICLR 2024) - conditional-probability curvature.

The reference zero-shot baseline and the mechanism behind Head B. Used first as
a Review-1 smoke test on HC3 English (Phase 1, Day 3-4) and later as a T1
baseline. Head B (:mod:`src.features.curvature`) extends this with script
normalisation and fertility-aware segmentation.

Shared contract: ``score(texts) -> np.ndarray``.

Serves docs/master-execution-plan.md Phase 1 (Day 3-4) and Phase 2 §2.2.1.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class FastDetectGPTDetector:
    """Conditional-probability curvature detector."""

    def __init__(
        self,
        scorer_model: str = "ai-forever/mGPT",
        reference_model: str | None = None,
        device: str = "cpu",
        config: dict | None = None,
    ) -> None:
        """Args:
        scorer_model: Causal LM used for scoring (fallback ``Qwen/Qwen2.5-0.5B``).
        reference_model: Optional separate sampling model; defaults to scorer.
        device: ``"cpu"`` or ``"cuda"``.
        config: Loaded ``configs/models.yaml``.
        """
        # TODO(phase-1 step-1.2/day-3): store config; defer model load.
        raise NotImplementedError

    def load(self) -> None:
        """Load the scorer (and reference) model + tokenizer."""
        # TODO(phase-1 day-3): load model(s).
        raise NotImplementedError

    def score(self, texts: list[str]) -> "np.ndarray":
        """Return per-text curvature scores (higher = more machine-like)."""
        # TODO(phase-1 day-3): implement conditional-probability curvature.
        raise NotImplementedError
