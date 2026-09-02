"""Baseline: DetectGPT (Mitchell 2023) - perturbation-based curvature.

Perturbs each text with a mask-filling model and measures the log-prob drop.
Slower than Fast-DetectGPT; included as a T1 baseline.

Shared contract: ``score(texts) -> np.ndarray``.

Serves docs/master-execution-plan.md Phase 2 §2.2.1.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class DetectGPTDetector:
    """Perturbation-discrepancy detector."""

    def __init__(
        self,
        scorer_model: str = "ai-forever/mGPT",
        perturbation_model: str = "google/mt5-small",
        n_perturbations: int = 20,
        device: str = "cpu",
        config: dict | None = None,
    ) -> None:
        # TODO(phase-2 step-2.2.1): store config; defer model loads.
        raise NotImplementedError

    def load(self) -> None:
        """Load scorer + perturbation models."""
        # TODO(phase-2 step-2.2.1): load both models.
        raise NotImplementedError

    def score(self, texts: list[str]) -> "np.ndarray":
        """Return per-text machine-likelihood scores (higher = more machine-like)."""
        # TODO(phase-2 step-2.2.1): implement perturbation-discrepancy scoring.
        raise NotImplementedError
