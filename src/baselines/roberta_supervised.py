"""Baseline: XLM-R supervised detector.

A supervised multilingual classifier (XLM-RoBERTa) trained on the seen split -
the supervised point of comparison against our zero-shot + calibrated approach.
NON-NEGOTIABLE #5: no fine-tuning of models above 2B params; XLM-R base fits
the budget. Evaluated on held-out generators only (non-negotiable #4).

Shared contract: ``score(texts) -> np.ndarray``; adds ``fit`` (supervised).

Serves docs/master-execution-plan.md Phase 2 §2.2.1.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class RobertaSupervisedDetector:
    """Supervised XLM-R sequence classifier."""

    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        device: str = "cpu",
        config: dict | None = None,
    ) -> None:
        # TODO(phase-2 step-2.2.1): store config; defer model load.
        raise NotImplementedError

    def load(self) -> None:
        """Load the encoder + classification head."""
        # TODO(phase-2 step-2.2.1): load model + tokenizer.
        raise NotImplementedError

    def fit(self, texts: list[str], labels: "np.ndarray") -> "RobertaSupervisedDetector":
        """Train the classifier on the seen split. Returns self."""
        # TODO(phase-2 step-2.2.1): train supervised head (seen generators only).
        raise NotImplementedError

    def score(self, texts: list[str]) -> "np.ndarray":
        """Return per-text machine-likelihood scores (higher = more machine-like)."""
        # TODO(phase-2 step-2.2.1): predict machine-probability.
        raise NotImplementedError
