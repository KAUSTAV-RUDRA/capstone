"""Head B - Fast-DetectGPT conditional-probability curvature.

NON-NEGOTIABLE #2: uses curvature (a second-order property), NOT raw perplexity.
Scorer model = mGPT-1.3B (fallback Qwen2.5-0.5B), a causal LM (locked §5).
NON-NEGOTIABLE #9: MuRIL is encoder-only and must NEVER be used here.
Includes script normalisation and fertility-aware segmentation for Romanised
code-mix (feeds Patent 2).

Implements the shared detector contract ``score(texts) -> np.ndarray``.

Serves docs/master-execution-plan.md Phase 1 (Day 3) and Phase 2 §2.2.3-§2.2.4.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class CurvatureScorer:
    """Fast-DetectGPT curvature scorer over a multilingual causal LM."""

    def __init__(
        self,
        scorer_model: str = "ai-forever/mGPT",
        device: str = "cpu",
        load_in_8bit: bool = False,
        config: dict | None = None,
    ) -> None:
        """Args:
        scorer_model: HF id of the causal scorer (fallback ``Qwen/Qwen2.5-0.5B``).
        device: ``"cpu"`` or ``"cuda"``.
        load_in_8bit: 8-bit loading to respect the 4GB VRAM floor (non-neg #5).
        config: Loaded ``configs/models.yaml``.
        """
        # TODO(phase-2 step-2.2.3): store config; defer model load to load().
        raise NotImplementedError

    def load(self) -> None:
        """Load the tokenizer and causal LM (lazily, once)."""
        # TODO(phase-2 step-2.2.3): load scorer model + tokenizer.
        raise NotImplementedError

    def curvature(self, text: str) -> float:
        """Compute the Fast-DetectGPT conditional-probability curvature for one text."""
        # TODO(phase-2 step-2.2.3): implement conditional-probability curvature.
        raise NotImplementedError

    def score(self, texts: list[str]) -> "np.ndarray":
        """Return per-text curvature-based machine-likelihood scores.

        Shared head/baseline contract; higher = more machine-like.
        """
        # TODO(phase-2 step-2.2.3): batch curvature over texts.
        raise NotImplementedError
