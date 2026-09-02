"""Head C - MuRIL embeddings + shallow head (OPTIONAL, VRAM >= 8GB only).

NON-NEGOTIABLE #9: MuRIL is encoder-only and CANNOT compute perplexity or
curvature. It is used here ONLY to produce embeddings for a shallow classifier.
Skipped entirely when VRAM < 8GB (locked §5 / risk register).

Implements the shared detector contract ``score(texts) -> np.ndarray``.

Serves docs/master-execution-plan.md Phase 2 §2.2.5.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class SemanticHead:
    """MuRIL embedding extractor with a shallow logistic head."""

    def __init__(
        self,
        model_name: str = "google/muril-base-cased",
        device: str = "cpu",
        config: dict | None = None,
    ) -> None:
        """Args:
        model_name: Encoder-only HF id (MuRIL).
        device: ``"cpu"`` or ``"cuda"``.
        config: Loaded ``configs/models.yaml``.
        """
        # TODO(phase-2 step-2.2.5): store config; guard on VRAM >= 8GB.
        raise NotImplementedError

    def load(self) -> None:
        """Load the MuRIL encoder and tokenizer."""
        # TODO(phase-2 step-2.2.5): load encoder + tokenizer.
        raise NotImplementedError

    def embed(self, texts: list[str]) -> "np.ndarray":
        """Return pooled MuRIL embeddings of shape ``(len(texts), hidden)``."""
        # TODO(phase-2 step-2.2.5): compute pooled embeddings.
        raise NotImplementedError

    def fit(self, texts: list[str], labels: "np.ndarray") -> "SemanticHead":
        """Fit the shallow head on embeddings of ``texts``. Returns self."""
        # TODO(phase-2 step-2.2.5): fit shallow classifier on embeddings.
        raise NotImplementedError

    def score(self, texts: list[str]) -> "np.ndarray":
        """Return per-text machine-likelihood scores (shared contract)."""
        # TODO(phase-2 step-2.2.5): embed then classify.
        raise NotImplementedError
