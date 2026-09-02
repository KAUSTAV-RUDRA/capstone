"""Per-decision explanation for the reviewer report.

NON-NEGOTIABLE #8: decision support, never an automatic accusation. Produces
the interpretable payload shown on the Django result page: which head drove the
decision, top-5 stylometric features, and the curvature score.

This module is imported by the web layer ONLY through
``webapp/detector/services.py`` (non-negotiable #7).

Serves docs/master-execution-plan.md Phase 3 §3.8.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


def explain_decision(
    head_scores: dict[str, float],
    feature_values: dict[str, float],
    feature_contributions: dict[str, float],
    top_k: int = 5,
) -> dict[str, Any]:
    """Build the explanation payload for a single decision.

    Args:
        head_scores: Per-head scores, e.g. ``{"stylometric": .., "curvature": ..}``.
        feature_values: Stylometric feature values for the sample.
        feature_contributions: Signed contribution of each feature to the score.
        top_k: How many top features to surface (default 5).

    Returns:
        A JSON-serialisable dict: driving head, top-k features, curvature score,
        and a decision-support disclaimer flag.
    """
    # TODO(phase-3 step-3.8): assemble driving head + top-k feature attributions.
    raise NotImplementedError


def top_features(
    feature_names: list[str],
    contributions: "np.ndarray",
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Return the top-k features by absolute contribution as name/value dicts."""
    # TODO(phase-3 step-3.8): rank and format top features.
    raise NotImplementedError
