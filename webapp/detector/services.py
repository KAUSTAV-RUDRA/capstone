"""The ONLY bridge between the Django web layer and the research package.

NON-NEGOTIABLE #7: no other file under ``webapp/`` may import from ``src/``.
``views.py`` calls into this module, never ``src`` directly.

For now every function returns hardcoded placeholder data shaped EXACTLY like
the real pipeline output will be, so the UI is fully clickable before the
research code is implemented.
"""
from __future__ import annotations

from typing import Any

# Bridge imports from the research package (safe to import: src stubs guard all
# heavy third-party imports under TYPE_CHECKING). Not yet exercised.
from src.calibration.abstention import Verdict  # noqa: F401  (wired in phase 3)
from src.data.schema import LANGUAGE_BUCKETS

MODEL_VERSION = "scaffold-0.0.0"


def _resolve_language(language: str | None) -> str:
    """Resolve an incoming language choice to a bucket (placeholder logic)."""
    if language and language in LANGUAGE_BUCKETS:
        return language
    # Real implementation will call src.features.language_id.assign_bucket().
    return "en"


def analyse_text(text: str, language: str | None) -> dict[str, Any]:
    """Analyse a single text and return a decision-support payload.

    The returned dict is the stable contract consumed by ``views.py`` and
    persisted into :class:`webapp.detector.models.Decision`. Every value below
    is a PLACEHOLDER.

    Args:
        text: The submitted text.
        language: One of auto/en/hi/te/cm (or None).

    Returns:
        A dict with keys: verdict, confidence, language, driving_head,
        stylometric_score, curvature_score, semantic_score, explanation,
        model_version.
    """
    # TODO(phase-3 step-3.8): wire to src.pipeline (language_id -> heads ->
    # fusion -> temperature -> conformal -> abstention -> explain). Until then,
    # return a fixed, correctly-shaped placeholder.
    resolved = _resolve_language(language)
    top_features = [
        {"name": "function_word_ratio", "value": 0.061, "contribution": 0.31},
        {"name": "burstiness", "value": -0.42, "contribution": 0.22},
        {"name": "mean_sentence_length", "value": 24.5, "contribution": 0.18},
        {"name": "punctuation_diversity", "value": 0.14, "contribution": -0.11},
        {"name": "mtld", "value": 88.3, "contribution": 0.09},
    ]
    return {
        "verdict": "ABSTAIN",
        "confidence": 0.62,
        "language": resolved,
        "driving_head": "curvature",
        "stylometric_score": 0.48,
        "curvature_score": 0.57,
        "semantic_score": None,
        "explanation": {
            "top_features": top_features,
            "note": "Placeholder output. Not a real detection.",
            "is_decision_support": True,
        },
        "model_version": MODEL_VERSION,
    }


def placeholder_batch_rows() -> list[dict[str, Any]]:
    """Return placeholder rows for the batch results table."""
    # TODO(phase-3 step-3.8): parse the uploaded CSV and analyse each row.
    return [
        {"row": 1, "filename": "sample_en_human.txt", "language": "en",
         "verdict": "HUMAN", "confidence": 0.88, "driving_head": "stylometric"},
        {"row": 2, "filename": "sample_hi_machine.txt", "language": "hi",
         "verdict": "MACHINE", "confidence": 0.91, "driving_head": "curvature"},
        {"row": 3, "filename": "sample_te_human.txt", "language": "te",
         "verdict": "ABSTAIN", "confidence": 0.55, "driving_head": "curvature"},
        {"row": 4, "filename": "sample_cm_machine.txt", "language": "cm",
         "verdict": "MACHINE", "confidence": 0.79, "driving_head": "fusion"},
    ]
