"""Strip generation artefacts from machine text.

Removes preambles ("Sure! Here is..."), refusals, and markdown so machine
text is not trivially separable by surface cues. Reports the percentage of
text stripped.

Serves docs/master-execution-plan.md Phase 2 §2.1.4.
"""
from __future__ import annotations

from src.data.schema import Sample


def clean_text(text: str) -> tuple[str, float]:
    """Strip artefacts from one text.

    Args:
        text: Raw machine (or human) text.

    Returns:
        ``(cleaned_text, fraction_stripped)``.
    """
    # TODO(phase-2 step-2.1.4): remove preambles/refusals/markdown.
    raise NotImplementedError


def clean_samples(samples: list[Sample]) -> tuple[list[Sample], dict[str, float]]:
    """Clean a list of samples and return per-bucket stripped-percentage stats.

    Args:
        samples: Samples to clean in place (returned as new list).

    Returns:
        ``(cleaned_samples, {bucket: mean_fraction_stripped})``.
    """
    # TODO(phase-2 step-2.1.4): apply clean_text across the corpus.
    raise NotImplementedError
