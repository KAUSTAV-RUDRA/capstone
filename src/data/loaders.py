"""Corpus loaders for human, machine, held-out, and adversarial splits.

Loads records conforming to :mod:`src.data.schema` from ``data/raw`` and
``data/processed`` per the frozen ``splits.json``.

Serves docs/master-execution-plan.md Phase 2 §2.1.2 (human/machine text).
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from src.data.schema import Sample

if TYPE_CHECKING:
    import pandas as pd


def load_split(
    split: str,
    config: dict | None = None,
    bucket: str | None = None,
) -> list[Sample]:
    """Load all samples for a given split (``train`` / ``cal`` / ``test``).

    Args:
        split: One of :data:`src.data.schema.SPLITS`.
        config: Loaded ``configs/data.yaml`` (or None to use defaults).
        bucket: Optional language filter (``en`` / ``hi`` / ``te`` / ``cm``).

    Returns:
        The samples belonging to that split (and bucket, if given).
    """
    # TODO(phase-2 step-2.1.2): read splits.json and load matching records.
    raise NotImplementedError


def load_calibration_human_only(bucket: str, config: dict | None = None) -> list[Sample]:
    """Load the human-only calibration set for one bucket.

    Conformal thresholds are fitted on human-only text (non-negotiable #3);
    at least 1000 human texts per bucket are required (locked decision §5).

    Args:
        bucket: Language bucket.
        config: Loaded ``configs/data.yaml``.

    Returns:
        Human-only calibration samples for the bucket.
    """
    # TODO(phase-3 step-3.2): load human-only calibration texts per bucket.
    raise NotImplementedError


def load_dataframe(config: dict | None = None) -> "pd.DataFrame":
    """Load the entire corpus as a DataFrame with SCHEMA_FIELDS columns."""
    # TODO(phase-2 step-2.1.2): assemble full corpus DataFrame.
    raise NotImplementedError


def texts_and_labels(samples: list[Sample]) -> tuple[list[str], list[int]]:
    """Split a list of samples into parallel ``(texts, labels)`` lists."""
    # TODO(phase-2 step-2.1.2): unzip samples.
    raise NotImplementedError
