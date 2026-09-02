"""Fairness audit: FPR by bucket and by L1/L2 band (T5).

Quantifies the native/non-native false-positive disparity for our system vs
baselines - the headline fairness result (RQ4, non-negotiable #3 rationale).

Serves docs/master-execution-plan.md Phase 3 §3.6.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


def fpr_by_bucket(
    labels: "np.ndarray",
    predictions: "np.ndarray",
    buckets: list[str],
) -> dict[str, float]:
    """FPR per language bucket (en / hi / te / cm)."""
    # TODO(phase-3 step-3.6): compute per-bucket FPR.
    raise NotImplementedError


def fpr_by_l1_band(
    labels: "np.ndarray",
    predictions: "np.ndarray",
    l1_bands: list[str],
) -> dict[str, float]:
    """FPR per writer L1/L2 band (native / non_native / unknown)."""
    # TODO(phase-3 step-3.6): compute per-L1-band FPR.
    raise NotImplementedError


def disparity(fpr_by_group: dict[str, float]) -> float:
    """Max-minus-min FPR across groups (the disparity we aim to reduce)."""
    # TODO(phase-3 step-3.6): compute disparity metric.
    raise NotImplementedError


def audit(
    labels: "np.ndarray",
    predictions: "np.ndarray",
    buckets: list[str],
    l1_bands: list[str],
    system_name: str,
) -> "pd.DataFrame":
    """Assemble the full T5 fairness table for one system."""
    # TODO(phase-3 step-3.6): build the T5 fairness DataFrame.
    raise NotImplementedError
