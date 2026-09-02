"""Core metrics: AUROC, F1, FPR, ECE (docs/project-context-master.md §7).

Shared by every experiment; per-bucket by construction.

Serves docs/master-execution-plan.md Phase 2 §2.2 and Phase 3 §3.1-§3.7.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


def auroc(labels: "np.ndarray", scores: "np.ndarray") -> float:
    """Area under the ROC curve (T1, T3)."""
    # TODO(phase-2 step-2.2.1): compute AUROC.
    raise NotImplementedError


def f1_at_threshold(labels: "np.ndarray", scores: "np.ndarray", threshold: float) -> float:
    """F1 at a given decision threshold (T1)."""
    # TODO(phase-2 step-2.2.1): compute F1 at threshold.
    raise NotImplementedError


def false_positive_rate(labels: "np.ndarray", predictions: "np.ndarray") -> float:
    """FPR = fraction of HUMAN texts flagged MACHINE (T5 fairness core)."""
    # TODO(phase-3 step-3.6): compute FPR.
    raise NotImplementedError


def expected_calibration_error(labels: "np.ndarray", probs: "np.ndarray", n_bins: int = 15) -> float:
    """Expected Calibration Error (T6)."""
    # TODO(phase-3 step-3.1): compute ECE.
    raise NotImplementedError


def compute_metrics(
    labels: "np.ndarray",
    scores: "np.ndarray",
    bucket: str | None = None,
) -> dict[str, Any]:
    """Bundle AUROC/F1/FPR (+bucket tag) into one result row for CSV output."""
    # TODO(phase-2 step-2.2.1): assemble a metrics row.
    raise NotImplementedError
