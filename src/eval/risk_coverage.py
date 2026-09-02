"""Risk-coverage analysis for the abstention gate (T6, F1).

Sweeps the abstention band to trace risk (error on non-abstained) against
coverage (fraction not abstained), per bucket, overlaid in Figure F1.

Serves docs/master-execution-plan.md Phase 3 §3.3.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def risk_coverage_curve(
    labels: "np.ndarray",
    probs: "np.ndarray",
    confidences: "np.ndarray",
) -> tuple["np.ndarray", "np.ndarray"]:
    """Return ``(coverage, risk)`` arrays as the confidence cutoff sweeps."""
    # TODO(phase-3 step-3.3): compute risk-coverage curve.
    raise NotImplementedError


def coverage_at_risk(
    labels: "np.ndarray",
    probs: "np.ndarray",
    confidences: "np.ndarray",
    target_risk: float,
) -> float:
    """Maximum coverage achievable while keeping risk <= ``target_risk``."""
    # TODO(phase-3 step-3.3): find coverage at a target risk.
    raise NotImplementedError


def metrics_at_coverage(
    labels: "np.ndarray",
    probs: "np.ndarray",
    confidences: "np.ndarray",
    coverage: float,
) -> dict[str, float]:
    """Accuracy/FPR at a fixed coverage level, e.g. 50/70/90% (T6)."""
    # TODO(phase-3 step-3.3): compute metrics at fixed coverage.
    raise NotImplementedError
