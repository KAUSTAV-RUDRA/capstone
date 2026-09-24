"""Risk-coverage analysis for the abstention gate (T6, F1).

Sweeps the abstention band to trace risk (error on non-abstained) against
coverage (fraction not abstained), per bucket, overlaid in Figure F1.

Confidence convention: ``confidence = max(prob, 1 - prob)``, matching
:meth:`src.calibration.abstention.AbstentionGate.decide` -- the sample the
fused model is least sure about (prob near 0.5) is the first one abstained.

Serves docs/master-execution-plan.md Phase 3 §3.3.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass


def risk_coverage_curve(
    labels: "np.ndarray",
    probs: "np.ndarray",
    confidences: "np.ndarray",
) -> tuple["np.ndarray", "np.ndarray"]:
    """Return ``(coverage, risk)`` arrays as the confidence cutoff sweeps.

    Samples are ranked by ``confidences`` descending (most confident kept
    first); ``coverage[i]`` is the fraction of samples covered by keeping the
    ``i+1`` most confident, ``risk[i]`` is the error rate (predicted at 0.5)
    among those.
    """
    labels = np.asarray(labels)
    probs = np.asarray(probs, dtype=float)
    confidences = np.asarray(confidences, dtype=float)
    n = len(labels)
    if n == 0:
        return np.array([]), np.array([])
    order = np.argsort(-confidences)
    labels_sorted = labels[order]
    preds_sorted = (probs[order] >= 0.5).astype(int)
    correct = (preds_sorted == labels_sorted).astype(float)
    cum_correct = np.cumsum(correct)
    counts = np.arange(1, n + 1)
    coverage = counts / n
    risk = 1.0 - cum_correct / counts
    return coverage, risk


def coverage_at_risk(
    labels: "np.ndarray",
    probs: "np.ndarray",
    confidences: "np.ndarray",
    target_risk: float,
) -> float:
    """Maximum coverage achievable while keeping risk <= ``target_risk``."""
    coverage, risk = risk_coverage_curve(labels, probs, confidences)
    if len(coverage) == 0:
        return 0.0
    ok = risk <= target_risk
    if not ok.any():
        return 0.0
    return float(coverage[ok].max())


def metrics_at_coverage(
    labels: "np.ndarray",
    probs: "np.ndarray",
    confidences: "np.ndarray",
    coverage: float,
) -> dict[str, float]:
    """Accuracy/FPR at a fixed coverage level, e.g. 50/70/90% (T6).

    Keeps the ``round(coverage * n)`` most-confident samples and reports
    accuracy and FPR (predicted at 0.5) among them.
    """
    labels = np.asarray(labels)
    probs = np.asarray(probs, dtype=float)
    confidences = np.asarray(confidences, dtype=float)
    n = len(labels)
    n_keep = max(1, round(coverage * n)) if n else 0
    if n == 0:
        return {"coverage": coverage, "n_covered": 0, "accuracy": float("nan"), "fpr": float("nan")}
    order = np.argsort(-confidences)[:n_keep]
    y = labels[order]
    preds = (probs[order] >= 0.5).astype(int)
    acc = float((preds == y).mean())
    human = y == 0
    fpr = float((preds[human] == 1).mean()) if human.any() else float("nan")
    return {"coverage": float(coverage), "n_covered": int(n_keep), "accuracy": acc, "fpr": fpr}
