"""Core metrics: AUROC, F1, FPR, ECE (docs/project-context-master.md §7).

Shared by every experiment; per-bucket by construction.

Serves docs/master-execution-plan.md Phase 2 §2.2 and Phase 3 §3.1-§3.7.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    pass


def auroc(labels: "np.ndarray", scores: "np.ndarray") -> float:
    """Area under the ROC curve (T1, T3)."""
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(labels, scores))


def f1_at_threshold(labels: "np.ndarray", scores: "np.ndarray", threshold: float) -> float:
    """F1 at a given decision threshold (T1)."""
    from sklearn.metrics import f1_score

    preds = (np.asarray(scores, dtype=float) >= threshold).astype(int)
    return float(f1_score(np.asarray(labels), preds, zero_division=0))


def best_f1_threshold(labels: "np.ndarray", scores: "np.ndarray") -> float:
    """The score threshold that maximises F1 on (labels, scores).

    Used to pick a decision threshold on the TRAIN split (the only split besides
    test with both classes -- the frozen ``cal`` split is human-only, locked §5),
    then applied unchanged to the test split for T1/T2's F1 column.
    """
    from sklearn.metrics import f1_score

    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    candidates = np.unique(scores)
    best_t, best_f1 = 0.5, -1.0
    for t in candidates:
        f1 = f1_score(labels, (scores >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def false_positive_rate(labels: "np.ndarray", predictions: "np.ndarray") -> float:
    """FPR = fraction of HUMAN texts flagged MACHINE (T5 fairness core)."""
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    human = labels == 0
    if not human.any():
        return float("nan")
    return float((predictions[human] == 1).mean())


def expected_calibration_error(labels: "np.ndarray", probs: "np.ndarray", n_bins: int = 15) -> float:
    """Expected Calibration Error (T6).

    Standard equal-width reliability-diagram ECE: bins ``probs`` (the fused
    machine-probability) into ``n_bins`` equal-width bins and averages
    ``|accuracy - confidence|`` per bin, weighted by bin occupancy. Here
    "accuracy" in a bin is the empirical machine rate (mean label), and
    "confidence" is the mean predicted machine-probability -- the natural
    reading when the model already outputs a single continuous P(machine)
    rather than a per-class softmax.
    """
    labels = np.asarray(labels, dtype=float)
    probs = np.asarray(probs, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(probs, bin_edges[1:-1], right=True), 0, n_bins - 1)
    n = len(labels)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        if not mask.any():
            continue
        conf = probs[mask].mean()
        acc = labels[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def compute_metrics(
    labels: "np.ndarray",
    scores: "np.ndarray",
    bucket: str | None = None,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Bundle AUROC/F1/FPR (+bucket tag) into one result row for CSV output.

    ``threshold`` defaults to 0.5 (used for methods whose score is already a
    probability); pass a train-fitted threshold (:func:`best_f1_threshold`)
    for raw statistics (curvature, perplexity, Binoculars) whose native range
    is not [0, 1].
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    t = 0.5 if threshold is None else float(threshold)
    preds = (scores >= t).astype(int)
    row: dict[str, Any] = {
        "auroc": auroc(labels, scores),
        "f1": f1_at_threshold(labels, scores, t),
        "fpr": false_positive_rate(labels, preds),
        "threshold": t,
        "n": int(len(labels)),
        "n_human": int((labels == 0).sum()),
        "n_machine": int((labels == 1).sum()),
    }
    if bucket is not None:
        row["bucket"] = bucket
    return row
