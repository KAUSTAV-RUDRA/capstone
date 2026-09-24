"""Tests for src.eval.metrics. Serves Phase 2/3."""
from __future__ import annotations

import numpy as np

from src.eval.metrics import (
    auroc,
    best_f1_threshold,
    compute_metrics,
    expected_calibration_error,
    f1_at_threshold,
    false_positive_rate,
)


def test_auroc_f1_fpr_ece() -> None:
    """Metrics match sklearn / hand-computed values on a tiny known example."""
    from sklearn.metrics import f1_score, roc_auc_score

    labels = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.6, 0.4, 0.8, 0.9])

    assert auroc(labels, scores) == roc_auc_score(labels, scores)

    t = 0.5
    preds = (scores >= t).astype(int)
    assert f1_at_threshold(labels, scores, t) == f1_score(labels, preds)

    # FPR: 1 of 3 humans (score 0.6) crosses threshold 0.5.
    assert false_positive_rate(labels, preds) == 1 / 3

    # A perfectly calibrated predictor has ECE 0.
    perfect_labels = np.array([0, 0, 1, 1] * 25)
    perfect_probs = np.array([0.0, 0.0, 1.0, 1.0] * 25)
    assert expected_calibration_error(perfect_labels, perfect_probs) == 0.0

    # A confidently-wrong predictor has high ECE.
    wrong_probs = np.array([1.0, 1.0, 0.0, 0.0] * 25)
    assert expected_calibration_error(perfect_labels, wrong_probs) > 0.9


def test_best_f1_threshold_separates_perfectly_separable_scores() -> None:
    labels = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    t = best_f1_threshold(labels, scores)
    preds = (scores >= t).astype(int)
    assert f1_at_threshold(labels, scores, t) == 1.0
    assert (preds == labels).all()


def test_compute_metrics_bundles_fields_and_bucket_tag() -> None:
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.6, 0.4, 0.9])
    row = compute_metrics(labels, scores, bucket="en", threshold=0.5)
    assert row["bucket"] == "en"
    assert row["n"] == 4 and row["n_human"] == 2 and row["n_machine"] == 2
    assert 0.0 <= row["auroc"] <= 1.0
    assert 0.0 <= row["f1"] <= 1.0
    assert 0.0 <= row["fpr"] <= 1.0
