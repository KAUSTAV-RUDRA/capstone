"""Tests for src.fusion.fuser. Serves Phase 2 §2.2.6."""
from __future__ import annotations

import numpy as np
import pytest

from src.fusion.fuser import Fuser


def _toy_data(n_per_bucket: int = 100, seed: int = 0):
    rng = np.random.default_rng(seed)
    buckets_vocab = ["en", "hi"]
    head_scores, labels, buckets = [], [], []
    for b in buckets_vocab:
        y = rng.integers(0, 2, n_per_bucket)
        headA = np.where(y == 1, rng.uniform(0.6, 1.0, n_per_bucket), rng.uniform(0.0, 0.4, n_per_bucket))
        headB = rng.normal(0, 1, n_per_bucket)  # uninformative
        head_scores.append(np.stack([headA, headB], axis=1))
        labels.append(y)
        buckets.extend([b] * n_per_bucket)
    return np.vstack(head_scores), np.concatenate(labels), buckets, buckets_vocab


def test_fit_predict_proba_shapes() -> None:
    """predict_proba returns (n,) in [0,1]; coefficients per head."""
    head_scores, labels, buckets, vocab = _toy_data()
    fuser = Fuser(method="logistic", buckets=vocab).fit(head_scores, labels, buckets,
                                                        head_names=["headA", "headB"])

    probs = fuser.predict_proba(head_scores, buckets)
    assert probs.shape == (len(labels),)
    assert ((probs >= 0) & (probs <= 1)).all()

    coefs = fuser.coefficients()
    assert set(coefs) == {"headA", "headB", "bucket_en", "bucket_hi"}
    # headA is informative and should dominate a near-uninformative headB.
    assert abs(coefs["headA"]) > abs(coefs["headB"])


def test_predict_thresholds_predict_proba() -> None:
    head_scores, labels, buckets, vocab = _toy_data()
    fuser = Fuser(buckets=vocab).fit(head_scores, labels, buckets)
    preds = fuser.predict(head_scores, buckets)
    probs = fuser.predict_proba(head_scores, buckets)
    assert (preds == (probs >= 0.5).astype(int)).all()


def test_decision_function_monotonic_with_predict_proba() -> None:
    head_scores, labels, buckets, vocab = _toy_data()
    fuser = Fuser(buckets=vocab).fit(head_scores, labels, buckets)
    logits = fuser.decision_function(head_scores, buckets)
    probs = fuser.predict_proba(head_scores, buckets)
    assert np.allclose(1 / (1 + np.exp(-logits)), probs)


def test_gbm_has_no_coefficients() -> None:
    head_scores, labels, buckets, vocab = _toy_data()
    fuser = Fuser(method="gbm", buckets=vocab).fit(head_scores, labels, buckets)
    with pytest.raises(ValueError):
        fuser.coefficients()
    with pytest.raises(ValueError):
        fuser.decision_function(head_scores, buckets)


def test_save_and_load_roundtrip(tmp_path) -> None:
    head_scores, labels, buckets, vocab = _toy_data()
    fuser = Fuser(buckets=vocab).fit(head_scores, labels, buckets, head_names=["headA", "headB"])
    path = tmp_path / "fuser.joblib"
    fuser.save(path)
    loaded = Fuser.load(path)
    assert np.allclose(loaded.predict_proba(head_scores, buckets), fuser.predict_proba(head_scores, buckets))
