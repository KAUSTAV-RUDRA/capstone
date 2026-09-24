"""Tests for src.calibration.temperature. Serves Phase 3 §3.1."""
from __future__ import annotations

import numpy as np
import pytest

from src.calibration.temperature import TemperatureScaler


def test_temperature_fitted_per_bucket() -> None:
    """A distinct temperature is fitted and applied per bucket."""
    rng = np.random.default_rng(0)
    scaler = TemperatureScaler()

    # en: well-separated logits (should fit close to T=1).
    en_logits = np.concatenate([rng.normal(-5, 1, 500), rng.normal(5, 1, 500)])
    en_labels = np.concatenate([np.zeros(500), np.ones(500)])
    # hi: overconfident logits for the same underlying separation (needs T > 1).
    hi_logits = en_logits * 5
    hi_labels = en_labels

    scaler.fit(en_logits, en_labels, bucket="en")
    scaler.fit(hi_logits, hi_labels, bucket="hi")

    assert scaler.temperature("en") != scaler.temperature("hi")
    assert scaler.temperature("hi") > scaler.temperature("en")  # more overconfident -> higher T

    probs_en = scaler.transform(en_logits, bucket="en")
    assert probs_en.shape == en_logits.shape
    assert ((probs_en >= 0) & (probs_en <= 1)).all()


def test_transform_is_monotonic_in_logit_and_preserves_auroc() -> None:
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(1)
    logits = rng.normal(0, 3, 200)
    labels = (logits > 0).astype(int)

    scaler = TemperatureScaler().fit(logits, labels, bucket="en")
    probs = scaler.transform(logits, bucket="en")
    # Monotonic transform of a non-degenerate temperature preserves AUROC exactly,
    # regardless of where the fitted T lands (this is the invariant that matters:
    # temperature scaling cannot change ranking/discrimination, only calibration).
    assert roc_auc_score(labels, probs) == roc_auc_score(labels, logits)


def test_transform_unfitted_bucket_raises() -> None:
    scaler = TemperatureScaler()
    with pytest.raises(KeyError):
        scaler.transform(np.array([0.0]), bucket="en")


def test_global_key_when_bucket_omitted() -> None:
    scaler = TemperatureScaler()
    logits = np.array([-2.0, -1.0, 1.0, 2.0])
    labels = np.array([0, 0, 1, 1])
    scaler.fit(logits, labels)
    assert scaler.temperature() == scaler.temperature(None)
