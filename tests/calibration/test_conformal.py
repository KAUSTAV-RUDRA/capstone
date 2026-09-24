"""Tests for src.calibration.conformal. Serves Phase 3 §3.2.

Per-language thresholds; empirical FPR <= alpha per bucket (non-negotiable #3).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.calibration.conformal import ConformalCalibrator, conformal_quantile


def test_threshold_per_bucket_and_fpr_bound() -> None:
    """Distinct threshold per bucket; empirical FPR <= alpha on held-out human scores."""
    rng = np.random.default_rng(0)
    calibrator = ConformalCalibrator(alpha=0.05)
    # Two buckets with very different human score distributions.
    en_cal = rng.uniform(0.0, 0.3, size=1000)
    hi_cal = rng.uniform(0.4, 0.9, size=1000)
    calibrator.fit(en_cal, "en")
    calibrator.fit(hi_cal, "hi")

    assert calibrator.is_calibrated("en") and calibrator.is_calibrated("hi")
    assert not calibrator.is_calibrated("te")
    assert calibrator.buckets() == ["en", "hi"]
    # Thresholds reflect each bucket's own distribution, not a shared one.
    assert calibrator.threshold("en") < calibrator.threshold("hi")

    # Empirical FPR bound: fresh human draws from the SAME distribution should
    # exceed tau at a rate close to (and, over repeated draws, no more than
    # roughly) alpha.
    for bucket, dist in (("en", lambda n: rng.uniform(0.0, 0.3, size=n)),
                        ("hi", lambda n: rng.uniform(0.4, 0.9, size=n))):
        fresh = dist(5000)
        fpr = float((fresh > calibrator.threshold(bucket)).mean())
        assert fpr < 0.05 + 0.02  # small slack for finite-sample noise


def test_global_threshold_is_separate_from_per_bucket() -> None:
    calibrator = ConformalCalibrator(alpha=0.05)
    calibrator.fit(np.linspace(0, 0.3, 1000), "en")
    calibrator.fit(np.linspace(0.4, 0.9, 1000), "hi")
    calibrator.fit_global(np.concatenate([np.linspace(0, 0.3, 1000), np.linspace(0.4, 0.9, 1000)]))

    assert calibrator.threshold("_global") != calibrator.threshold("en")
    assert calibrator.threshold("_global") != calibrator.threshold("hi")
    assert "_global" not in calibrator.buckets()  # buckets() excludes the global key


def test_conformal_quantile_too_few_samples_returns_inf() -> None:
    # n=10, alpha=0.01 needs the ceil(11*0.99)=11th smallest of 10 -> unreachable.
    assert conformal_quantile(np.arange(10, dtype=float), alpha=0.01) == float("inf")


def test_threshold_unfitted_bucket_raises() -> None:
    calibrator = ConformalCalibrator(alpha=0.05)
    with pytest.raises(KeyError):
        calibrator.threshold("en")
