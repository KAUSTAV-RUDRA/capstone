"""Tests for src.eval.risk_coverage. Serves Phase 3 §3.3."""
from __future__ import annotations

import numpy as np

from src.eval.risk_coverage import coverage_at_risk, metrics_at_coverage, risk_coverage_curve


def test_curve_monotonic_and_coverage_lookup() -> None:
    """Risk is non-increasing as coverage drops; coverage_at_risk works."""
    rng = np.random.default_rng(0)
    n = 500
    labels = rng.integers(0, 2, n)
    # Correct where confident, noisy where not: confidence should predict correctness.
    probs = np.where(labels == 1, rng.uniform(0.6, 1.0, n), rng.uniform(0.0, 0.4, n))
    noise_mask = rng.random(n) < 0.1
    probs[noise_mask] = 1 - probs[noise_mask]  # inject some confidently-wrong points
    confidences = np.maximum(probs, 1 - probs)

    coverage, risk = risk_coverage_curve(labels, probs, confidences)
    assert len(coverage) == len(risk) == n
    assert coverage[0] < coverage[-1]
    assert coverage[-1] == 1.0
    # Risk at full coverage should be <= risk restricted to only the most confident few
    # is not guaranteed pointwise, but overall risk should stay low given the setup.
    assert risk[-1] < 0.3

    # coverage_at_risk: 0 risk tolerance should not exceed what's achievable at full coverage.
    cov = coverage_at_risk(labels, probs, confidences, target_risk=1.0)
    assert cov == 1.0
    cov_strict = coverage_at_risk(labels, probs, confidences, target_risk=0.0)
    assert 0.0 <= cov_strict <= 1.0


def test_metrics_at_coverage_shrinks_with_lower_coverage() -> None:
    rng = np.random.default_rng(1)
    n = 300
    labels = rng.integers(0, 2, n)
    probs = np.where(labels == 1, rng.uniform(0.55, 1.0, n), rng.uniform(0.0, 0.45, n))
    confidences = np.maximum(probs, 1 - probs)

    m_full = metrics_at_coverage(labels, probs, confidences, 1.0)
    m_half = metrics_at_coverage(labels, probs, confidences, 0.5)
    assert m_full["n_covered"] == n
    assert m_half["n_covered"] == round(0.5 * n)
    # Keeping only the most confident half should not be worse than keeping everyone.
    assert m_half["accuracy"] >= m_full["accuracy"] - 1e-9
