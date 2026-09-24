"""Tests for src.calibration.abstention. Serves Phase 3 §3.3."""
from __future__ import annotations

import numpy as np

from src.calibration.abstention import AbstentionGate, coverage_sweep
from src.calibration.conformal import ConformalCalibrator


def _gate() -> AbstentionGate:
    calibrator = ConformalCalibrator(alpha=0.01)
    calibrator.thresholds["en"] = 0.9
    gate = AbstentionGate(calibrator, lower_threshold={"en": 0.1})
    return gate


def test_decide_returns_three_way_verdict() -> None:
    """decide(prob, bucket) -> ('HUMAN'|'ABSTAIN'|'MACHINE', confidence)."""
    gate = _gate()

    verdict, conf = gate.decide(0.95, "en")
    assert verdict == "MACHINE" and conf == 0.95

    verdict, conf = gate.decide(0.05, "en")
    assert verdict == "HUMAN" and conf == 0.95  # max(0.05, 0.95)

    verdict, conf = gate.decide(0.5, "en")
    assert verdict == "ABSTAIN" and conf == 0.5

    # Boundary: exactly at tau is not > tau, so not MACHINE.
    verdict, _ = gate.decide(0.9, "en")
    assert verdict == "ABSTAIN"


def test_decide_batch_matches_decide() -> None:
    gate = _gate()
    probs = [0.95, 0.05, 0.5]
    out = gate.decide_batch(probs, ["en", "en", "en"])
    assert out == [gate.decide(p, "en") for p in probs]


def test_coverage_sweep_monotonic_and_bounded() -> None:
    rng = np.random.default_rng(0)
    probs = np.concatenate([rng.uniform(0, 0.4, 200), rng.uniform(0.6, 1.0, 200)])
    labels = np.concatenate([np.zeros(200), np.ones(200)])
    tau_machine = 0.8

    rows = coverage_sweep(probs, labels, tau_machine, coverage_targets=(0.3, 0.5, 0.7, 1.0))
    coverages = [r["actual_coverage"] for r in rows]
    # Coverage should be non-decreasing as the target grows.
    assert all(c1 <= c2 + 1e-9 for c1, c2 in zip(coverages, coverages[1:]))
    # At target 1.0, everything not already MACHINE ends up HUMAN -> full coverage.
    assert rows[-1]["actual_coverage"] == 1.0
    for r in rows:
        assert 0.0 <= r["actual_coverage"] <= 1.0
        assert r["n_covered"] <= len(probs)
