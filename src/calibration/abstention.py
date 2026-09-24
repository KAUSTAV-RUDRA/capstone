"""Abstention gate - three-way HUMAN / ABSTAIN / MACHINE decision.

NON-NEGOTIABLE #3: the three-way output with per-language thresholds is the
contribution. NON-NEGOTIABLE #8: output is decision support, never an automatic
accusation. Uses the per-bucket conformal thresholds from
:class:`src.calibration.conformal.ConformalCalibrator`.

Rule (review2-sprint Day 4 spec):
    MACHINE   if prob > tau_0.01 (the alpha=0.01 per-bucket conformal threshold)
    HUMAN     if prob < median(bucket's human calibration scores)
    ABSTAIN   otherwise

The lower (HUMAN) cutoff starts at the calibration median and can be swept to
trace coverage from 30% to 100% for the risk-coverage curve (F1) and T6's
accuracy-at-coverage rows -- see :func:`coverage_sweep`.

Contract: ``decide(prob, bucket) -> (Literal["HUMAN","ABSTAIN","MACHINE"], float)``.

Serves docs/master-execution-plan.md Phase 3 §3.3.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Sequence

import numpy as np

if TYPE_CHECKING:
    from src.calibration.conformal import ConformalCalibrator

Verdict = Literal["HUMAN", "ABSTAIN", "MACHINE"]


class AbstentionGate:
    """Maps a calibrated probability + bucket to a three-way verdict."""

    def __init__(self, calibrator: "ConformalCalibrator", config: dict | None = None,
                lower_threshold: dict[str, float] | None = None) -> None:
        """Args:
        calibrator: A fitted per-language conformal calibrator (alpha=0.01;
            the MACHINE cutoff).
        config: Loaded ``configs/default.yaml`` (``abstention`` block).
        lower_threshold: Per-bucket HUMAN cutoff, e.g. the calibration-split
            median machine-probability. Set via :meth:`set_lower_threshold`
            if not supplied here.
        """
        self.calibrator = calibrator
        self.config = config or {}
        self.lower_threshold: dict[str, float] = dict(lower_threshold) if lower_threshold else {}

    def set_lower_threshold(self, bucket: str, value: float) -> None:
        self.lower_threshold[bucket] = float(value)

    def decide(self, prob: float, bucket: str) -> tuple[Verdict, float]:
        """Return the verdict and confidence for one calibrated probability.

        Args:
            prob: Calibrated machine-probability for the sample.
            bucket: Language bucket used to select the conformal threshold.

        Returns:
            ``(verdict, confidence)`` where verdict is HUMAN / ABSTAIN / MACHINE
            and confidence is ``max(prob, 1 - prob)``.
        """
        tau = self.calibrator.threshold(bucket)
        if bucket not in self.lower_threshold:
            raise KeyError(f"no lower (HUMAN) threshold set for bucket {bucket!r}")
        lower = self.lower_threshold[bucket]
        confidence = max(prob, 1 - prob)
        if prob > tau:
            return "MACHINE", confidence
        if prob < lower:
            return "HUMAN", confidence
        return "ABSTAIN", confidence

    def decide_batch(
        self,
        probs: "np.ndarray",
        buckets: list[str],
    ) -> list[tuple[Verdict, float]]:
        """Vectorised :meth:`decide` over parallel ``probs`` and ``buckets``."""
        return [self.decide(float(p), b) for p, b in zip(probs, buckets)]


def coverage_sweep(
    probs: "np.ndarray",
    labels: "np.ndarray",
    tau_machine: float,
    coverage_targets: Sequence[float] = tuple(x / 100 for x in range(30, 101, 5)),
) -> list[dict[str, float]]:
    """Sweep the HUMAN lower cutoff to trace coverage from 30% to 100%.

    For one bucket. The rule stays ``MACHINE if prob > tau_machine``; only the
    lower (HUMAN) cutoff ``t`` moves, from 0 up to ``tau_machine``. Coverage
    (fraction not ABSTAIN) is monotonically non-decreasing in ``t``, so for
    each target we take the smallest ``t`` whose actual coverage is >= target
    (falling back to the maximum achievable coverage if the target is
    unreachable, e.g. because every score sits in the ABSTAIN band even at
    ``t = tau_machine``).

    Returns one row per target: ``target_coverage``, ``lower_threshold``,
    ``actual_coverage``, ``accuracy`` and ``fpr`` on the covered (non-abstained)
    subset, and ``n_covered``.
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels)
    n = len(probs)
    is_machine_verdict = probs > tau_machine
    candidates = np.sort(np.unique(np.clip(probs, 0, tau_machine)))
    rows: list[dict[str, float]] = []
    for target in coverage_targets:
        chosen_t = float(candidates[-1]) if len(candidates) else 0.0
        chosen_coverage = 0.0
        chosen_covered = np.zeros(n, dtype=bool)
        for t in candidates:
            covered = is_machine_verdict | (probs < t)
            coverage = covered.mean() if n else 0.0
            chosen_t, chosen_coverage, chosen_covered = float(t), float(coverage), covered
            if coverage >= target:
                break
        preds = is_machine_verdict.astype(int)
        if chosen_covered.any():
            acc = float((preds[chosen_covered] == labels[chosen_covered]).mean())
            human_covered = chosen_covered & (labels == 0)
            fpr = float((preds[human_covered] == 1).mean()) if human_covered.any() else float("nan")
        else:
            acc = float("nan")
            fpr = float("nan")
        rows.append({
            "target_coverage": float(target),
            "lower_threshold": chosen_t,
            "actual_coverage": chosen_coverage,
            "accuracy": acc,
            "fpr": fpr,
            "n_covered": int(chosen_covered.sum()),
        })
    return rows
