"""Abstention gate - three-way HUMAN / ABSTAIN / MACHINE decision.

NON-NEGOTIABLE #3: the three-way output with per-language thresholds is the
contribution. NON-NEGOTIABLE #8: output is decision support, never an automatic
accusation. Uses the per-bucket conformal thresholds from
:class:`src.calibration.conformal.ConformalCalibrator`.

Contract: ``decide(prob, bucket) -> (Literal["HUMAN","ABSTAIN","MACHINE"], float)``.

Serves docs/master-execution-plan.md Phase 3 §3.3.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import numpy as np

    from src.calibration.conformal import ConformalCalibrator

Verdict = Literal["HUMAN", "ABSTAIN", "MACHINE"]


class AbstentionGate:
    """Maps a calibrated probability + bucket to a three-way verdict."""

    def __init__(self, calibrator: "ConformalCalibrator", config: dict | None = None) -> None:
        """Args:
        calibrator: A fitted per-language conformal calibrator.
        config: Loaded ``configs/default.yaml`` (``abstention`` block).
        """
        # TODO(phase-3 step-3.3): store calibrator + abstention band settings.
        raise NotImplementedError

    def decide(self, prob: float, bucket: str) -> tuple[Verdict, float]:
        """Return the verdict and confidence for one calibrated probability.

        Args:
            prob: Calibrated machine-probability for the sample.
            bucket: Language bucket used to select the conformal threshold.

        Returns:
            ``(verdict, confidence)`` where verdict is HUMAN / ABSTAIN / MACHINE.
        """
        # TODO(phase-3 step-3.3): apply per-bucket threshold + abstention band.
        raise NotImplementedError

    def decide_batch(
        self,
        probs: "np.ndarray",
        buckets: list[str],
    ) -> list[tuple[Verdict, float]]:
        """Vectorised :meth:`decide` over parallel ``probs`` and ``buckets``."""
        # TODO(phase-3 step-3.3): batch decisions for a coverage sweep.
        raise NotImplementedError
