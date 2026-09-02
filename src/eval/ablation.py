"""Ablation sweep: A / B / A+B / +calibration / +abstention (T2).

Isolates the contribution of each component, demonstrating that calibration and
abstention (non-negotiable #3) add measurable value.

Serves docs/master-execution-plan.md Phase 3 §3.7.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

# Ablation configurations, in reporting order (T2 rows).
ABLATION_CONFIGS: tuple[str, ...] = ("A", "B", "A+B", "A+B+cal", "A+B+cal+abstain")


def run_ablation(config: dict | None = None) -> "pd.DataFrame":
    """Run every ablation configuration and return the T2 table per bucket."""
    # TODO(phase-3 step-3.7): evaluate each ABLATION_CONFIGS variant per bucket.
    raise NotImplementedError
