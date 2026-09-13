"""Freeze train/cal/test splits into data/processed/splits.json.

NON-NEGOTIABLE #6: ``data/processed/splits.json`` is frozen once written and
must never be modified or regenerated. This module writes it exactly once and
refuses to overwrite an existing file (the "overwrite guard").

Serves docs/master-execution-plan.md Phase 2 §2.1.1.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.data.schema import Sample


def calibration_eligible_sources(config: dict[str, Any]) -> dict[str, bool]:
    """Map ``source`` name -> may its rows enter the ``cal`` split?

    Reads ``human_corpus`` from ``configs/data.yaml``: every source is eligible
    unless its spec sets ``calibration_eligible: false`` (the default comes from
    ``calibration_eligible_default``).

    The conformal guarantee is only as strong as the certainty that the
    calibration text is human, so a source whose collection date cannot rule out
    machine text is barred from ``cal`` and routed to train/test instead. As of
    2026-09-13 that is ``comi_lingua`` (53 % of the ``cm`` bucket), which leaves
    cm calibration to cmu_hinglish_dog + hinge = 1,025 rows against a 1,000-row
    floor. See docs/decisions.md.
    """
    human_corpus = config.get("human_corpus") or {}
    default = bool(human_corpus.get("calibration_eligible_default", True))
    eligible: dict[str, bool] = {}
    for bucket_cfg in (human_corpus.get("buckets") or {}).values():
        for band_cfg in (bucket_cfg.get("bands") or {}).values():
            for spec in band_cfg.get("sources") or []:
                eligible[spec["name"]] = bool(spec.get("calibration_eligible", default))
    return eligible


def freeze_splits(
    samples: list[Sample],
    output_path: str | Path,
    config: dict | None = None,
    force: bool = False,
) -> None:
    """Assign and persist immutable train/cal/test splits.

    Held-out generators and one held-out domain (named in ``configs/data.yaml``)
    are routed to ``test`` only (non-negotiable #4).

    Args:
        samples: All corpus samples to partition.
        output_path: Destination ``splits.json`` path.
        config: Loaded ``configs/data.yaml`` with split ratios and held-outs.
        force: Must stay False in normal use; the guard raises if the file
            already exists.

    Raises:
        FileExistsError: If ``output_path`` already exists (the freeze guard).
    """
    # TODO(phase-2 step-2.1.1): partition + write splits.json with overwrite guard.
    # MUST, when assigning the `cal` split (see docs/decisions.md 2026-09-13):
    #   1. call calibration_eligible_sources(config) and route every row whose
    #      `source` maps to False into train/test, never `cal`;
    #   2. assert each bucket still meets config["min_human_calibration_per_bucket"]
    #      AFTER that routing, and fail loudly if not — a silently undersized
    #      calibration set voids the conformal guarantee without any visible error.
    raise NotImplementedError


def load_frozen_splits(path: str | Path) -> dict[str, list[str]]:
    """Load the frozen split assignment (mapping split -> list of sample ids)."""
    # TODO(phase-2 step-2.1.1): read splits.json.
    raise NotImplementedError


def assert_splits_frozen(path: str | Path) -> None:
    """Raise if ``splits.json`` is missing; used as a precondition by experiments."""
    # TODO(phase-2 step-2.1.1): verify the frozen file exists.
    raise NotImplementedError
