"""Freeze train/cal/test splits into data/processed/splits.json.

NON-NEGOTIABLE #6: ``data/processed/splits.json`` is frozen once written and
must never be modified or regenerated. This module writes it exactly once and
refuses to overwrite an existing file (the "overwrite guard").

Serves docs/master-execution-plan.md Phase 2 §2.1.1.
"""
from __future__ import annotations

from pathlib import Path

from src.data.schema import Sample


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
    raise NotImplementedError


def load_frozen_splits(path: str | Path) -> dict[str, list[str]]:
    """Load the frozen split assignment (mapping split -> list of sample ids)."""
    # TODO(phase-2 step-2.1.1): read splits.json.
    raise NotImplementedError


def assert_splits_frozen(path: str | Path) -> None:
    """Raise if ``splits.json`` is missing; used as a precondition by experiments."""
    # TODO(phase-2 step-2.1.1): verify the frozen file exists.
    raise NotImplementedError
