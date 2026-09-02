"""YAML configuration loading and merging.

Every script takes ``--config`` pointing at ``configs/*.yaml`` (project
convention). This module is the single place that reads those files so no
component hardcodes paths.

Serves: cross-cutting, all phases of docs/master-execution-plan.md.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a plain dict.

    Args:
        path: Path to a ``configs/*.yaml`` file.

    Returns:
        Parsed configuration as a nested dict.
    """
    # TODO(phase-1 step-1.2): implement YAML load via pyyaml.
    raise NotImplementedError


def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge ``override`` onto ``base`` and return a new dict.

    Args:
        base: Base configuration (e.g. ``configs/default.yaml``).
        override: Values that take precedence (e.g. ``configs/models.yaml``).

    Returns:
        The merged configuration.
    """
    # TODO(phase-1 step-1.2): implement recursive merge.
    raise NotImplementedError


def get(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Read a nested value with a dotted key, e.g. ``"calibration.method"``.

    Args:
        config: The configuration dict.
        dotted_key: Dot-separated path into the config.
        default: Returned when the key is absent.

    Returns:
        The resolved value or ``default``.
    """
    # TODO(phase-1 step-1.2): implement dotted lookup.
    raise NotImplementedError
