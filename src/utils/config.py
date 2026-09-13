"""YAML configuration loading and merging.

Every script takes ``--config`` pointing at ``configs/*.yaml`` (project
convention). This module is the single place that reads those files so no
component hardcodes paths.

Serves: cross-cutting, all phases of docs/master-execution-plan.md.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a plain dict.

    Args:
        path: Path to a ``configs/*.yaml`` file.

    Returns:
        Parsed configuration as a nested dict (empty dict for an empty file).
    """
    import yaml  # lazy: keeps ``src`` importable with nothing installed

    with open(path, encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path}: top level must be a mapping, got {type(loaded).__name__}")
    return loaded


def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge ``override`` onto ``base`` and return a new dict.

    Nested dicts are merged recursively; any other value in ``override``
    replaces the base value. Neither input is mutated.

    Args:
        base: Base configuration (e.g. ``configs/default.yaml``).
        override: Values that take precedence (e.g. ``configs/models.yaml``).

    Returns:
        The merged configuration.
    """
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def get(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Read a nested value with a dotted key, e.g. ``"calibration.method"``.

    Args:
        config: The configuration dict.
        dotted_key: Dot-separated path into the config.
        default: Returned when the key is absent.

    Returns:
        The resolved value or ``default``.
    """
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current
