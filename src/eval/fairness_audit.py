"""Fairness audit: FPR by bucket and by L1/L2 band (T5).

Quantifies the native/non-native false-positive disparity for our system vs
baselines - the headline fairness result (RQ4, non-negotiable #3 rationale).

Serves docs/master-execution-plan.md Phase 3 §3.6.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


def fpr_by_bucket(
    labels: "np.ndarray",
    predictions: "np.ndarray",
    buckets: list[str],
) -> dict[str, float]:
    """FPR per language bucket (en / hi / te / cm)."""
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    buckets_arr = np.asarray(buckets)
    out: dict[str, float] = {}
    for bucket in sorted(set(buckets_arr.tolist())):
        mask = buckets_arr == bucket
        human = mask & (labels == 0)
        out[bucket] = float((predictions[human] == 1).mean()) if human.any() else float("nan")
    return out


def fpr_by_l1_band(
    labels: "np.ndarray",
    predictions: "np.ndarray",
    l1_bands: list[str],
) -> dict[str, float]:
    """FPR per writer L1/L2 band (native / non_native / unknown)."""
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    bands = np.asarray(l1_bands)
    out: dict[str, float] = {}
    for band in sorted({b for b in bands.tolist() if b is not None}):
        mask = (bands == band) & (labels == 0)
        out[band] = float((predictions[mask] == 1).mean()) if mask.any() else float("nan")
    return out


def disparity(fpr_by_group: dict[str, float]) -> float:
    """Max-minus-min FPR across groups (the disparity we aim to reduce)."""
    values = [v for v in fpr_by_group.values() if v == v]  # drop NaN
    if not values:
        return float("nan")
    return float(max(values) - min(values))


def audit(
    labels: "np.ndarray",
    predictions: "np.ndarray",
    buckets: list[str],
    l1_bands: list[str],
    system_name: str,
) -> "pd.DataFrame":
    """Assemble the full T5 fairness table for one system."""
    import pandas as pd

    rows: list[dict[str, object]] = []
    by_bucket = fpr_by_bucket(labels, predictions, buckets)
    for bucket, fpr in by_bucket.items():
        rows.append({"system": system_name, "group_type": "bucket", "group": bucket, "fpr": fpr})
    rows.append({"system": system_name, "group_type": "bucket", "group": "disparity",
                "fpr": disparity(by_bucket)})

    by_band = fpr_by_l1_band(labels, predictions, l1_bands)
    for band, fpr in by_band.items():
        rows.append({"system": system_name, "group_type": "l1_band", "group": band, "fpr": fpr})
    if by_band:
        rows.append({"system": system_name, "group_type": "l1_band", "group": "disparity",
                    "fpr": disparity(by_band)})
    return pd.DataFrame(rows)
