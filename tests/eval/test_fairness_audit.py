"""Tests for src.eval.fairness_audit. Serves Phase 3 §3.6."""
from __future__ import annotations

import numpy as np
import pytest

from src.eval.fairness_audit import audit, disparity, fpr_by_bucket, fpr_by_l1_band


def test_fpr_by_group_and_disparity() -> None:
    """Per-bucket/L1 FPR computed; disparity = max-min FPR."""
    labels = np.array([0, 0, 0, 0, 1, 1])  # 4 human, 2 machine
    predictions = np.array([1, 0, 0, 0, 1, 1])  # 1 human false-positive
    buckets = ["en", "en", "hi", "hi", "en", "hi"]

    by_bucket = fpr_by_bucket(labels, predictions, buckets)
    assert by_bucket["en"] == 0.5  # 1 of 2 en humans flagged
    assert by_bucket["hi"] == 0.0  # 0 of 2 hi humans flagged
    assert disparity(by_bucket) == 0.5

    l1_bands = ["native", "native", "non_native", "non_native", "native", "non_native"]
    by_band = fpr_by_l1_band(labels, predictions, l1_bands)
    assert by_band["native"] == 0.5
    assert by_band["non_native"] == 0.0


def test_disparity_ignores_nan_groups() -> None:
    assert disparity({"a": 0.1, "b": float("nan"), "c": 0.4}) == pytest.approx(0.3)


def test_audit_assembles_dataframe_with_disparity_rows() -> None:
    labels = np.array([0, 0, 1, 1])
    predictions = np.array([1, 0, 1, 1])
    buckets = ["en", "hi", "en", "hi"]
    l1_bands = ["native", "non_native", "native", "non_native"]

    df = audit(labels, predictions, buckets, l1_bands, system_name="fused")
    assert set(df["system"]) == {"fused"}
    assert "disparity" in set(df[df["group_type"] == "bucket"]["group"])
    assert "disparity" in set(df[df["group_type"] == "l1_band"]["group"])
