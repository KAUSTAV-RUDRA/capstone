"""Tests for src.eval.ablation. Serves Phase 3 §3.7.

Integration test: exercises the real corpus/scores.parquet + fitted pipeline
(fast -- everything here is logistic regression / quantiles on already-scored
data, no GPU, no model loading), since ablation has no meaningful synthetic
scaffold independent of the fusion+calibration pipeline it sweeps.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    not Path("results/scores.parquet").exists() or not Path("data/processed/corpus.jsonl").exists(),
    reason="needs results/scores.parquet + data/processed/corpus.jsonl (review2-sprint Day 2-3 outputs)",
)


def test_all_configs_present() -> None:
    """The ablation table has a row per ABLATION_CONFIGS entry per bucket."""
    from src.data.schema import LANGUAGE_BUCKETS
    from src.eval.ablation import ABLATION_CONFIGS, run_ablation

    df = run_ablation()
    assert set(df["bucket"]) == set(LANGUAGE_BUCKETS)
    for bucket in LANGUAGE_BUCKETS:
        configs_present = set(df[df["bucket"] == bucket]["config"].astype(str))
        assert configs_present == set(ABLATION_CONFIGS), f"{bucket} missing configs"


def test_temperature_row_matches_ab_row_auroc() -> None:
    """AUROC for +temperature must equal A+B exactly (monotonic transform)."""
    from src.eval.ablation import run_ablation

    df = run_ablation()
    for bucket in df["bucket"].unique():
        ab = df[(df["bucket"] == bucket) & (df["config"] == "A+B")]["auroc"].iloc[0]
        temp = df[(df["bucket"] == bucket) & (df["config"] == "+temperature")]["auroc"].iloc[0]
        assert ab == pytest.approx(temp)


def test_abstain_row_has_no_auroc_but_has_coverage() -> None:
    from src.eval.ablation import run_ablation

    df = run_ablation()
    abstain = df[df["config"] == "+abstain"]
    assert abstain["auroc"].isna().all()
    assert abstain["f1"].isna().all()
    assert (abstain["coverage"] > 0).all() and (abstain["coverage"] <= 1).all()
