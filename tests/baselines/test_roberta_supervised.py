"""Tests for src.baselines.roberta_supervised.

Skipped: the supervised XLM-R baseline was never built (T1 reports three zero-shot baselines); src/baselines/roberta_supervised.py is an unimplemented stub.
"""
from __future__ import annotations

import pytest


@pytest.mark.skip(reason='the supervised XLM-R baseline was never built (T1 reports three zero-shot baselines); src/baselines/roberta_supervised.py is an unimplemented stub.')
def test_fit_then_score_contract() -> None:
    raise NotImplementedError
