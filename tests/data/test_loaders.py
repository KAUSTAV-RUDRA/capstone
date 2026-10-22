"""Tests for src.data.loaders.

Skipped: src/data/loaders.py is an unimplemented stub; splits are read from the frozen corpus.jsonl by the callers (tests/data/test_freeze_splits.py covers splitting and the human-only calibration set).
"""
from __future__ import annotations

import pytest


@pytest.mark.skip(reason='src/data/loaders.py is an unimplemented stub; splits are read from the frozen corpus.jsonl by the callers (tests/data/test_freeze_splits.py covers splitting and the human-only calibration set).')
def test_load_split_and_calibration() -> None:
    raise NotImplementedError
