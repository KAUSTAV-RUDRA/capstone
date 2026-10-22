"""Tests for src.baselines.fast_detectgpt: the score contract, on CPU with the Qwen2.5-0.5B fallback scorer."""
from __future__ import annotations

import numpy as np
import pytest

from src.baselines.fast_detectgpt import FastDetectGPTDetector

MODEL = "Qwen/Qwen2.5-0.5B"


@pytest.fixture(scope="module")
def detector():
    det = FastDetectGPTDetector(MODEL, device="cpu", config={"max_length": 64})
    try:
        det.load()
    except Exception as exc:  # noqa: BLE001 - model not in the local HF cache / offline
        pytest.skip(f"{MODEL} unavailable locally: {exc}")
    return det


def test_score_contract(detector) -> None:
    texts = ["The cat sat on the mat and looked out of the window at the rain.",
             "Machine learning models learn patterns from large amounts of data."]
    scores = detector.score(texts)
    assert isinstance(scores, np.ndarray) and scores.shape == (2,)
    assert np.isfinite(scores).all()
    np.testing.assert_allclose(detector.score(texts[:1]), scores[:1], rtol=1e-4)   # deterministic, batch-independent
