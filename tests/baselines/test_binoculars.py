"""Tests for src.baselines.binoculars. Serves Phase 2 §2.2.1 / review2-sprint Day 3.

No download: builds two tiny random GPT-2 models (same vocab) and feeds token
ids directly, mirroring tests/features/test_curvature.py's tiny_scorer pattern.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.baselines.binoculars import BinocularsDetector


@pytest.fixture(scope="module")
def tiny_detector() -> BinocularsDetector:
    torch = pytest.importorskip("torch")
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    det = BinocularsDetector(config={"vocab_chunk_positions": 3})
    cfg = GPT2Config(vocab_size=50, n_positions=32, n_embd=16, n_layer=2, n_head=2)
    det.observer = GPT2LMHeadModel(cfg).eval()
    torch.manual_seed(1)                             # a genuinely different performer
    det.performer = GPT2LMHeadModel(cfg).eval()
    return det


def _reference(det: BinocularsDetector, ids) -> float:
    """mean_i(l_i - m_i) from full (unchunked) logits, no batching."""
    import torch

    with torch.no_grad():
        logp_obs = torch.log_softmax(det.observer(ids[None]).logits[0, :-1].float(), -1)
        p_perf = torch.softmax(det.performer(ids[None]).logits[0, :-1].float(), -1)
    l = logp_obs.gather(-1, ids[1:, None]).squeeze(-1)
    m = (p_perf * logp_obs).sum(-1)
    return float((l - m).mean())


def test_cross_stats_matches_full_logits_reference(tiny_detector) -> None:
    import torch

    ids = torch.randint(0, 50, (9,))
    got = tiny_detector._cross_stats(ids[None], torch.ones(1, 9, dtype=torch.long))[0]
    assert got == pytest.approx(_reference(tiny_detector, ids), abs=1e-5)


def test_padded_batch_matches_single_rows(tiny_detector) -> None:
    import torch

    rows = [torch.randint(0, 50, (n,)) for n in (9, 4, 7)]
    ids = torch.zeros(3, 9, dtype=torch.long)
    mask = torch.zeros(3, 9, dtype=torch.long)
    for k, r in enumerate(rows):
        ids[k, :len(r)], mask[k, :len(r)] = r, 1
    batched = tiny_detector._cross_stats(ids, mask)
    for r, got in zip(rows, batched):
        single = tiny_detector._cross_stats(r[None], torch.ones(1, len(r), dtype=torch.long))[0]
        assert got == pytest.approx(single, abs=1e-5)
        assert got == pytest.approx(_reference(tiny_detector, r), abs=1e-5)


def test_score_batches_via_token_budget_groups(tiny_detector, monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    from transformers import GPT2TokenizerFast

    # A tiny whitespace-ish tokenizer would need real BPE files; reuse the model's
    # own vocab range by driving score() through a stand-in tokenizer that maps
    # characters to ids within [0, 50).
    class FakeTok:
        pad_token = "<pad>"

        def __call__(self, texts, **kw):
            if isinstance(texts, str):
                texts = [texts]
            ids = [[ord(c) % 50 for c in t] for t in texts]
            if kw.get("padding"):
                n = max(len(row) for row in ids)
                mask = [[1] * len(row) + [0] * (n - len(row)) for row in ids]
                ids = [row + [0] * (n - len(row)) for row in ids]
                return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}
            return {"input_ids": ids}

    tiny_detector.tokenizer = FakeTok()
    tiny_detector.max_length = 32
    monkeypatch.setattr(tiny_detector, "load", lambda: None)
    scores = tiny_detector.score(["hello there", "hi", "a longer sentence to score"])
    assert scores.shape == (3,)
    assert np.isfinite(scores).all()
