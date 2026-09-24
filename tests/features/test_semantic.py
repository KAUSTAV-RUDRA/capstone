"""Tests for src.features.semantic - Head C. Serves Phase 2 §2.2.5 / review2-sprint Day 3.

MuRIL is encoder-only; never used for curvature (non-negotiable #9).
No download: MurilEmbedder tests build a tiny random BERT from a config.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.features.semantic import HeadC, MurilEmbedder


@pytest.fixture(scope="module")
def tiny_embedder() -> MurilEmbedder:
    torch = pytest.importorskip("torch")
    from transformers import BertConfig, BertModel

    torch.manual_seed(0)
    emb = MurilEmbedder(config={"vocab_chunk_positions": 3})
    emb.model = BertModel(BertConfig(vocab_size=99, hidden_size=16, num_hidden_layers=2,
                                     num_attention_heads=2, intermediate_size=32,
                                     max_position_embeddings=32)).eval()
    return emb


def _mean_pool(model, ids, mask) -> np.ndarray:
    import torch

    with torch.no_grad():
        hidden = model(input_ids=ids, attention_mask=mask).last_hidden_state
    m = mask.unsqueeze(-1).float()
    return ((hidden * m).sum(1) / m.sum(1).clamp_min(1e-6)).double().numpy()


def test_pool_masks_out_padding(tiny_embedder) -> None:
    import torch

    ids = torch.randint(0, 99, (1, 9))
    mask = torch.ones(1, 9, dtype=torch.long)
    unpadded = _mean_pool(tiny_embedder.model, ids, mask)

    padded_ids = torch.cat([ids, torch.zeros(1, 4, dtype=torch.long)], dim=1)
    padded_mask = torch.cat([mask, torch.zeros(1, 4, dtype=torch.long)], dim=1)
    padded = _mean_pool(tiny_embedder.model, padded_ids, padded_mask)

    np.testing.assert_allclose(unpadded, padded, atol=1e-5)


def test_pool_writes_into_correct_result_rows(tiny_embedder) -> None:
    import torch

    result = np.full((3, 16), np.nan)
    texts = ["a", "bb", "ccc"]
    ids = [torch.randint(0, 99, (n,)) for n in (3, 5, 4)]

    class FakeTok:
        def __call__(self, batch, **kw):
            rows = [ids[texts.index(t)] for t in batch]
            n = max(len(r) for r in rows)
            padded = torch.zeros(len(rows), n, dtype=torch.long)
            mask = torch.zeros(len(rows), n, dtype=torch.long)
            for k, r in enumerate(rows):
                padded[k, :len(r)], mask[k, :len(r)] = r, 1
            return {"input_ids": padded, "attention_mask": mask}

    tiny_embedder.tokenizer = FakeTok()
    tiny_embedder._pool(texts, [2, 0], result)
    assert not np.isnan(result[2]).any() and not np.isnan(result[0]).any()
    assert np.isnan(result[1]).all()          # untouched row stays nan
    expected0 = _mean_pool(tiny_embedder.model, ids[0][None], torch.ones(1, len(ids[0]), dtype=torch.long))[0]
    np.testing.assert_allclose(result[0], expected0, atol=1e-5)


def test_head_c_per_bucket_fit_predict_and_roundtrip(tmp_path) -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 16))
    y = np.array([0, 1] * 40)
    X[y == 1, 0] += 2.0
    buckets = ["en"] * 40 + ["cm"] * 40
    head = HeadC(hidden_size=16).fit(X, y, buckets)
    p = head.predict(X, buckets)
    assert p.shape == (80,) and ((p >= 0) & (p <= 1)).all()
    assert p[y == 1].mean() > p[y == 0].mean()
    head.save(tmp_path / "c.joblib")
    reloaded = HeadC.load(tmp_path / "c.joblib")
    np.testing.assert_allclose(reloaded.predict(X, buckets), p)


def test_head_c_unknown_bucket_raises() -> None:
    X = np.zeros((4, 3))
    y = np.array([0, 1, 0, 1])
    head = HeadC(hidden_size=3).fit(X, y, ["en"] * 4)
    with pytest.raises(KeyError):
        head.predict(X, ["te"] * 4)
