"""Tests for src.features.curvature - Head B. Serves Phase 2 §2.2.3 / review2-sprint Day 2.

No download: the model-level tests build a tiny random GPT-2 from a config and
feed it token ids directly.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.features.curvature import CurvatureScorer, curvature_stats, token_word_ids


def test_token_word_ids_folds_leading_space_and_whitespace_tokens() -> None:
    text = "yaar kal\n\nok"
    offsets = [(0, 2), (2, 4), (4, 8), (8, 9), (9, 10), (10, 12)]   # ya|ar| kal|\n|\n|ok
    assert token_word_ids(text, offsets) == [0, 0, 1, 1, 1, 2]


def test_token_word_ids_leading_whitespace_joins_first_word() -> None:
    assert token_word_ids("\nhi there", [(0, 1), (1, 3), (3, 9)]) == [0, 0, 1]


def test_word_sum_then_pool_equals_token_level() -> None:
    # The identity that makes per-word standardisation necessary (module docstring).
    rng = np.random.default_rng(0)
    lp, mean, var = rng.normal(-3, 1, 20), rng.normal(-3, 1, 20), rng.uniform(0.5, 2, 20)
    words = np.repeat(np.arange(5), 4)
    d_tok, _ = curvature_stats(lp, mean, var, words)
    dev_w = np.bincount(words, weights=lp - mean)
    assert d_tok == pytest.approx(dev_w.sum() / np.sqrt(np.bincount(words, weights=var).sum()))


def test_one_token_words_give_mean_token_z() -> None:
    lp, mean, var = np.array([-1.0, -2.0, -4.0]), np.array([-2.0, -2.0, -2.0]), np.array([1.0, 4.0, 4.0])
    d_tok, d_word = curvature_stats(lp, mean, var, np.arange(3))
    assert d_tok == pytest.approx((1 + 0 - 2) / 3)
    assert d_word == pytest.approx((1 + 0 - 1) / np.sqrt(3))


def test_fragmentation_does_not_change_word_vote() -> None:
    # Splitting a word's deviation and variance across k tokens leaves d_word unchanged.
    one = curvature_stats(np.array([-1.0]), np.array([-2.0]), np.array([1.0]), np.array([0]))[1]
    six = curvature_stats(np.full(6, -1 / 6), np.full(6, -2 / 6), np.full(6, 1 / 6), np.zeros(6, int))[1]
    assert one == pytest.approx(six)


def test_empty_is_nan() -> None:
    assert all(np.isnan(curvature_stats(np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0, int))))


def test_token_budget_groups_split_long_rows_and_pack_short_ones() -> None:
    scorer = CurvatureScorer(config={"max_batch_tokens": 100, "batch_size": 8})
    # Two long rows (60 each: 2*60=120 > 100, so they split) then five short ones
    # (5*10=50 <= 100, packed together).
    groups = scorer._token_budget_groups([60, 60, 10, 10, 10, 10, 10])
    assert sorted(sum(groups, [])) == list(range(7))
    for g in groups:
        lengths = [[60, 60, 10, 10, 10, 10, 10][i] for i in g]
        assert len(g) * max(lengths) <= 100 or len(g) == 1


def test_token_budget_groups_cap_row_count_even_when_tokens_allow_more() -> None:
    scorer = CurvatureScorer(config={"max_batch_tokens": 100000, "batch_size": 4})
    groups = scorer._token_budget_groups([5] * 10)
    assert [len(g) for g in groups] == [4, 4, 2]


@pytest.fixture(scope="module")
def tiny_scorer() -> CurvatureScorer:
    torch = pytest.importorskip("torch")
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    scorer = CurvatureScorer(config={"vocab_chunk_positions": 3})
    scorer.model = GPT2LMHeadModel(GPT2Config(vocab_size=50, n_positions=32, n_embd=16,
                                              n_layer=2, n_head=2)).eval()
    return scorer


def test_ll_mean_matches_position_stats_directly(tiny_scorer) -> None:
    # full_stats()'s 3rd column is lp.mean() from the same position_stats this
    # module already checks against a full-logits reference; exercised here via
    # position_stats directly (no real tokenizer needed) since _score_group is a
    # thin wrapper: tokenize -> position_stats -> curvature_stats + lp.mean().
    import torch

    r = torch.randint(0, 50, (11,))
    lp, mean, var = tiny_scorer.position_stats(r[None], torch.ones(1, 11, dtype=torch.long))[0]
    words = np.arange(len(lp))
    d_tok, d_word = curvature_stats(lp, mean, var, words)
    expected = [d_tok, d_word, float(lp.mean())]
    assert expected[2] == pytest.approx(lp.mean())         # the line full_stats adds beyond stats()
    assert np.isfinite(expected).all()


def test_padded_batch_matches_single_rows_and_reference(tiny_scorer) -> None:
    import torch

    rows = [torch.randint(0, 50, (n,)) for n in (9, 4, 7)]
    ids = torch.zeros(3, 9, dtype=torch.long)
    mask = torch.zeros(3, 9, dtype=torch.long)
    for k, r in enumerate(rows):
        ids[k, :len(r)], mask[k, :len(r)] = r, 1
    batched = tiny_scorer.position_stats(ids, mask)
    for r, (lp, m, v) in zip(rows, batched):
        single = tiny_scorer.position_stats(r[None], torch.ones(1, len(r), dtype=torch.long))[0]
        assert lp.shape == (len(r) - 1,)
        for a, b in zip((lp, m, v), single):
            np.testing.assert_allclose(a, b, atol=1e-5)
        # Reference: the Fast-DetectGPT formula on the full logits, as in the baseline.
        with torch.no_grad():
            logp = torch.log_softmax(tiny_scorer.model(r[None]).logits[0, :-1].float(), -1)
        ref_m = (logp.exp() * logp).sum(-1).numpy()
        np.testing.assert_allclose(lp, logp.gather(-1, r[1:, None]).squeeze(-1).numpy(), atol=1e-5)
        np.testing.assert_allclose(m, ref_m, atol=1e-5)
