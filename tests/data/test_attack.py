"""Tests for the hybrid attack in src/data/attack.py."""
import random

from src.data.attack import attacked_id, hybrid_edit, run_hybrid, split_sentences

EN = ("The model is very important, and it helps students learn quickly. "
      "However, it is hard to use.\n\nMany people try it every day. It can be big.")


def test_attacked_id_suffix():
    assert attacked_id("en_x_1", "hybrid") == "en_x_1_hyb"


def test_split_sentences_rejoins_exactly():
    sents, seps = split_sentences(EN)
    assert "".join(s + p for s, p in zip(sents, seps)) == EN


def test_hybrid_edits_and_preserves_paragraphs():
    out, counts = hybrid_edit(EN, "en", random.Random(1))
    assert out != EN
    assert out.count("\n") == EN.count("\n")
    assert sum(counts.values()) == 1  # round(0.2 * 4 sentences) = 1 targeted


def test_run_hybrid_resumable(tmp_path):
    rows = [{"id": "en_1", "text": EN, "language": "en", "label": 1, "length_words": 20}]
    out = tmp_path / "a.jsonl"
    assert run_hybrid(rows, out, 7)["rows"] == 1
    assert run_hybrid(rows, out, 7)["skipped"] == 1
    assert len(out.read_text(encoding="utf-8").splitlines()) == 1
