"""Offline tests for src.data.generate helpers (no model load).

Run with pytest, or directly:  python tests/data/test_generate.py
Serves docs/parts-plan.md Part 4(a).
"""
from __future__ import annotations

from src.data import generate
from src.utils import modelload


def test_generator_slug_is_filesystem_safe() -> None:
    assert generate.generator_slug("Qwen/Qwen2.5-7B-Instruct") == "qwen-qwen2-5-7b-instruct"
    assert generate.generator_slug("ai-forever/mGPT") == "ai-forever-mgpt"
    assert "/" not in generate.generator_slug("meta-llama/Llama-3.1-8B-Instruct")


def test_first_sentence_handles_danda_and_truncates() -> None:
    assert generate.first_sentence("This is one. This is two.") == "This is one."
    assert generate.first_sentence("यह पहला है। यह दूसरा है।") == "यह पहला है।"
    long_sentence = " ".join(f"word{i}" for i in range(100))
    assert len(generate.first_sentence(long_sentence, max_words=40).split()) == 40


def test_length_bin_picks_the_nearest_bin() -> None:
    assert generate.length_bin(20) == 25
    assert generate.length_bin(140) == 150
    assert generate.length_bin(1000) == 300


def test_prompt_is_matched_to_the_human_passage() -> None:
    prompt = generate.build_prompt("Antibodies are proteins. They bind antigens.", "en", 150)
    assert "Antibodies are proteins." in prompt      # topic seed
    assert "about 150 words" in prompt               # length match
    assert "in English" in prompt
    hi = generate.build_prompt("यह एक वाक्य है। और दूसरा।", "hi", 200)
    assert "in Hindi" in hi and "about 200 words" in hi


def test_cm_prompt_asks_for_romanised_informal_hinglish() -> None:
    prompt = generate.build_prompt("Yaar yeh movie bahut acchi thi.", "cm", 50)
    for expected in ("Hinglish", "Roman script", "the way students text"):
        assert expected in prompt, f"cm prompt must say '{expected}'"
    assert "in English" not in prompt


def test_4bit_threshold_reads_the_size_from_the_model_id() -> None:
    assert modelload.estimate_params_b("Qwen/Qwen2.5-7B-Instruct") == 7.0
    assert modelload.estimate_params_b("Qwen/Qwen2.5-0.5B") == 0.5
    assert modelload.estimate_params_b("google/gemma-2-9b-it") == 9.0
    assert modelload.estimate_params_b("ai-forever/mGPT") == 1.3   # size not in the id
    assert modelload.should_use_4bit("Qwen/Qwen2.5-7B-Instruct") is True
    assert modelload.should_use_4bit("google/gemma-2-9b-it") is True
    assert modelload.should_use_4bit("Qwen/Qwen2.5-0.5B") is False
    assert modelload.should_use_4bit("ai-forever/mGPT") is False   # 1.3B < 3B


if __name__ == "__main__":
    import sys

    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {exc!r}")
    sys.exit(1 if failures else 0)
