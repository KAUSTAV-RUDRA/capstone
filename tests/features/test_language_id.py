"""Tests for src.features.language_id. Serves Phase 2 §2.2.3.

Run with pytest, or directly:  python tests/features/test_language_id.py
"""
from __future__ import annotations

from src.features import language_id


def test_assign_bucket_in_vocab() -> None:
    """Placeholder: assign_bucket returns one of en/hi/te/cm; code_mix_ratio in [0,1]."""
    # TODO(phase-2 step-2.2.3): test language ID + bucket assignment.
    raise NotImplementedError


def test_romanised_hindi_share_counts_function_words() -> None:
    # yeh, mera, nahi, hai are function words; kaam is a content word.
    assert language_id.romanised_hindi_share("yeh kaam mera nahi hai") == 4 / 5
    assert language_id.romanised_hindi_share("Yeh KAAM mera NAHI hai!") == 4 / 5   # case, punctuation
    assert language_id.romanised_hindi_share("The results held across every bucket.") == 0.0


def test_fillers_are_excluded_so_tagged_english_does_not_score() -> None:
    # English with a Hinglish tag bolted on is not code-mixed text.
    assert language_id.romanised_hindi_share("great session today yaar, thanks bhai") == 0.0
    assert not language_id.EXCLUDED_FILLERS & language_id.ROMANISED_HINDI_FUNCTION_WORDS


def test_share_is_zero_without_latin_words() -> None:
    assert language_id.romanised_hindi_share("") == 0.0
    assert language_id.romanised_hindi_share("यह हिंदी है") == 0.0     # cm is the Romanised bucket


if __name__ == "__main__":
    import sys

    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except NotImplementedError:
                print(f"TODO {name}")          # scaffold placeholder, not a failure
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {exc!r}")
    sys.exit(1 if failures else 0)
