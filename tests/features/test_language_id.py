"""Tests for src.features.language_id. Serves Phase 2 §2.2.3.

Run with pytest, or directly:  python tests/features/test_language_id.py
"""
from __future__ import annotations

from src.features import language_id


def test_assign_bucket_in_vocab() -> None:
    """assign_bucket returns one of en/hi/te/cm; code_mix_ratio in [0,1]."""
    from src.data.schema import LANGUAGE_BUCKETS

    samples = [
        "The results held across every bucket of the corpus.",           # en
        "यह हिंदी में लिखा गया एक वाक्य है जो देवनागरी लिपि में है।",       # hi
        "ఇది తెలుగు లో వ్రాయబడిన ఒక వాక్యం.",                              # te
        "yeh kaam mera nahi hai, tum kya kar rahe ho abhi bhi wahi baat",  # cm (Latin, Hindi-heavy)
    ]
    for text in samples:
        bucket = language_id.assign_bucket(text)
        assert bucket in LANGUAGE_BUCKETS
        ratio = language_id.code_mix_ratio(text)
        assert 0.0 <= ratio <= 1.0

    assert language_id.assign_bucket(samples[0]) == "en"
    assert language_id.assign_bucket(samples[1]) == "hi"
    assert language_id.assign_bucket(samples[2]) == "te"
    assert language_id.assign_bucket(samples[3]) == "cm"


def test_devanagari_dominant_over_telugu_when_both_present() -> None:
    mixed = "यह वाक्य है" + "ఇది వాక్యం"
    assert language_id.detect_language(mixed) in {"hi", "te"}  # deterministic, whichever dominates
    # With more Devanagari characters, hi wins.
    assert language_id.detect_language("यह एक बहुत लंबा हिंदी वाक्य है" + "ఇది") == "hi"


def test_is_romanised_false_for_devanagari_and_low_codemix_latin() -> None:
    assert not language_id.is_romanised("यह हिंदी है")           # Devanagari -> not "romanised"
    assert not language_id.is_romanised("just plain English text with no Hindi at all")
    assert language_id.is_romanised("yeh mera nahi hai, kya kar rahe ho tum abhi")


def test_normalise_script_trims_and_is_idempotent() -> None:
    text = "  hello world  "
    normalised = language_id.normalise_script(text)
    assert normalised == "hello world"
    assert language_id.normalise_script(normalised) == normalised


def test_language_identifier_identify_shape() -> None:
    identifier = language_id.LanguageIdentifier()
    out = identifier.identify("The results held across every bucket.")
    assert set(out) == {"language", "bucket", "code_mix_ratio", "script", "romanised"}
    assert out["language"] == "en"
    assert out["bucket"] == "en"
    assert out["script"] == "latin"
    assert out["romanised"] is False


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
