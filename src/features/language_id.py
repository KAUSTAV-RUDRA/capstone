"""Preprocessing: language ID, script normalisation, code-mix ratio.

Front of the pipeline (docs/project-context-master.md §3): script normalise,
Romanised detection, and code-mix ratio, then assignment to a bucket
(en / hi / te / cm) that selects the per-language calibration.

Serves docs/master-execution-plan.md Phase 2 §2.2 (preprocessing) and
Phase 3 (per-bucket routing).
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any

#: Closed-class romanised Hindi: postpositions, copulas, negation, pronouns,
#: conjunctions, auxiliaries. Function words, not content words, because a text
#: only carries them if Hindi is actually running through its grammar — a
#: machine cm row at the English floor scored 0.02-0.04 on this list against
#: 0.30-0.33 for the human text it was matched to (docs/progress.md 2026-09-21).
#:
#: Known collisions with English ("to", "me", "par", "log") give English text a
#: floor of ~0.03. The generation gate compares a machine text against its own
#: human passage, so the floor cancels there; anything reading this as an
#: absolute code-mix ratio must allow for it.
ROMANISED_HINDI_FUNCTION_WORDS: frozenset[str] = frozenset(
    "hai hain ho ka ki ke ko se mein me nahi nhi na kya bhi to toh tha thi aur ek kar karo karna "
    "raha rahe rhe hota hoga ab bas kuch sab log wala wali abhi apna apne mera meri tera tum aap "
    "hum ye yeh wo woh kaise kyun kyu jo agar par pe".split())

#: Discourse fillers deliberately NOT counted. They are the cheapest way to look
#: Hinglish without being it — "great session today, yaar" is English with a tag —
#: so a generator that bolts them onto English sentences must not score as mixed.
EXCLUDED_FILLERS: frozenset[str] = frozenset({"yaar", "yar", "bhai"})

#: Runtime bucket-assignment floor for Latin-script code-mixed text: the same
#: quantity (romanised_hindi_share) and the same value as the corpus-construction
#: source-inclusion gate (configs/data.yaml `min_hindi_word_ratio: 0.3`) — a
#: text needs at least this much Hindi running through its grammar to be routed
#: to the `cm` calibration bucket rather than `en`.
DEFAULT_CODEMIX_MIN_RATIO = 0.3

_LATIN_WORD_RE = re.compile(r"[a-z]+")

#: Unicode block ranges used for script-based language detection.
_DEVANAGARI_RANGE = (0x0900, 0x097F)
_TELUGU_RANGE = (0x0C00, 0x0C7F)


def romanised_hindi_share(text: str) -> float:
    """Share of a text's Latin-script words that are romanised Hindi function words.

    A proxy for how much Hindi grammar runs through Romanised code-mixed text,
    used by the generation gate (``src.data.generate``) to catch cm rows that
    are English in all but name. Devanagari is not counted, by design: cm is the
    Romanised bucket. 0.0 for text with no Latin-script words.
    """
    words = _LATIN_WORD_RE.findall(text.lower())
    if not words:
        return 0.0
    return sum(word in ROMANISED_HINDI_FUNCTION_WORDS for word in words) / len(words)


def _script_counts(text: str) -> dict[str, int]:
    """Count characters falling in each recognised script block."""
    counts = {"devanagari": 0, "telugu": 0, "latin": 0}
    for ch in text:
        code = ord(ch)
        if _DEVANAGARI_RANGE[0] <= code <= _DEVANAGARI_RANGE[1]:
            counts["devanagari"] += 1
        elif _TELUGU_RANGE[0] <= code <= _TELUGU_RANGE[1]:
            counts["telugu"] += 1
        elif ch.isalpha() and ch.isascii():
            counts["latin"] += 1
    return counts


def detect_language(text: str) -> str:
    """Detect the dominant script/language of a text: ``en`` / ``hi`` / ``te``.

    Script-based, not calibration-bucket-aware: Latin-script Hindi (Hinglish)
    detects as ``en`` here — :func:`assign_bucket` is what routes it to ``cm``
    using :func:`romanised_hindi_share` on top of this.
    """
    counts = _script_counts(text)
    if counts["devanagari"] >= counts["telugu"] and counts["devanagari"] > 0:
        return "hi"
    if counts["telugu"] > 0:
        return "te"
    return "en"


def code_mix_ratio(text: str) -> float:
    """Return the fraction of tokens from the embedded language (0.0-1.0).

    For the Romanised code-mixed case (the project's ``cm`` bucket) this is
    :func:`romanised_hindi_share`: the share of Latin-script words that are
    romanised Hindi function words.
    """
    return romanised_hindi_share(text)


def normalise_script(text: str) -> str:
    """Normalise Indic scripts (feeds script-aware curvature, Patent 2).

    Unicode NFC normalisation (composes combining marks into precomposed
    Devanagari/Telugu characters) plus whitespace trimming. Full script-aware
    normalisation (transliteration variants, ZWJ handling per
    docs/progress.md 2026-09-22) is Patent 2 scope beyond this sprint.
    """
    return unicodedata.normalize("NFC", text).strip()


def is_romanised(text: str, config: dict | None = None) -> bool:
    """Return True if the text is Romanised Indic (e.g. Hinglish in Latin script).

    True when the dominant script is Latin (no Devanagari/Telugu) but the
    romanised-Hindi function-word share clears the code-mix floor.
    """
    counts = _script_counts(text)
    if counts["devanagari"] > 0 or counts["telugu"] > 0:
        return False
    threshold = (config or {}).get("codemix_min_ratio", DEFAULT_CODEMIX_MIN_RATIO)
    return romanised_hindi_share(text) >= threshold


def assign_bucket(text: str, config: dict | None = None) -> str:
    """Assign one of the four calibration buckets (``en`` / ``hi`` / ``te`` / ``cm``).

    Devanagari-dominant -> ``hi``; Telugu-dominant -> ``te``; Latin-dominant
    with the romanised-Hindi share at or above the code-mix floor -> ``cm``;
    otherwise -> ``en``.
    """
    counts = _script_counts(text)
    if counts["devanagari"] >= counts["telugu"] and counts["devanagari"] > 0:
        return "hi"
    if counts["telugu"] > 0:
        return "te"
    if is_romanised(text, config):
        return "cm"
    return "en"


class LanguageIdentifier:
    """Bundles detection, normalisation, and bucket assignment for the pipeline."""

    def __init__(self, config: dict | None = None) -> None:
        """Args:
        config: Loaded config dict; only ``codemix_min_ratio`` is read here
            (defaults to :data:`DEFAULT_CODEMIX_MIN_RATIO`). Pure script/lexicon
            detection — no external models to load.
        """
        self.config = config or {}

    def identify(self, text: str) -> dict[str, Any]:
        """Return ``{language, bucket, code_mix_ratio, script, romanised}``."""
        normalised = normalise_script(text)
        language = detect_language(normalised)
        bucket = assign_bucket(normalised, self.config)
        return {
            "language": language,
            "bucket": bucket,
            "code_mix_ratio": code_mix_ratio(normalised),
            "script": "devanagari" if language == "hi" else "telugu" if language == "te" else "latin",
            "romanised": is_romanised(normalised, self.config),
        }
