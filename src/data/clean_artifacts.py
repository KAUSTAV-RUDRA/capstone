"""Strip generation artefacts from machine text, and drop rows that stay separable.

Removes preambles ("Sure! Here is..."), refusals, and markdown so machine
text is not trivially separable by surface cues. Reports the percentage of
text stripped.

Serves docs/master-execution-plan.md Phase 2 §2.1.4.

Two stages, in this order
-------------------------
1. **Strip** (:func:`clean_text`, which includes :func:`strip_boilerplate` for
   trailing assistant chatter): a text-level transform, as above.
2. **Drop**: four row-level rules below. They run on the *stripped* text, so a
   preamble that stage 1 removes cannot cause a drop.

The generation gate (:func:`src.data.generate.gate_stats`) imports
:func:`strip_boilerplate` and applies it before every check, for the same reason:
judging raw text would fail rows that stage 1 saves (decisions.md 2026-09-22).

The drop rules apply to **every generator and every bucket** — qwen7b, gemma,
mistral, llama, phi — not only to the generator they were calibrated on. Each
model produces its own version of these defects, so each bucket-and-generator
combination is measured and reported separately rather than assumed clean.

Rules 1, 2 and 4 apply **symmetrically to human and machine rows**. Dropping a
defect from the machine side only would make machine text *cleaner* than human
text, which is itself a separable signal — the shortcut, reversed. Rule 3 is
machine-only because ``truncated`` exists only on generated rows. Dropped rows are
**moved to data/raw/discarded/, never deleted** (the practice set on 2026-09-22
for qwen7b's v0 cm rows).

The rules
---------
1. **Emoji** — drop any row containing one (:func:`src.data.generate.has_emoji`,
   which excludes the lone ZWJ; it joins Devanagari conjuncts).
2. **Instruction echo** — drop any row that leaks the prompt's own wording. The
   pattern is **per bucket, derived from that bucket's own template**, since each
   template leaks different words. For cm: ``\\broman\\b|hinglish|\\bemoji``. A
   pattern is only usable if its rate on that bucket's *human* rows is near zero
   (see the baselines below); one that fires on ordinary human text is matching
   topic, not echo. ``\\broman\\b`` is deliberately bounded — unbounded ``roman``
   matches "romance" and scores 42 of 2,200 human cm rows.
3. **Truncated** — drop rows with ``truncated`` true (``done_reason == "length"``).
   The token cap cut them mid-sentence, which is a shape cue independent of
   authorship.
4. **Stray script** — drop rows carrying a script the bucket does not use.
   Allowed: en Latin · cm Latin · hi Devanagari + Latin · te Telugu + Latin (hi and
   te keep Latin because their human corpora were built with a 0.8 script-purity
   gate, so loanwords and acronyms are normal). Anything else — Cyrillic, Hangul,
   CJK, Arabic, and Devanagari inside cm — is a tokenizer artefact. Guard: if a
   rule would drop more than ~2 % of a bucket's **human** rows, it is mis-tuned for
   that bucket and must be re-measured, not applied.

Calibration evidence (measured 2026-09-22)
------------------------------------------
Human baselines, per bucket, n = 2,200 each — the reference the thresholds are set
against, and the cost of applying each rule symmetrically::

    bucket   emoji   echo (cm pattern)   stray script
    en       2       -                   4   (deva 1, hangul 1, cjk 2)
    hi       2       -                   13  (telu 1, cyr 2, hangul 3, cjk 5, arab 5)
    te       2       -                   26  (deva 20, cjk 2, arab 5)
    cm       0       3                   9   (deva 9)

qwen7b cm v1, n = 701 (the regenerated bucket, docs/progress.md 2026-09-22)::

    emoji 3 · echo 25 · truncated 13 · stray script 18 (deva 13, cjk 7, cyr 3,
    hangul 1) · union 54 rows = 7.7 % of the bucket

The rules overlap barely (largest overlap: 2 rows), so the union is close to the
sum. Of the 25 echo hits, 1-2 are false positives ("roman numeral watch"), which
is the accepted cost of a rule that catches 23 real leaks against a human rate of
3 in 2,200. For contrast, the discarded v0 rows scored emoji 28 and echo 7 — the
old regime leaked less prompt wording because it was barely following the prompt.

Stage 1 in detail (measured on qwen7b, 2026-09-24)
--------------------------------------------------
- **Preamble**: a leading "Certainly!" / "Of course." / "Absolutely!" (44 qwen7b
  en rows open this way), a "Here is a short passage of about 150 words on …:"
  line, and "Sure, I can …" sentences. A bare leading "Sure," is **kept**: in cm it
  is in character ("Sure, buddy! Abhi main khel rha hoon…").
- **Markdown**: bold/italic markers (``**``, ``__``) and header hashes. 14 qwen7b
  rows carry them. List numbering is left alone; it is also human.
- **cm**: :func:`src.data.build_human_corpus.clean_informal`, the normalisation
  every human cm passage already went through (corpus card §4 carry-over).

Instruction echo also catches **meta responses** that answer the instruction
instead of the topic ("Certainly, please provide the topic you would like me to
write about", "As an AI, I don't…") and **chat-template leaks** (one qwen7b hi row
ends in ``<|im_start|>user 请继续用 Hindi 回答我…``). Stripping cannot rescue either.

Resolved
--------
Dropping a machine row leaves its prompt-matched human passage without a
counterpart. The human row is **kept** (review2-sprint Day 1: every clean human
row goes to cal or train/test); pairs are not dropped together.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any

#: Trailing assistant chatter: an offer to continue, or a "hope this helps" sign-off,
#: as a WHOLE final line. gemma ends about 3 in 24 cm passages this way ("Let me know
#: if you'd like me to continue the conversation! 😊"), qwen7b none in 701.
#:
#: Each pattern must match the entire line, and that line must be the last one. Both
#: anchors are load-bearing. In the same 24-row probe, two rows end with the same
#: words *inside* the passage — "You know kya film dekhne ka mood hai? Let me know!"
#: — and a substring rule would eat text the model was asked to write. A trailing
#: emoji, or any sign-off punctuation, is part of the line and goes with it.
_BOILERPLATE_LINE: tuple[re.Pattern[str], ...] = (
    re.compile(r"let me know if you(?:'d| would)? (?:like|want).*", re.I),
    re.compile(r"(?:i )?hope (?:this|that|it) (?:helps|was helpful).*", re.I),
    re.compile(r"feel free to (?:ask|let me know|share|reach out).*", re.I),
    re.compile(r"would you like (?:me )?to .*", re.I),
    re.compile(r"do you want (?:me )?to .*", re.I),
    re.compile(r"i can (?:also )?(?:continue|write|add|expand|provide) .*if you.*", re.I),
)

#: How many trailing chatter lines to take off one passage. More than one happens
#: ("...continue? / Let me know!"); an unbounded loop on a pathological row could
#: strip the passage itself.
_MAX_BOILERPLATE_LINES = 3


def strip_boilerplate(text: str) -> str:
    """Drop trailing assistant chatter ("Let me know if you'd like me to continue! 😊").

    Only a **final line** that matches one of :data:`_BOILERPLATE_LINE` in full is
    removed, up to :data:`_MAX_BOILERPLATE_LINES` times. This is generation
    packaging rather than text the prompt asked for, and it carries surface cues —
    emoji, English sign-offs — that would otherwise separate the classes for
    reasons unrelated to authorship.

    Shared with the generation gate, which must judge what cleaning will keep.
    """
    lines = text.rstrip().split("\n")
    for _ in range(_MAX_BOILERPLATE_LINES):
        while lines and not lines[-1].strip():
            lines.pop()
        if not lines or not any(p.fullmatch(lines[-1].strip()) for p in _BOILERPLATE_LINE):
            break
        lines.pop()
    return "\n".join(lines).rstrip()


#: Leading assistant preamble, removed from the start of a text. Ordered: the
#: "Here is …:" form first, since it often follows an interjection.
_PREAMBLE: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*(?:(?:sure|certainly|of course|absolutely)[!,.]?\s+)?"
               r"here(?:'s| is| are)\b[^\n:]{0,150}:\s*", re.I),
    re.compile(r"^\s*sure[!,.]\s+i(?: can|'d be happy to| will)\b[^.!\n]*[.!]\s+", re.I),
    re.compile(r"^\s*(?:certainly|of course|absolutely)[!.]\s+", re.I),
    re.compile(r"^\s*(?:ज़रूर|जरूर|निश्चित रूप से)[!।,]\s*"),
    re.compile(r"^\s*ఖచ్చితంగా[!.,]\s*"),
)

_MD_EMPHASIS = re.compile(r"\*\*|__")
_MD_HEADER = re.compile(r"^[ \t]*#{1,6}[ \t]+", re.M)

#: Rule 2, per bucket. Every pattern was measured against that bucket's 2,200
#: human rows before it was used (see the module docstring for the rule).
#: ``<|…|>`` is a chat-template token leaking into the output, in any bucket.
_TEMPLATE_LEAK = r"<\|[a-z_]+\|>"
ECHO_PATTERNS: dict[str, re.Pattern[str]] = {
    "en": re.compile(_TEMPLATE_LEAK + r"|\bprovide the topic\b|\babout \d+ words\b"
                     r"|\bthe given instruction\b|\bas an ai\b|\bcontinuing naturally\b", re.I),
    # Not "in Hindi"/"in Telugu": they match 6 hi and 16 te human news bylines
    # ("News18 Telugu - …", "… in Hindi (करंट अफेयर्स)") against 1-2 machine rows.
    "hi": re.compile(_TEMPLATE_LEAK + r"|\b\d+ words\b", re.I),
    "te": re.compile(_TEMPLATE_LEAK + r"|\b\d+ words\b|\bdo not stop early\b", re.I),
    "cm": re.compile(_TEMPLATE_LEAK + r"|\broman\b|hinglish|\bemoji", re.I),
}

#: Rule 4. Scripts a bucket may carry; any letter from a script in STRAY_SCRIPTS
#: that is not allowed here drops the row.
ALLOWED_SCRIPTS: dict[str, frozenset[str]] = {
    "en": frozenset({"LATIN"}),
    "cm": frozenset({"LATIN"}),
    "hi": frozenset({"LATIN", "DEVANAGARI"}),
    "te": frozenset({"LATIN", "TELUGU"}),
}
#: First word of the Unicode character name -> script. Greek is absent on
#: purpose: human en uses it for symbols (α, π) and it is not a tokenizer artefact.
STRAY_SCRIPTS: frozenset[str] = frozenset({
    "DEVANAGARI", "TELUGU", "CYRILLIC", "HANGUL", "CJK", "HIRAGANA", "KATAKANA",
    "ARABIC", "BENGALI", "GUJARATI", "GURMUKHI", "KANNADA", "MALAYALAM", "ORIYA",
    "TAMIL", "THAI", "HEBREW",
})

#: Rule guard: a symmetric rule dropping more than this share of a bucket's HUMAN
#: rows is matching the bucket, not the defect, and must be re-measured.
HUMAN_DROP_GUARD = 0.02

DROP_RULES: tuple[str, ...] = ("emoji", "echo", "truncated", "stray_script", "empty")


def clean_text(text: str, bucket: str) -> tuple[str, float]:
    """Stage 1: strip artefacts from one text, human or machine alike.

    Args:
        text: Raw text.
        bucket: Language bucket; ``cm`` also gets ``clean_informal``.

    Returns:
        ``(cleaned_text, fraction_of_characters_stripped)``.
    """
    from src.data.build_human_corpus import clean_informal   # heavy module; import on use

    out = strip_boilerplate(text)
    for pattern in _PREAMBLE:
        out = pattern.sub("", out, count=1)
    out = _MD_HEADER.sub("", _MD_EMPHASIS.sub("", out))
    if bucket == "cm":
        out = clean_informal(out)
    out = out.strip()
    stripped = 1.0 - len(out) / len(text) if text else 0.0
    return out, max(0.0, stripped)


def stray_scripts(text: str, bucket: str) -> set[str]:
    """Scripts in ``text`` that ``bucket`` does not use (rule 4)."""
    allowed = ALLOWED_SCRIPTS[bucket]
    found: set[str] = set()
    for ch in set(text):
        if not ch.isalpha():
            continue
        script = unicodedata.name(ch, "").split(" ", 1)[0]
        if script in STRAY_SCRIPTS and script not in allowed:
            found.add(script)
    return found


def drop_reasons(row: dict[str, Any], cleaned: str) -> list[str]:
    """Stage 2: every drop rule ``row`` fails, judged on its stripped text."""
    from src.data.generate import has_emoji   # generate imports this module

    bucket = row["language"]
    reasons: list[str] = []
    if has_emoji(cleaned):
        reasons.append("emoji")
    if ECHO_PATTERNS[bucket].search(cleaned):
        reasons.append("echo")
    if row.get("truncated"):
        reasons.append("truncated")
    if stray_scripts(cleaned, bucket):
        reasons.append("stray_script")
    if not cleaned.split():
        reasons.append("empty")
    return reasons


def row_source(row: dict[str, Any]) -> str:
    """Reporting key: the human corpus source, or the generator for machine rows."""
    return row.get("generator") or row.get("source") or "unknown"


def clean_rows(rows: Iterable[dict[str, Any]], *, enforce_guard: bool = True
               ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Strip then drop, identically for human and machine rows.

    Each returned row is a copy with ``text`` replaced by the cleaned text and
    ``length_words`` recounted; dropped rows also carry ``drop_reasons``.

    Returns:
        ``(kept, dropped, report)``. ``report["by_source"]`` maps
        ``(bucket, source)`` to ``{"n", "dropped", "stripped_rows", rule: count}``.

    Raises:
        ValueError: If a symmetric rule drops more than :data:`HUMAN_DROP_GUARD`
            of a bucket's human rows (``enforce_guard``).
    """
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    by_source: dict[tuple[str, str], Counter] = defaultdict(Counter)
    human_rule: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        cleaned, frac = clean_text(row["text"], row["language"])
        out = dict(row, text=cleaned, length_words=len(cleaned.split()))
        reasons = drop_reasons(row, cleaned)
        stats = by_source[(row["language"], row_source(row))]
        stats["n"] += 1
        stats["stripped_rows"] += frac > 0
        for reason in reasons:
            stats[reason] += 1
            if row["label"] == 0:
                human_rule[row["language"]][reason] += 1
        if reasons:
            stats["dropped"] += 1
            dropped.append(dict(out, drop_reasons=reasons, raw_text=row["text"]))
        else:
            kept.append(out)
    human_n = Counter(r["language"] for r in kept + dropped if r["label"] == 0)
    breaches = [f"{b} {rule}: {n}/{human_n[b]} human rows"
                for b, rules in human_rule.items() for rule, n in rules.items()
                if n > HUMAN_DROP_GUARD * human_n[b]]
    if breaches and enforce_guard:
        raise ValueError("drop rule mis-tuned (over the human guard): " + "; ".join(breaches))
    return kept, dropped, {"by_source": dict(by_source), "guard_breaches": breaches}


def format_report(report: dict[str, Any]) -> str:
    """Per bucket x source: rows, % dropped, and the count behind each rule."""
    lines = [f"{'bucket':6} {'source':18} {'rows':>5} {'dropped':>8} {'%':>6}  "
             + " ".join(f"{r:>12}" for r in DROP_RULES) + f" {'stripped':>9}"]
    for (bucket, source), s in sorted(report["by_source"].items()):
        pct = 100.0 * s["dropped"] / s["n"] if s["n"] else 0.0
        lines.append(f"{bucket:6} {source:18} {s['n']:>5} {s['dropped']:>8} {pct:>5.1f}%  "
                     + " ".join(f"{s[r]:>12}" for r in DROP_RULES) + f" {s['stripped_rows']:>9}")
    return "\n".join(lines)
