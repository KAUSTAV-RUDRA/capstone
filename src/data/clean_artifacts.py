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

Open
----
Dropping a machine row leaves its prompt-matched human passage without a
counterpart. Whether the pair is dropped with it, or the human row is kept, is
P2's call at split time (docs/project-context-master.md §6, §9) and is not
decided here.
"""
from __future__ import annotations

import re

from src.data.schema import Sample

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


def clean_text(text: str) -> tuple[str, float]:
    """Strip artefacts from one text.

    Args:
        text: Raw machine (or human) text.

    Returns:
        ``(cleaned_text, fraction_stripped)``.
    """
    # TODO(phase-2 step-2.1.4): remove preambles/refusals/markdown.
    raise NotImplementedError


def clean_samples(samples: list[Sample]) -> tuple[list[Sample], dict[str, float]]:
    """Clean a list of samples and return per-bucket stripped-percentage stats.

    Args:
        samples: Samples to clean in place (returned as new list).

    Returns:
        ``(cleaned_samples, {bucket: mean_fraction_stripped})``.
    """
    # TODO(phase-2 step-2.1.4): apply clean_text across the corpus.
    raise NotImplementedError
