"""Adversarial attacks on machine text (resumable CLI).

Usage::

    python -m src.data.attack --attack hybrid

Robustness under paraphrase is the first way commercial detectors fail
(docs/project-context-master.md §2), so T4 is what shows the stylometry-curvature
fusion holding where either head alone collapses. Three families, matching
RAID/DetectRL practice:

``hybrid``  (implemented, Part 23)
    20 % of sentences get a synonym swap, clause reorder, or split/merge. No GPU.
    Models the realistic case of a student editing generated text.
``paraphrase``  (stub, Part 24)
    Qwen2.5-7B-Instruct 4-bit: "Rewrite in your own words, same language, same
    length".
``backtranslation``  (stub, Part 25)
    IndicTrans2 en<->indic: en->hi->en, hi->en->hi, te->en->te; for cm,
    Devanagari-normalise via indic-nlp-library, hi->en->hi, then re-romanise.

Input is TEST-split rows only, sampled per bucket per attack: at most
``MACHINE_ROWS_PER_BUCKET`` machine rows and ``HUMAN_ROWS_PER_BUCKET`` human
rows, stratified by length bin and by generator so no generator drops out of the
table. Human rows are attacked too, or a rise in false positives under attack
would be invisible. T4 needs a precise estimate of degradation, not the whole
test set: 400 rows per cell holds the standard error on AUROC under 0.02
(decisions.md 2026-09-18). Output rows keep the source id with a suffix (``_hyb``,
``_para``, ``_bt``), set ``attack_type``, and are appended to
``data/processed/attacked.jsonl``.

Serves docs/parts-plan.md Part 4(c) and Stage 6.
"""
from __future__ import annotations

import argparse
import logging
import random
import re
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.data.generate import length_bin
from src.data.schema import ATTACK_TYPES, LABEL_HUMAN, LABEL_MACHINE
from src.utils.config import load_config
from src.utils.io import read_jsonl
from src.utils.resumable import DEFAULT_BATCH_SIZE, append_rows, read_done_ids

log = logging.getLogger("attack")

DEFAULT_CORPUS = "data/processed/corpus.jsonl"
DEFAULT_ATTACKED = "data/processed/attacked.jsonl"

#: CLI name -> (schema attack_type, id suffix, implementing part).
ATTACKS: dict[str, tuple[str, str, str]] = {
    "hybrid": ("hybrid", "_hyb", "Part 23"),
    "paraphrase": ("paraphrase", "_para", "Part 24"),
    "backtranslation": ("back_translation", "_bt", "Part 25"),
}

#: Machine TEST rows attacked per bucket per attack, stratified by length bin and
#: by generator. 400 per cell keeps the standard error on AUROC under 0.02, which
#: is the precision T4 needs (decisions.md 2026-09-18).
MACHINE_ROWS_PER_BUCKET = 400
#: Human TEST rows attacked per bucket, so attacked false positives stay measurable.
HUMAN_ROWS_PER_BUCKET = 150
#: Fraction of sentences edited by the hybrid attack.
HYBRID_EDIT_FRACTION = 0.2


def attacked_id(row_id: str, attack: str) -> str:
    """Id of the attacked copy of a row: ``en_hc3_123`` -> ``en_hc3_123_para``."""
    if attack not in ATTACKS:
        raise KeyError(f"unknown attack '{attack}'; valid: {', '.join(ATTACKS)}")
    return f"{row_id}{ATTACKS[attack][1]}"


# --- row selection -----------------------------------------------------------
def _round_robin(strata: dict[Any, list[dict[str, Any]]], quota: int,
                 rng: random.Random) -> list[dict[str, Any]]:
    """Draw up to ``quota`` rows, one per stratum in turn, so small strata are not crowded out."""
    pools = {k: rng.sample(v, len(v)) for k, v in sorted(strata.items(), key=lambda kv: str(kv[0]))}
    picked: list[dict[str, Any]] = []
    while len(picked) < quota and any(pools.values()):
        for key in pools:
            if pools[key] and len(picked) < quota:
                picked.append(pools[key].pop())
    return picked


def select_rows(corpus_path: Path, config: dict[str, Any], seed: int = 42) -> list[dict[str, Any]]:
    """Sampled TEST rows per bucket: <= MACHINE_ROWS_PER_BUCKET machine + HUMAN_ROWS_PER_BUCKET human.

    Machine rows are balanced across (generator, length bin), human rows across
    length bin. Train and cal rows are never touched: T4 measures degradation on
    test, and attacking cal rows would contaminate the conformal thresholds.
    """
    rows = [r for r in read_jsonl(corpus_path)
            if r["split"] == "test" and r["attack_type"] == "clean"]
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    for bucket in sorted({r["language"] for r in rows}):
        machine: dict[Any, list] = defaultdict(list)
        human: dict[Any, list] = defaultdict(list)
        for r in rows:
            if r["language"] != bucket:
                continue
            b = length_bin(int(r["length_words"]))
            if r["label"] == LABEL_MACHINE:
                machine[(r["generator"], b)].append(r)
            elif r["label"] == LABEL_HUMAN:
                human[b].append(r)
        out += _round_robin(machine, MACHINE_ROWS_PER_BUCKET, rng)
        out += _round_robin(human, HUMAN_ROWS_PER_BUCKET, rng)
    return out


# --- hybrid attack -----------------------------------------------------------
# No WordNet / Indic WordNet is installed (nltk is not in requirements.txt), so
# the synonym swap uses small in-repo tables of common, near-interchangeable
# words. Coverage is partial by design: a sentence with no swappable word falls
# through to a clause reorder or split/merge. Telugu is agglutinative, so whole-
# token matching hits rarely there and the structural edits carry most of it.
_EN_SYN = [
    ("big", "large"), ("small", "little"), ("important", "significant"), ("help", "assist"),
    ("show", "demonstrate"), ("use", "employ"), ("start", "begin"), ("end", "finish"),
    ("get", "obtain"), ("buy", "purchase"), ("many", "numerous"), ("need", "require"),
    ("try", "attempt"), ("hard", "difficult"), ("easy", "simple"), ("fast", "quick"),
    ("often", "frequently"), ("also", "additionally"), ("however", "nevertheless"),
    ("because", "since"), ("about", "regarding"), ("enough", "sufficient"), ("choose", "select"),
    ("keep", "retain"), ("ask", "inquire"), ("answer", "response"), ("idea", "concept"),
    ("problem", "issue"), ("way", "method"), ("change", "alter"), ("build", "construct"),
    ("make", "create"), ("lead", "guide"), ("main", "primary"), ("huge", "enormous"),
    ("good", "excellent"), ("bad", "poor"), ("happy", "pleased"), ("job", "task"),
    ("people", "individuals"), ("students", "learners"), ("quickly", "rapidly"),
    ("really", "truly"), ("very", "highly"), ("mainly", "chiefly"),
    ("especially", "particularly"), ("various", "different"), ("improve", "enhance"),
    ("increase", "raise"), ("reduce", "lessen"), ("allow", "permit"), ("provide", "offer"),
    ("understand", "grasp"), ("complete", "finish"), ("beautiful", "lovely"),
    ("strong", "powerful"), ("clear", "obvious"), ("whole", "entire"),
]
_HI_SYN = [
    ("बहुत", "अत्यधिक"), ("ज़रूरी", "आवश्यक"), ("जरूरी", "आवश्यक"), ("मदद", "सहायता"),
    ("शुरू", "आरंभ"), ("खत्म", "समाप्त"), ("सुंदर", "खूबसूरत"), ("बड़ा", "विशाल"),
    ("समय", "वक्त"), ("दुनिया", "संसार"), ("देश", "राष्ट्र"), ("काम", "कार्य"),
    ("कारण", "वजह"), ("महत्वपूर्ण", "अहम"), ("समस्या", "दिक्कत"), ("तरीका", "ढंग"),
    ("छात्र", "विद्यार्थी"), ("अक्सर", "प्रायः"), ("हमेशा", "सदैव"), ("इसलिए", "अतः"),
    ("लेकिन", "परंतु"), ("सही", "उचित"), ("ज्ञान", "जानकारी"), ("सरकार", "शासन"),
]
_TE_SYN = [
    ("చాలా", "ఎంతో"), ("ముఖ్యమైన", "ప్రధానమైన"), ("సహాయం", "తోడ్పాటు"), ("ప్రపంచం", "లోకం"),
    ("సమయం", "కాలం"), ("పని", "కార్యం"), ("కారణం", "హేతువు"), ("అందమైన", "సుందరమైన"),
    ("కానీ", "అయితే"), ("ఎల్లప్పుడూ", "ఎప్పుడూ"), ("సమస్య", "ఇబ్బంది"), ("ప్రజలు", "జనాలు"),
]
_CM_SYN = [
    ("bahut", "kaafi"), ("accha", "badhiya"), ("achha", "badhiya"), ("mushkil", "kathin"),
    ("madad", "help"), ("dikkat", "problem"), ("dost", "friend"), ("log", "people"),
    ("kaam", "work"), ("samay", "time"), ("lekin", "magar"), ("shuru", "start"),
]


def _syn_table(bucket: str) -> dict[str, str]:
    pairs = {"en": _EN_SYN, "hi": _HI_SYN, "te": _TE_SYN, "cm": _CM_SYN + _EN_SYN}[bucket]
    table: dict[str, str] = {}
    for a, b in pairs:  # both directions; the first pairing wins for a repeated word
        table.setdefault(a, b)
        table.setdefault(b, a)
    return table


_SYN: dict[str, dict[str, str]] = {b: _syn_table(b) for b in ("en", "hi", "te", "cm")}

#: Sentence boundary: terminal punctuation (incl. danda) followed by whitespace.
#: Capturing, so ``re.split`` keeps the original whitespace for an exact rejoin.
_SENT_SPLIT_RE = re.compile(r"((?<=[.!?।॥])\s+)")
_TERMINAL_RE = re.compile(r"([.!?।॥]+[\"')\]]*)\s*$")
_WORD_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)?", re.UNICODE)

#: Connective used to merge two sentences, per bucket.
_MERGE_CONJ = {"en": ", and", "hi": " और", "te": " మరియు", "cm": ", aur"}
#: Whole-word clause boundaries (besides a comma) a sentence may be split or reordered on.
#: cm is longer: chat-style Romanised text is often one unpunctuated run-on.
_CLAUSE_WORDS = {
    "en": ("and", "but", "while", "because", "so"),
    "hi": ("और", "लेकिन", "क्योंकि", "तथा", "परंतु"),
    "te": ("మరియు", "కానీ", "ఎందుకంటే"),
    "cm": ("aur", "lekin", "kyunki", "but", "and", "toh", "par", "ya", "phir", "jab", "agar", "ki"),
}
#: Minimum words on each side of a cut.
_MIN_CLAUSE_WORDS = 2


def split_sentences(text: str) -> tuple[list[str], list[str]]:
    """Split on terminal punctuation into (sentences, separators).

    ``separators[i]`` is the original whitespace after ``sentences[i]`` and the
    last is ``""``. Rejoining untouched pieces reproduces ``text`` exactly, so
    the attack never alters newlines or spacing outside the sentences it edits.
    """
    parts = _SENT_SPLIT_RE.split(text)
    return parts[0::2], parts[1::2] + [""]


def _clause_cut(sentence: str, bucket: str, rng: random.Random) -> tuple[str, str] | None:
    """Cut a sentence into (head, tail) at a comma or clause word, or None.

    Both halves keep >= ``_MIN_CLAUSE_WORDS`` words so the edit moves a real
    clause. A comma is consumed; a clause word stays at the start of the tail.
    """
    words = _TERMINAL_RE.sub("", sentence).split(" ")
    m = _MIN_CLAUSE_WORDS
    cuts: list[int] = []  # word index where the tail starts
    for i, w in enumerate(words):
        if m <= i <= len(words) - m - 1:
            if w.endswith(","):
                cuts.append(i + 1)
            elif w.lower() in _CLAUSE_WORDS[bucket]:
                cuts.append(i)
    cuts = [i for i in cuts if m <= i <= len(words) - m]
    if not cuts:
        return None
    i = rng.choice(cuts)
    return " ".join(words[:i]).rstrip(","), " ".join(words[i:])


def _terminal(sentence: str) -> str:
    m = _TERMINAL_RE.search(sentence)
    return m.group(1) if m else ""


def _swap_case(src: str, dst: str) -> str:
    return dst[:1].upper() + dst[1:] if src[:1].isupper() else dst


#: Sentence-initial words safe to lowercase mid-sentence; anything else may be a proper noun.
_LOWERABLE = frozenset(
    "the a an it its this that these those he she they we i you his her their our my your there "
    "in on at for with by as if when while although however but and so then also yaar yeh woh "
    "main hum tum mujhe mera meri aap kya agar jab toh".split())


def _recase(clause: str, upper: bool) -> str:
    if upper:
        return clause[:1].upper() + clause[1:]
    first = clause.split(" ", 1)[0].lower()
    return clause[:1].lower() + clause[1:] if first in _LOWERABLE else clause


def _edit_synonym(sentence: str, bucket: str, rng: random.Random) -> str | None:
    table = _SYN[bucket]
    hits = [m for m in _WORD_RE.finditer(sentence) if m.group(0).lower() in table]
    if not hits:
        return None
    chosen = sorted(rng.sample(hits, min(2, len(hits))), key=lambda m: m.start(), reverse=True)
    out = sentence
    for m in chosen:
        out = out[:m.start()] + _swap_case(m.group(0), table[m.group(0).lower()]) + out[m.end():]
    return out if out != sentence else None


def _edit_reorder(sentence: str, bucket: str, rng: random.Random) -> str | None:
    cut = _clause_cut(sentence, bucket, rng)
    if cut is None:
        return None
    head, tail = cut
    words = tail.split(" ")
    conj = ""
    if words[0].lower() in _CLAUSE_WORDS[bucket]:
        conj, words = words[0] + " ", words[1:]  # "A, but B" -> "B, but A": the connective stays between
    if len(words) < _MIN_CLAUSE_WORDS:
        return None
    first, second = " ".join(words), head
    if bucket in ("en", "cm"):
        first, second = _recase(first, sentence[:1].isupper()), _recase(second, False)
    return f"{first}, {conj}{second}{_terminal(sentence)}"


def _edit_split(sentence: str, bucket: str, rng: random.Random) -> str | None:
    cut = _clause_cut(sentence, bucket, rng)
    if cut is None:
        return None
    head, tail = cut
    words = tail.split(" ")
    if words[0].lower() in _CLAUSE_WORDS[bucket]:
        words = words[1:]  # the connective is dropped when its clause becomes a sentence
    if len(words) < _MIN_CLAUSE_WORDS:
        return None
    tail = " ".join(words)
    if bucket in ("en", "cm"):
        tail = _recase(tail, True)
    stop = "।" if bucket == "hi" else "."
    return f"{head}{stop} {tail}{_terminal(sentence)}"


def _edit_merge(sentence: str, nxt: str, bucket: str) -> str:
    left = _TERMINAL_RE.sub("", sentence)
    right = _recase(nxt, False) if bucket in ("en", "cm") else nxt
    return f"{left}{_MERGE_CONJ[bucket]} {right}"


def hybrid_edit(text: str, bucket: str, rng: random.Random,
                fraction: float = HYBRID_EDIT_FRACTION) -> tuple[str, dict[str, int]]:
    """Apply the hybrid edit to one text; returns (edited text, count per edit kind).

    ``round(fraction * n_sentences)`` sentences (at least 1) are each given one
    of synonym swap, clause reorder, or split/merge, chosen at random; an
    inapplicable edit falls through to the others. A merge consumes the
    following sentence. ``unedited`` counts targeted sentences no edit fitted.
    Whitespace between and around sentences is preserved.
    """
    lead = text[:len(text) - len(text.lstrip())]
    trail = text[len(text.rstrip()):]
    sents, seps = split_sentences(text.strip())
    n = len(sents)
    counts = {"synonym": 0, "reorder": 0, "split_merge": 0, "unedited": 0}
    if n == 0 or not text.strip():
        return text, counts
    targets = set(rng.sample(range(n), min(max(1, round(fraction * n)), n)))
    out: list[str] = []
    i = 0
    while i < n:
        s = sents[i]
        if i not in targets:
            out.append(s + seps[i])
            i += 1
            continue
        can_merge = i + 1 < n and (i + 1) not in targets and "\n" not in seps[i]  # never merge across a line break
        kinds = ["synonym", "reorder", "split_merge"]
        rng.shuffle(kinds)
        res, used, consumed = None, "", 0
        for kind in kinds:
            if kind == "synonym":
                res = _edit_synonym(s, bucket, rng)
            elif kind == "reorder":
                res = _edit_reorder(s, bucket, rng)
            elif can_merge and rng.random() < 0.5:
                res, consumed = _edit_merge(s, sents[i + 1], bucket), 1
            else:
                res = _edit_split(s, bucket, rng)
                if res is None and can_merge:
                    res, consumed = _edit_merge(s, sents[i + 1], bucket), 1
            if res is not None:
                used = kind
                break
        if res is None:
            out.append(s + seps[i])
            counts["unedited"] += 1
        else:
            out.append(res + seps[i + consumed])
            counts[used] += 1
        i += 1 + consumed
    return lead + "".join(out) + trail, counts


def apply_hybrid(texts: Sequence[str], buckets: Sequence[str], seed: int = 42) -> list[str]:
    """Edit HYBRID_EDIT_FRACTION of sentences: synonym swap, clause reorder, or split/merge."""
    rng = random.Random(seed)
    return [hybrid_edit(t, b, rng)[0] for t, b in zip(texts, buckets)]


def run_hybrid(rows: list[dict[str, Any]], out_path: str | Path, seed: int) -> dict[str, int]:
    """Attack ``rows`` and append to ``out_path``; ids already there are skipped (resumable)."""
    done = read_done_ids(out_path)
    totals = {"rows": 0, "skipped": 0, "no_change": 0,
              "synonym": 0, "reorder": 0, "split_merge": 0, "unedited": 0}
    new_rows: list[dict[str, Any]] = []
    for r in rows:
        new_id = attacked_id(r["id"], "hybrid")
        if new_id in done:
            totals["skipped"] += 1
            continue
        rng = random.Random(f"{seed}:{r['id']}")  # per-row stream: independent of row order / resume
        text, counts = hybrid_edit(r["text"], r["language"], rng)
        for k, v in counts.items():
            totals[k] += v
        totals["no_change"] += text == r["text"]
        new = dict(r)
        new.update(id=new_id, text=text, attack_type="hybrid",
                   length_words=len(text.split()), source_id=r["id"])
        new_rows.append(new)
        totals["rows"] += 1
    append_rows(new_rows, out_path)
    return totals


# --- stubs -------------------------------------------------------------------
def apply_paraphrase(texts: Sequence[str], buckets: Sequence[str], config: dict[str, Any]) -> list[str]:
    """Rewrite each text with Qwen2.5-7B-Instruct 4-bit, same language and length."""
    # TODO(stage-6 part-24): batch prompts through the 4-bit generator.
    raise NotImplementedError("apply_paraphrase lands in parts-plan Part 24")


def apply_backtranslation(texts: Sequence[str], buckets: Sequence[str], config: dict[str, Any]) -> list[str]:
    """Round-trip each text through IndicTrans2 and back into its own language."""
    # TODO(stage-6 part-25): en->hi->en, hi->en->hi, te->en->te; cm via
    # Devanagari normalisation, hi->en->hi, then re-romanisation.
    raise NotImplementedError("apply_backtranslation lands in parts-plan Part 25")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--attack", required=True, choices=sorted(ATTACKS))
    parser.add_argument("--max-minutes", type=float, default=0)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--corpus", default=DEFAULT_CORPUS)
    parser.add_argument("--out", default=DEFAULT_ATTACKED)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=None, help="default: corpus.seed from the config")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    from src.utils.logging import configure_logging

    configure_logging(args.log_level)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    attack_type, suffix, part = ATTACKS[args.attack]
    assert attack_type in ATTACK_TYPES, f"{attack_type} missing from schema.ATTACK_TYPES"
    config = load_config(args.config)
    seed = args.seed if args.seed is not None else int(config["corpus"]["seed"])
    log.info("attack=%s attack_type=%s suffix=%s out=%s seed=%d", args.attack, attack_type, suffix,
             args.out, seed)
    if args.attack == "hybrid":
        rows = select_rows(Path(args.corpus), config, seed)
        totals = run_hybrid(rows, args.out, seed)
        log.info("hybrid: selected %d rows; %s", len(rows), totals)
        return
    raise NotImplementedError(f"the '{args.attack}' attack lands in {part}.")


if __name__ == "__main__":
    main()
