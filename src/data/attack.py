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
``paraphrase``  (implemented, Part 24)
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
import time
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
# English synonym swap uses WordNet (nltk) with spaCy POS tags, so only words
# that need no inflection are touched (NN, JJ, RB, VB, VBP) and the swap stays in
# the word's part of speech. hi/te/cm have no usable WordNet here, so they use
# small in-repo tables of near-interchangeable words (cm also the English table).
# Coverage is partial by design: a sentence with no swappable word falls through
# to a clause reorder or split/merge. Telugu is agglutinative, so whole-token
# matching hits rarely there and the structural edits carry most of it.
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
    "many some most all each several other now what how why who main hum tum mujhe mera meri aap kya agar jab toh".split())


def _recase(clause: str, upper: bool) -> str:
    if upper:
        return clause[:1].upper() + clause[1:]
    first = clause.split(" ", 1)[0].lower()
    return clause[:1].lower() + clause[1:] if first in _LOWERABLE else clause


_WN_POS = {"NN": "n", "JJ": "a", "RB": "r", "VB": "v", "VBP": "v"}
_EN_NLP: Any = None


def _en_nlp() -> Any:
    global _EN_NLP
    if _EN_NLP is None:
        import spacy
        _EN_NLP = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])
    return _EN_NLP


def _wordnet_synonym(word: str, pos: str) -> str | None:
    """Most frequent single-word synonym from the word's first WordNet sense, or None."""
    from nltk.corpus import wordnet as wn

    senses = wn.synsets(word, pos=pos if pos != "a" else None)
    senses = [x for x in senses if x.pos() in (pos, "s")] if pos == "a" else senses
    if not senses:
        return None
    lemmas = [l for l in senses[0].lemmas()
              if l.name().isalpha() and l.name().lower() != word.lower()]
    # The candidate must be corpus-attested and its own first sense must be this
    # one, so "solve" -> "lick" (a rare sense of lick) is rejected.
    lemmas = [l for l in lemmas if l.count() > 0
              and wn.synsets(l.name(), pos=pos if pos != "a" else None)[:1] == senses[:1]]
    return max(lemmas, key=lambda l: l.count()).name() if lemmas else None


def _edit_synonym_en(sentence: str, rng: random.Random) -> str | None:
    cands = []
    for tok in _en_nlp()(sentence):
        pos = _WN_POS.get(tok.tag_)
        if pos and tok.is_alpha and len(tok.text) >= 4 and not tok.is_stop:
            syn = _wordnet_synonym(tok.text.lower(), pos)
            if syn:
                cands.append((tok, syn))
    if not cands:
        return None
    chosen = sorted(rng.sample(cands, min(2, len(cands))), key=lambda c: c[0].idx, reverse=True)
    out = sentence
    for tok, syn in chosen:
        out = out[:tok.idx] + _swap_case(tok.text, syn) + out[tok.idx + len(tok.text):]
    return out if out != sentence else None


def _edit_synonym(sentence: str, bucket: str, rng: random.Random) -> str | None:
    if bucket == "en":
        return _edit_synonym_en(sentence, rng)
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
                   length_words=len(text.split()), source_id=r["id"],
                   attack_changed=text != r["text"])  # False rows stay in T4, reported as "attack produced no change"
        new_rows.append(new)
        totals["rows"] += 1
    append_rows(new_rows, out_path)
    return totals


# --- stubs -------------------------------------------------------------------
_BUCKET_LANG = {"en": "English", "hi": "Hindi (Devanagari script)", "te": "Telugu (Telugu script)",
                "cm": "Hinglish: Hindi written in Roman letters, mixed with English words"}


def paraphrase_prompt(text: str, bucket: str) -> str:
    """Prompt for one rewrite: own words, same language and script, same length, no commentary."""
    n_words = len(text.split())
    return (f"Rewrite the following text in your own words. Keep it in the same language "
            f"({_BUCKET_LANG[bucket]}) and the same script, keep the meaning, and keep the length "
            f"close to {n_words} words. Do not translate: if the text mixes languages or is written in "
            f"Roman letters, your rewrite must do the same and must not become plain English. "
            f"Keep paragraph breaks. Output only the rewritten text, with "
            f"no introduction, notes or explanation.\n\nText:\n{text}\n\nRewritten text:")


def make_paraphrase_worker(client: Any, model: str, info: dict[str, Any], *, fertility: dict[str, float],
                           cap: int, num_ctx: int, keep_alive: str, temperature: float, top_p: float,
                           seed: int, retries: int = 2, retry_wait_s: float = 5.0):
    """Return ``process_item(row) -> attacked row`` that paraphrases through Ollama (thread-safe)."""
    from src.data.clean_artifacts import strip_boilerplate
    from src.data.generate import token_budget

    def process_item(row: dict[str, Any]) -> dict[str, Any]:
        text = row["text"]
        num_predict = token_budget(len(text.split()), fertility[row["language"]], 1.5, 128, cap)
        options = {"num_predict": num_predict, "num_ctx": num_ctx, "temperature": temperature,
                   "top_p": top_p, "seed": hash64(f"{seed}:{row['id']}")}
        for attempt in range(retries + 1):
            try:
                resp = client.generate(model, paraphrase_prompt(text, row["language"]), options, keep_alive)
                break
            except Exception as exc:  # noqa: BLE001 - timeouts, dropped connections
                if attempt == retries:
                    raise
                log.warning("%s: %s - retrying", row["id"], exc)
                time.sleep(retry_wait_s)
        out = strip_boilerplate((resp.get("response") or "").strip())
        if not out:
            raise RuntimeError(f"{row['id']}: empty paraphrase")
        new = dict(row)
        new.update(id=attacked_id(row["id"], "paraphrase"), text=out, attack_type="paraphrase",
                   length_words=len(out.split()), source_id=row["id"], attack_changed=out != text,
                   paraphrase_model=model, paraphrase_quant=info.get("quantization"),
                   paraphrase_truncated=resp.get("done_reason") == "length",
                   length_ratio=round(len(out.split()) / max(1, len(text.split())), 3))
        return new

    return process_item


def hash64(key: str) -> int:
    import hashlib
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:4], "big")


def processing_order(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    """Seeded interleaving of buckets/labels used by the slow (LLM / MT) attacks."""
    return random.Random(seed).sample(rows, len(rows))


def cap_per_bucket(order: list[dict[str, Any]], max_machine: int, max_human: int,
                   done_ids: set[str], attack: str) -> tuple[list[dict[str, Any]], int]:
    """Keep the first ``max_machine`` machine and ``max_human`` human rows per bucket, in ``order``.

    ``order`` is the seeded processing order, so the kept rows are a random subset
    of the hybrid source rows and the same subset on every run. Returns the kept
    rows and the number of already-done rows that fall outside the cap (those stay
    in the output file but are not in scope).
    """
    caps = {LABEL_MACHINE: max_machine, LABEL_HUMAN: max_human}
    seen: dict[tuple[str, Any], int] = defaultdict(int)
    kept: list[dict[str, Any]] = []
    outside = 0
    for r in order:
        key = (r["language"], r["label"])
        if seen[key] < caps.get(r["label"], 0):
            seen[key] += 1
            kept.append(r)
        elif attacked_id(r["id"], attack) in done_ids:
            outside += 1
    return kept, outside


def _apply_cap(rows: list[dict[str, Any]], out_path: str | Path, seed: int, attack: str,
               cap: Sequence[int] | None) -> list[dict[str, Any]]:
    """Order ``rows`` and apply the optional per-bucket cap; log the in-scope total."""
    order = processing_order(rows, seed)
    if not cap:
        return order
    done = read_done_ids(out_path)
    kept, outside = cap_per_bucket(order, cap[0], cap[1], done, attack)
    in_done = sum(attacked_id(r["id"], attack) in done for r in kept)
    per = defaultdict(int)
    for r in kept:
        per[(r["language"], "machine" if r["label"] == LABEL_MACHINE else "human")] += 1
    log.info("%s cap %d machine + %d human per bucket: in-scope %d rows (%s); %d done in scope, "
             "%d done outside cap (kept in file, excluded from scope)",
             attack, cap[0], cap[1], len(kept), ", ".join(f"{k[0]}/{k[1]}={v}" for k, v in sorted(per.items())),
             in_done, outside)
    return kept


def run_paraphrase(rows: list[dict[str, Any]], out_path: str | Path, config: dict[str, Any],
                   seed: int, max_minutes: float, parallel: int | None = None,
                   cap: Sequence[int] | None = None):
    """Paraphrase ``rows`` with Qwen2.5-7B-Instruct (4-bit, local Ollama); resumable and time-boxed."""
    from src.data.generate import OllamaClient, describe_ollama_model, load_fertility, load_with_fallback
    from src.utils.resumable import run_concurrent

    mc = config["machine_corpus"]
    spec = mc["generators"]["qwen7b"]
    model = spec["ollama"]
    oc = mc.get("ollama") or {}
    client = OllamaClient(oc.get("host", "http://127.0.0.1:11434"), float(oc.get("request_timeout_s", 900)))
    info = describe_ollama_model(client, model)
    budget = mc.get("budget") or {}
    fertility, src_model = load_fertility(budget.get("fertility_csv", "results/tokenizer_fertility.csv"),
                                          spec.get("tokenizer"),
                                          budget.get("fallback_tokenizer", "Qwen/Qwen2.5-0.5B"))
    num_ctx = int(oc.get("num_ctx", 4608))
    keep_alive = str(oc.get("keep_alive", "30m"))
    workers = load_with_fallback(client, model, num_ctx, keep_alive, parallel or int(oc.get("parallel", 4)))
    dec = mc.get("decoding") or {}
    log.info("paraphrase: %s (%s), fertility from %s, %d workers", model, info["quantization"], src_model, workers)
    worker = make_paraphrase_worker(client, model, info, fertility=fertility, cap=int(budget.get("cap", 3600)),
                                    num_ctx=num_ctx, keep_alive=keep_alive,
                                    temperature=float(dec.get("temperature", 0.8)),
                                    top_p=float(dec.get("top_p", 0.95)), seed=seed)
    # Interleave buckets/labels so a time-boxed run covers every cell evenly instead of one bucket first.
    order = _apply_cap(rows, out_path, seed, "paraphrase", cap)
    return run_concurrent(order, id_of=lambda r: attacked_id(r["id"], "paraphrase"), process_item=worker,
                          out_path=out_path, workers=workers, max_minutes=max_minutes, label="paraphrases")


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
    parser.add_argument("--max-per-bucket", type=int, nargs=2, metavar=("MACHINE", "HUMAN"), default=None,
                        help="paraphrase/backtranslation only: attack the first MACHINE machine + HUMAN human "
                             "rows per bucket in the seeded processing order (a subset of the hybrid sources)")
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
    if args.attack == "paraphrase":
        rows = select_rows(Path(args.corpus), config, seed)  # same seed => same sources as hybrid
        report = run_paraphrase(rows, args.out, config, seed, args.max_minutes, cap=args.max_per_bucket)
        log.info("paraphrase: %s", report.summary())
        return
    raise NotImplementedError(f"the '{args.attack}' attack lands in {part}.")


if __name__ == "__main__":
    main()
