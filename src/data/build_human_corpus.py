"""Build the human-written corpus for one language bucket (resumable CLI).

Usage::

    python -m src.data.build_human_corpus --bucket en --target 2200
        [--config configs/data.yaml] [--seed 42] [--max-minutes 0]

For every ``writer_L1_band`` configured for the bucket under
``human_corpus.buckets.<bucket>.bands`` in ``configs/data.yaml``, the sources
listed for that band are pulled in order. Each source document yields at most
ONE passage of ``passage_words.min``-``passage_words.max`` whitespace words
(cut at a sentence boundary; the target length is drawn deterministically per
document so long sources do not all land on the cap), near-duplicates are
dropped with MinHash LSH (datasketch), and every accepted row is appended to
``<output_dir>/<bucket>.jsonl`` immediately, so a killed run loses nothing.

Ids are deterministic (``<bucket>_<source>_<source_id>``). On start the script
loads what is already on disk, rebuilds the dedup index from it, and pulls only
the shortfall per band / source. Re-running the same command is always safe.

Row shape - a superset of :mod:`src.data.schema` (``split`` is assigned later
by ``freeze_splits``; ``length_tokens`` is filled at scoring time)::

    id | text | label=0 | language | code_mix_ratio | generator=None | domain |
    length_words | attack_type="clean" | writer_L1_band | split=None |
    source | source_id | provenance

Adding a source = one generator that yields :class:`RawDoc`, registered in
:data:`SOURCE_LOADERS`, plus its entry in ``configs/data.yaml``.

Serves docs/parts-plan.md Parts 1-3 and docs/master-execution-plan.md 2.1.2.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import random
import re
import sys
import time
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from src.data.schema import LABEL_HUMAN, LANGUAGE_BUCKETS
from src.utils.config import load_config
from src.utils.io import ensure_dir, read_jsonl

log = logging.getLogger("build_human_corpus")

# ---------------------------------------------------------------------------
# Text helpers (pure functions; unit-tested offline)
# ---------------------------------------------------------------------------
_WS_RE = re.compile(r"\s+")
# Sentence end = terminal punctuation (Latin or Indic danda) followed by space.
_SENT_END_RE = re.compile(r"(?<=[.!?।॥])\s+")
_LATIN_ALPHA_RE = re.compile(r"[A-Za-z]")
_ANY_ALPHA_RE = re.compile(r"[^\W\d_]", re.UNICODE)
_URL_TOKEN_RE = re.compile(r"\bURL_\d+\b")
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)  # word tokens: no digits, no punctuation
_REPLACEMENT_CHAR = "�"
# Surface artefacts that would make human text trivially separable from machine
# text: LaTeX left by Wikipedia scrapes, wiki markup, placeholder tokens, and a
# space before punctuation (a template "hole" in wikimedia dumps, or PTB
# tokenisation). A spaced ellipsis (" ...") is allowed.
_ARTEFACT_RE = re.compile(
    r"\\(?:displaystyle|textstyle|scriptstyle|left|right|frac|sqrt|mathbf|ldots)\b"
    r"|\[\[|\]\]|\{\{|<ref|&nbsp;|thumb\||URL_\d"
    r"|\s[,;:!?।॥]|\s\.(?!\.)"
)

# Alphabet of each bucket's script, for the purity check that keeps a bucket
# monolingual (per-language calibration is meaningless if hi is half English).
# Indic loanwords and acronyms stay in Latin, so the gate is a ratio, not "all".
SCRIPT_RANGES: dict[str, str] = {
    "deva": r"ऀ-ॿ꣠-ꣿ",   # Devanagari (+ extended) - hi, and cm when not romanised
    "telu": r"ఀ-౿",                # Telugu - te
    "latn": r"A-Za-z",                       # Latin - en, and romanised cm
}
_SCRIPT_RE_CACHE: dict[str, re.Pattern[str]] = {}


def script_ratio(text: str, script: str) -> float:
    """Fraction of the letters in ``text`` that belong to ``script`` (0.0-1.0).

    Only characters counted as letters by ``_ANY_ALPHA_RE`` form the
    denominator, and the numerator is the subset of *those* in the script's
    range. Counting the range directly would include Devanagari combining
    vowel signs, which are not letters, and push the ratio above 1.0.
    """
    if script not in SCRIPT_RANGES:
        raise KeyError(f"unknown script '{script}' (known: {sorted(SCRIPT_RANGES)})")
    letters = _ANY_ALPHA_RE.findall(text)
    if not letters:
        return 0.0
    pattern = _SCRIPT_RE_CACHE.get(script)
    if pattern is None:
        pattern = _SCRIPT_RE_CACHE[script] = re.compile(f"[{SCRIPT_RANGES[script]}]")
    return sum(1 for ch in letters if pattern.match(ch)) / len(letters)


def repair_holes(text: str) -> str:
    """Repair the commonest template holes in wikimedia dump text and empty brackets:
    ``Albedo (; ) is`` -> ``Albedo is``; ``Lincoln ( ; February 12`` -> ``Lincoln (February 12``;
    ``see ( ) here`` -> ``see here``. Holes that cannot be repaired are caught by :func:`has_artefact`."""
    text = re.sub(r"\s*[(\[]\s*[;,]*\s*[)\]]", "", text)   # empty / punctuation-only brackets
    text = re.sub(r"\(\s*[;,]+\s*", "(", text)              # leading hole inside a parenthesis
    return normalise_ws(text)


def has_artefact(text: str) -> bool:
    """True if ``text`` still carries a surface artefact (see ``_ARTEFACT_RE``) or U+FFFD."""
    return _REPLACEMENT_CHAR in text or bool(_ARTEFACT_RE.search(text))


def normalise_ws(text: str) -> str:
    """Collapse all whitespace (incl. newlines) to single spaces."""
    return _WS_RE.sub(" ", text or "").strip()


def n_words(text: str) -> int:
    """Whitespace word count - the length unit used for passages and bins."""
    return len(text.split())


def split_sentences(text: str) -> list[str]:
    """Split on terminal punctuation + whitespace (script-agnostic, no abbreviation model)."""
    return [s for s in _SENT_END_RE.split(normalise_ws(text)) if s]


def detokenise_ptb(text: str) -> str:
    """Undo PTB-style tokenisation (``word , word 's ( x )``) found in HC3's ELI5/WikiQA answers.

    Left as-is, the space-before-punctuation pattern would make human text
    trivially separable from machine text by a surface artefact.
    """
    text = _URL_TOKEN_RE.sub("", text)
    text = normalise_ws(text)
    text = re.sub(r'"\s*([^"]*?)\s*"', r'"\1"', text)            # " quoted " -> "quoted"
    text = re.sub(r"\s+([.,;:!?%)\]}])", r"\1", text)            # word , -> word,
    text = re.sub(r"([(\[{$])\s+", r"\1", text)                  # ( word -> (word
    text = re.sub(r"\s+(n't)\b", r"\1", text)                    # do n't -> don't
    text = re.sub(r"\s+('(?:s|re|ve|m|ll|d))\b", r"\1", text)    # it 's -> it's
    text = re.sub(r"(\w) - (\w)", r"\1-\2", text)                # oscar - winning -> oscar-winning
    return repair_holes(text)                                    # "( URL_0 )" -> "( )" -> gone


def clean_informal(text: str) -> str:
    """Normalise spacing in social / chat text (``kaisa hai ?`` -> ``kaisa hai?``).

    Informal writing genuinely contains a space before punctuation, so unlike a
    scraped template hole this is not corrupt text. It is normalised rather than
    rejected for two reasons: it would otherwise cost ~20 % of the scarce
    pre-ChatGPT chat data, and a typing habit no LLM reproduces would act as a
    shortcut feature that inflates every AUROC.

    NOTE: ``clean_artifacts.py`` must apply this same normalisation to machine
    ``cm`` text, or the asymmetry becomes a shortcut in the other direction.
    """
    return detokenise_ptb(text)


def fix_sentence_spacing(text: str) -> str:
    """Insert the missing space in ``...449 BC.In 494 BC...`` (HC3 open_qa)."""
    return re.sub(r"([.!?])([A-Z][a-z])", r"\1 \2", text)


_WIKI_SKIP_PREFIXES = ("*", "|", "!", "{", "==", "Category:", "thumb|")


def wikipedia_prose(text: str) -> str:
    """Keep prose paragraphs of a wikimedia/wikipedia article in order; drop headings, lists, tables."""
    paragraphs: list[str] = []
    for para in re.split(r"\n+", text or ""):
        para = para.strip()
        if not para or para.startswith(_WIKI_SKIP_PREFIXES):
            continue
        # Section headings are short lines with no terminal punctuation.
        if n_words(para) <= 8 and not para.endswith((".", "!", "?", '"', ")", "।", "॥")):
            continue
        paragraphs.append(para)
    return repair_holes(" ".join(paragraphs))


def _target_len(key: str, lo: int, hi: int) -> int:
    """Deterministic per-document target length in ``[lo, hi]`` (stable across re-runs)."""
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=4).digest()
    return lo + int.from_bytes(digest, "big") % (hi - lo + 1)


def passage_from_doc(text: str, min_words: int, max_words: int, key: str = "") -> str | None:
    """Return ONE passage from the start of ``text``: leading sentences up to a
    per-document target length (never above ``max_words``), or None if the
    document cannot reach ``min_words`` without breaking a sentence."""
    target = _target_len(key, min_words, max_words) if key else max_words
    kept: list[str] = []
    total = 0
    for sentence in split_sentences(text):
        words = n_words(sentence)
        if total + words > max_words:
            break
        kept.append(sentence)
        total += words
        if total >= target:
            break
    if total < min_words:
        return None
    return " ".join(kept)


def samanantar_sentence_ok(sentence: str) -> bool:
    """Keep only complete, English-script prose sentences from the Samanantar ``src`` column."""
    words = sentence.split()
    if not 6 <= len(words) <= 60:
        return False
    alpha = _ANY_ALPHA_RE.findall(sentence)
    if not alpha or len(_LATIN_ALPHA_RE.findall(sentence)) / len(alpha) < 0.97:
        return False  # not the English side / script leak
    if _REPLACEMENT_CHAR in sentence or sentence.isupper():
        return False
    first = sentence[0]
    if not (first.isupper() or first.isdigit() or first in "\"'("):
        return False  # lowercase start = fragment
    if any(tok in sentence for tok in ("http", "www.", "@", '""', "|", "\t")):
        return False
    if has_artefact(sentence) or sentence[-1] in ":;,-(":
        return False  # tokenised fragment ("Alia Bhatt , Sidharth") / list item / trailing separator
    if sentence[-1] not in ".!?\"'":
        capitalised = sum(1 for w in words if w[:1].isupper())
        if capitalised / len(words) > 0.5:
            return False  # Title Case headline
    return True


# Romanised Hindi function words, for measuring how much Hindi is in a Latin-script
# code-mixed passage. Function words are used rather than content words because
# they carry the grammar (so their share tracks the matrix language), they are a
# closed class, and their romanisation is comparatively stable. Common spelling
# variants are listed side by side (hai/hain/he, nahi/nahin, mein/me/mai).
# Forms that are also ordinary English words are deliberately LEFT OUT ("to",
# "is", "us", "me", "he", "main", "the", "in", "so", "do", "hi", "an", "at"),
# so English-only text scores near zero and the ratio stays a Hindi signal.
HINDI_ROMAN_WORDS: frozenset[str] = frozenset("""
hai hain hoon hun hota hoti hote hua hui hue tha thi thay
ka ke ki ko se kaa kee kii koi kuch kuchh sab sabhi
aur ya kyunki kyuki kyonki lekin magar par phir fir toh bhi
nahi nahin nahiin naa nai mat bina
kya kyun kyon kaise kaisa kaisi kab kahan kahaan kidhar kitna kitne kitni
jo joh wo woh vo ye yeh yah vah inko unko iska uska inka unka isko
apna apne apni khud
hum ham hamara hamare humara tum tumhara tumhe tumne aap aapka aapke aapki aapko
mera meri mere tera teri tere uske uski unke unki mein
mujhe mujhko humein hamein tumko usko
karna karne karta karti karte kiya kiye kar karo kare karen karunga karenge
raha rahi rahe rahna rahta rahti
gaya gayi gaye gya jana jata jati jate
diya diye dena dete deta lena liya lete leta
hona hone jaise agar warna varna
bahut bohot bohut thoda zyada jyada achha acha accha achhi theek thik sahi galat
saath sath baad pehle pahle abhi kal aaj andar bahar upar niche
yaar bhai behen dost matlab bas arre haan haa
chahiye chahta chahti chahte milega milta mila
dekh dekha dekhna dekhte suna sunna bol bola bolna keh kaha kehna
samajh samjha pata malum
""".split())


def hindi_word_ratio(text: str) -> float:
    """Fraction of word tokens that are Romanised Hindi function words (0.0-1.0).

    This is the ``cm`` bucket's gate and also fills ``code_mix_ratio``: it
    separates genuine Hinglish from the English-only and Devanagari-only rows
    that sit in the same source files.
    """
    tokens = _WORD_RE.findall(text.lower())
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if t in HINDI_ROMAN_WORDS) / len(tokens)


def allocate(target: int, shares: dict[str, float]) -> dict[str, int]:
    """Largest-remainder split of ``target`` rows across bands by share (sums exactly to target)."""
    total_share = sum(shares.values())
    if total_share <= 0:
        raise ValueError("band shares must sum to a positive number")
    raw = {band: target * share / total_share for band, share in shares.items()}
    alloc = {band: int(math.floor(v)) for band, v in raw.items()}
    remainder = target - sum(alloc.values())
    for band in sorted(raw, key=lambda b: raw[b] - alloc[b], reverse=True)[:remainder]:
        alloc[band] += 1
    return alloc


# ---------------------------------------------------------------------------
# Near-duplicate filter
# ---------------------------------------------------------------------------
class Deduper:
    """MinHash LSH over lower-cased word ``shingle_words``-grams (datasketch)."""

    def __init__(self, num_perm: int = 128, threshold: float = 0.8, shingle_words: int = 5) -> None:
        from datasketch import MinHash, MinHashLSH  # lazy: optional at import time

        self._minhash_cls = MinHash
        self.num_perm = int(num_perm)
        self.shingle_words = int(shingle_words)
        self.lsh = MinHashLSH(threshold=float(threshold), num_perm=self.num_perm)

    def _minhash(self, text: str):
        tokens = text.lower().split()
        k = self.shingle_words
        shingles = {" ".join(tokens[i:i + k]) for i in range(max(1, len(tokens) - k + 1))}
        m = self._minhash_cls(num_perm=self.num_perm)
        m.update_batch([s.encode("utf-8") for s in shingles])
        return m

    def add(self, key: str, text: str) -> None:
        if key not in self.lsh:
            self.lsh.insert(key, self._minhash(text))

    def is_duplicate(self, text: str) -> bool:
        return bool(self.lsh.query(self._minhash(text)))

    def check_and_add(self, key: str, text: str) -> bool:
        """Insert ``text`` under ``key`` and return True, or return False if a near-duplicate exists."""
        m = self._minhash(text)
        if self.lsh.query(m) or key in self.lsh:
            return False
        self.lsh.insert(key, m)
        return True


# ---------------------------------------------------------------------------
# Source loaders: each yields RawDoc lazily; the puller stops when it has enough
# ---------------------------------------------------------------------------
@dataclass
class RawDoc:
    source: str
    source_id: str
    text: str
    domain: str
    provenance: str
    prechunked: bool = False  # True when the loader already sized the passage


@dataclass
class LoaderContext:
    min_words: int
    max_words: int
    seed: int


HC3_REPO = "Hello-SimpleAI/HC3"
HC3_DOMAINS = {"reddit_eli5": "qa_reddit_eli5", "open_qa": "qa_wikiqa", "wiki_csai": "wiki_cs_ai"}
HC3_CLEANERS: dict[str, Callable[[str], str]] = {
    "reddit_eli5": detokenise_ptb,
    "open_qa": lambda t: fix_sentence_spacing(detokenise_ptb(t)),
    "wiki_csai": repair_holes,
}


def iter_hc3(spec: dict[str, Any], ctx: LoaderContext) -> Iterator[RawDoc]:
    """HC3 human answers, round-robin across subsets, longest human answer per question."""
    from huggingface_hub import hf_hub_download

    subsets = list(spec.get("subsets", list(HC3_DOMAINS)))
    domains = {**HC3_DOMAINS, **spec.get("domains", {})}
    streams: list[tuple[str, Iterator[tuple[int, str]]]] = []
    for subset in subsets:
        path = hf_hub_download(spec.get("repo", HC3_REPO), f"{subset}.jsonl", repo_type="dataset")
        with open(path, encoding="utf-8", errors="replace") as fh:
            rows = [(i, line) for i, line in enumerate(fh) if line.strip()]
        random.Random(ctx.seed).shuffle(rows)  # deterministic visiting order
        log.info("hc3/%s: %d questions", subset, len(rows))
        streams.append((subset, iter(rows)))
    while streams:
        for entry in list(streams):
            subset, it = entry
            try:
                i, line = next(it)
            except StopIteration:
                streams.remove(entry)
                continue
            row = json.loads(line)
            answers = [a for a in (row.get("human_answers") or []) if a and a.strip()]
            if not answers:
                continue
            best = max(answers, key=n_words)
            cleaner = HC3_CLEANERS.get(subset, normalise_ws)
            yield RawDoc(
                source=spec["name"], source_id=f"{subset}:{i}", text=cleaner(best),
                domain=domains.get(subset, subset), provenance=spec.get("provenance", "hc3"),
            )


def iter_wikipedia(spec: dict[str, Any], ctx: LoaderContext) -> Iterator[RawDoc]:
    """wikimedia/wikipedia articles with page id <= max_page_id (dump is page-id ordered)."""
    from datasets import load_dataset

    ds = load_dataset(spec.get("repo", "wikimedia/wikipedia"), spec["config"], split="train", streaming=True)
    max_id = int(spec.get("max_page_id", 2_000_000))
    over_limit = 0
    for row in ds:
        page_id = int(row["id"])
        if page_id > max_id:
            over_limit += 1
            if over_limit >= 200:  # tolerate mild disorder, then stop
                log.info("wikipedia: passed page id %d, stopping", max_id)
                return
            continue
        over_limit = 0
        yield RawDoc(
            source=spec["name"], source_id=str(page_id), text=wikipedia_prose(row["text"]),
            domain=spec.get("domain", "wiki_general"), provenance=spec.get("provenance", "wikipedia"),
        )


def iter_samanantar_en(spec: dict[str, Any], ctx: LoaderContext) -> Iterator[RawDoc]:
    """English side of ai4bharat/samanantar, regrouped into passages.

    CAVEAT: the HF release is *shuffled single sentences* (verified at several
    offsets), not documents. Passages are therefore runs of consecutive,
    filtered sentences - topically incoherent but genuinely Indian-authored
    English at sentence level. Swap this for a document-level Indian-English
    source in configs/data.yaml when one becomes available.
    """
    from datasets import load_dataset

    source = spec["name"]
    domain = spec.get("domain", "news_gov_mixed")
    provenance = spec.get("provenance", "samanantar")
    for cfg in spec.get("configs", ["hi"]):
        ds = load_dataset(spec.get("repo", "ai4bharat/samanantar"), cfg, split="train", streaming=True)
        seen: set[str] = set()
        buf: list[str] = []
        idxs: list[int] = []
        total = 0
        for row in ds:
            sentence = normalise_ws(row.get("src") or "")
            if not samanantar_sentence_ok(sentence):
                continue
            key = sentence.lower()
            if key in seen:
                continue
            seen.add(key)
            if sentence[-1] not in ".!?\"'":
                sentence += "."
            words = n_words(sentence)
            if total + words > ctx.max_words:
                if total >= ctx.min_words:
                    yield RawDoc(source, f"{cfg}:{idxs[0]}-{idxs[-1]}", " ".join(buf), domain, provenance, True)
                buf, idxs, total = [], [], 0
            buf.append(sentence)
            idxs.append(int(row["idx"]))
            total += words
            if total >= _target_len(f"{cfg}:{idxs[0]}", ctx.min_words, ctx.max_words):
                yield RawDoc(source, f"{cfg}:{idxs[0]}-{idxs[-1]}", " ".join(buf), domain, provenance, True)
                buf, idxs, total = [], [], 0


def iter_indiccorp_v2(spec: dict[str, Any], ctx: LoaderContext) -> Iterator[RawDoc]:
    """ai4bharat/IndicCorpV2 (CC-0): one document per line, blank-line separated.

    The repo exposes a SINGLE config, ``indiccorp_v2``, whose *splits* are the
    languages (``hin_Deva``, ``tel_Telu``) - there is no per-language config.
    The language files are 16-27 GB, so this always streams and never downloads;
    ``max_docs`` caps how far it reads. Only ~7 % (hi) and ~2 % (te) of documents
    are long enough to yield a 120-300 word passage, so max_docs must be well
    above the row target.
    """
    from datasets import load_dataset

    split = spec["split"]
    ds = load_dataset(spec.get("repo", "ai4bharat/IndicCorpV2"),
                      spec.get("config", "indiccorp_v2"), split=split, streaming=True)
    max_docs = int(spec.get("max_docs", 250_000))
    for i, row in enumerate(ds):
        if i >= max_docs:
            log.info("indiccorp_v2/%s: stopping at max_docs=%d", split, max_docs)
            return
        text = normalise_ws(row.get("text") or "")
        if not text:
            continue  # blank separator line
        yield RawDoc(
            source=spec["name"], source_id=f"{split}:{i}", text=text,
            domain=spec.get("domain", "news_web"), provenance=spec.get("provenance", "indiccorp_v2"),
        )


def iter_comi_lingua(spec: dict[str, Any], ctx: LoaderContext) -> Iterator[RawDoc]:
    """LingoIITGN/COMI-LINGUA (CC-BY-4.0), ``TN`` config: naturally-written Hinglish.

    The ``TN`` (transliteration-normalisation) config is the only one whose
    ``Sentences`` column is Roman script - LID/MLI/NER/POS carry Devanagari
    code-mixing instead, which does not belong in a Romanised ``cm`` bucket.
    The RAW ``Sentences`` column is used, not the annotator-normalised columns:
    normalisation is an edit by a third party, and the unedited comment is the
    authentic human writing this corpus is supposed to represent.
    """
    from datasets import load_dataset

    config = spec.get("config", "TN")
    column = spec.get("column", "Sentences")
    for split in spec.get("splits", ["train", "test"]):
        ds = load_dataset(spec.get("repo", "LingoIITGN/COMI-LINGUA"), config, split=split)
        for i, row in enumerate(ds):
            text = normalise_ws(row.get(column) or "")
            if text:
                yield RawDoc(
                    source=spec["name"], source_id=f"{config}:{split}:{i}", text=clean_informal(text),
                    domain=spec.get("domain", "social_comments"),
                    provenance=spec.get("provenance", "comi_lingua"), prechunked=True,
                )


def iter_hinge(spec: dict[str, Any], ctx: LoaderContext) -> Iterator[RawDoc]:
    """LingoIITGN/HinGE (CC-BY-4.0): human-written Hinglish for Hindi-English pairs.

    ``Human-generated Hinglish`` holds a *stringified Python list* of several
    human variants per source sentence. They paraphrase one another, so only the
    longest is taken - the rest would be near-duplicates that the MinHash filter
    would drop anyway.
    """
    import ast

    from datasets import load_dataset

    column = spec.get("column", "Human-generated Hinglish")
    ds = load_dataset(spec.get("repo", "LingoIITGN/HinGE"), split=spec.get("split", "train"))
    for i, row in enumerate(ds):
        raw = row.get(column)
        if not raw:
            continue
        try:
            variants = ast.literal_eval(raw) if isinstance(raw, str) else list(raw)
        except (ValueError, SyntaxError):
            continue
        variants = [normalise_ws(v) for v in variants if isinstance(v, str) and v.strip()]
        if not variants:
            continue
        yield RawDoc(
            source=spec["name"], source_id=str(i), text=clean_informal(max(variants, key=n_words)),
            domain=spec.get("domain", "hinglish_generation"),
            provenance=spec.get("provenance", "hinge"), prechunked=True,
        )


def iter_cmu_hinglish_dog(spec: dict[str, Any], ctx: LoaderContext) -> Iterator[RawDoc]:
    """festvox/cmu_hinglish_dog (CC-BY-SA-3.0): Hinglish chat about Wikipedia film docs.

    Rows are single utterances, most far under the length floor, so consecutive
    utterances of one conversation are joined into a transcript. Conversations
    are delimited by ``(date, docIdx, user2_id)`` changing.
    """
    from datasets import load_dataset

    buf: list[str] = []
    key: tuple[Any, ...] | None = None
    start = 0
    for split in spec.get("splits", ["train", "validation", "test"]):
        ds = load_dataset(spec.get("repo", "festvox/cmu_hinglish_dog"), split=split)
        for i, row in enumerate(ds):
            this_key = (split, row.get("date"), row.get("docIdx"), row.get("user2_id"))
            utterance = normalise_ws((row.get("translation") or {}).get("hi_en") or "")
            if this_key != key:
                if buf:
                    yield RawDoc(spec["name"], f"{key[0]}:{start}-{i - 1}", clean_informal(" ".join(buf)),
                                 spec.get("domain", "chat_dialogue"),
                                 spec.get("provenance", "cmu_hinglish_dog"), True)
                buf, key, start = [], this_key, i
            if utterance:
                buf.append(utterance)
        if buf:
            yield RawDoc(spec["name"], f"{split}:{start}-end", clean_informal(" ".join(buf)),
                         spec.get("domain", "chat_dialogue"),
                         spec.get("provenance", "cmu_hinglish_dog"), True)
            buf, key = [], None


SOURCE_LOADERS: dict[str, Callable[[dict[str, Any], LoaderContext], Iterator[RawDoc]]] = {
    "hc3": iter_hc3,
    "wikipedia": iter_wikipedia,
    "samanantar_en": iter_samanantar_en,
    "indiccorp_v2": iter_indiccorp_v2,
    "comi_lingua": iter_comi_lingua,
    "hinge": iter_hinge,
    "cmu_hinglish_dog": iter_cmu_hinglish_dog,
}

# ---------------------------------------------------------------------------
# Puller / driver
# ---------------------------------------------------------------------------
_ID_SAFE_RE = re.compile(r"[^\w:.\-]", re.UNICODE)


def make_row(bucket: str, band: str, doc: RawDoc, passage: str, row_id: str,
             code_mix_ratio: float = 0.0) -> dict[str, Any]:
    return {
        "id": row_id,
        "text": passage,
        "label": LABEL_HUMAN,
        "language": bucket,
        "code_mix_ratio": round(code_mix_ratio, 4),
        "generator": None,
        "domain": doc.domain,
        "length_words": n_words(passage),
        "attack_type": "clean",
        "writer_L1_band": band,
        "split": None,
        "source": doc.source,
        "source_id": doc.source_id,
        "provenance": doc.provenance,
    }


def pull_source(
    spec: dict[str, Any],
    want: int,
    *,
    bucket: str,
    band: str,
    out_path: Path,
    existing_ids: set[str],
    deduper: Deduper,
    ctx: LoaderContext,
    deadline: float | None,
) -> tuple[int, bool, dict[str, Any]]:
    """Pull up to ``want`` new passages from one source, appending each to ``out_path``.

    Returns ``(accepted, stopped_by_time_budget, stats)`` where ``stats`` holds
    ``seen`` (documents read) and ``rejected`` (a reason -> count Counter).
    """
    name = spec["name"]
    script = spec.get("script")
    min_script = float(spec.get("min_script_ratio", 0.8))
    min_hindi = spec.get("min_hindi_word_ratio")
    min_hindi = float(min_hindi) if min_hindi is not None else None
    loader = SOURCE_LOADERS.get(spec.get("loader", name))
    if loader is None:
        raise KeyError(f"no loader registered for source '{name}' (known: {sorted(SOURCE_LOADERS)})")
    log.info("[%s/%s] %s: pulling up to %d passages", bucket, band, name, want)
    accepted, seen = 0, 0
    rejected: Counter[str] = Counter()
    with open(out_path, "a", encoding="utf-8") as fh:
        for doc in loader(spec, ctx):
            seen += 1
            row_id = f"{bucket}_{doc.source}_{_ID_SAFE_RE.sub('_', doc.source_id)}"
            if row_id in existing_ids:
                rejected["already_present"] += 1
                continue
            passage = doc.text if doc.prechunked else passage_from_doc(doc.text, ctx.min_words, ctx.max_words, row_id)
            if passage is None or not ctx.min_words <= n_words(passage) <= ctx.max_words:
                rejected["length"] += 1
                continue
            if has_artefact(passage):
                rejected["artefact"] += 1
                continue
            if script and script_ratio(passage, script) < min_script:
                rejected["wrong_script"] += 1
                continue
            cm_ratio = 0.0
            if min_hindi is not None:
                cm_ratio = hindi_word_ratio(passage)
                if cm_ratio < min_hindi:
                    rejected["not_code_mixed"] += 1
                    continue
            if not deduper.check_and_add(row_id, passage):
                rejected["near_duplicate"] += 1
                continue
            fh.write(json.dumps(make_row(bucket, band, doc, passage, row_id, cm_ratio),
                                ensure_ascii=False) + "\n")
            fh.flush()
            existing_ids.add(row_id)
            accepted += 1
            if accepted % 100 == 0:
                log.info("[%s/%s] %s: %d/%d accepted (%d docs seen)", bucket, band, name, accepted, want, seen)
            if accepted >= want:
                break
            if deadline is not None and time.monotonic() > deadline:
                log.warning("[%s/%s] %s: time budget reached after %d passages", bucket, band, name, accepted)
                return accepted, True, {"seen": seen, "rejected": dict(rejected)}
    log.info("[%s/%s] %s: accepted %d of %d docs seen; rejected %s",
             bucket, band, name, accepted, seen, dict(rejected) or "none")
    if accepted < want:
        log.warning("[%s/%s] %s exhausted %d short of its quota", bucket, band, name, want - accepted)
    return accepted, False, {"seen": seen, "rejected": dict(rejected)}


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-band and per-source counts and mean passage length (words)."""
    by_band: dict[str, list[int]] = {}
    by_source: dict[tuple[str, str, str], list[int]] = {}
    for row in rows:
        by_band.setdefault(row["writer_L1_band"], []).append(int(row["length_words"]))
        by_source.setdefault((row["writer_L1_band"], row["source"], row["domain"]), []).append(int(row["length_words"]))

    def mean(xs: list[int]) -> float:
        return (sum(xs) / len(xs)) if xs else 0.0

    return {
        "rows": len(rows),
        "mean_words": mean([int(r["length_words"]) for r in rows]),
        "bands": {b: {"rows": len(v), "mean_words": mean(v)} for b, v in sorted(by_band.items())},
        "sources": {k: {"rows": len(v), "mean_words": mean(v)} for k, v in sorted(by_source.items())},
    }


def build(bucket: str, target: int, config: dict[str, Any], seed: int = 42, max_minutes: float = 0) -> dict[str, Any]:
    """Grow ``<output_dir>/<bucket>.jsonl`` toward ``target`` rows and return the summary."""
    if bucket not in LANGUAGE_BUCKETS:
        raise ValueError(f"--bucket must be one of {LANGUAGE_BUCKETS}, got '{bucket}'")
    hc = config.get("human_corpus")
    if not hc:
        raise KeyError("config has no 'human_corpus' section (see configs/data.yaml)")
    bucket_cfg = hc.get("buckets", {}).get(bucket)
    if not bucket_cfg or not bucket_cfg.get("bands"):
        raise KeyError(f"human_corpus.buckets.{bucket}.bands is not configured")
    bands: dict[str, Any] = bucket_cfg["bands"]
    if not any(b.get("sources") for b in bands.values()):
        raise KeyError(f"no sources configured for bucket '{bucket}' yet (see docs/parts-plan.md)")

    # Passage length is global unless the bucket overrides it. cm must override:
    # Romanised Hinglish occurs as comments, chat turns and single sentences,
    # and a 120-word floor would reject essentially all of it.
    words = {**hc.get("passage_words", {}), **(bucket_cfg.get("passage_words") or {})}
    ctx = LoaderContext(int(words.get("min", 120)), int(words.get("max", 300)), int(seed))
    log.info("[%s] passage length %d-%d words", bucket, ctx.min_words, ctx.max_words)
    out_path = ensure_dir(hc.get("output_dir", "data/raw/human")) / f"{bucket}.jsonl"

    existing = read_jsonl(out_path, skip_bad_lines=True) if out_path.exists() else []
    existing_ids = {r["id"] for r in existing}
    deduper = Deduper(**hc.get("dedup", {}))
    for row in existing:
        deduper.add(row["id"], row["text"])
    counts: Counter[tuple[str, str]] = Counter((r["writer_L1_band"], r["source"]) for r in existing)
    log.info("%s: %d rows already on disk (%s)", out_path, len(existing), dict(counts) or "empty")

    band_targets = allocate(int(target), {b: float(c.get("share", 1.0)) for b, c in bands.items()})
    deadline = time.monotonic() + max_minutes * 60 if max_minutes and max_minutes > 0 else None
    stopped = False
    pulls: dict[tuple[str, str], dict[str, Any]] = {}
    for band, band_cfg in bands.items():
        band_target = band_targets[band]
        have_band = sum(v for (b, _), v in counts.items() if b == band)
        log.info("[%s/%s] target %d, have %d", bucket, band, band_target, have_band)
        for spec in band_cfg.get("sources") or []:
            if have_band >= band_target:
                break
            cap = math.ceil(float(spec.get("max_share", 1.0)) * band_target)
            want = min(cap - counts[(band, spec["name"])], band_target - have_band)
            if want <= 0:
                continue
            got, stopped, stats = pull_source(
                spec, want, bucket=bucket, band=band, out_path=out_path, existing_ids=existing_ids,
                deduper=deduper, ctx=ctx, deadline=deadline,
            )
            pulls[(band, spec["name"])] = {"accepted": got, **stats}
            counts[(band, spec["name"])] += got
            have_band += got
            if stopped:
                break
        if stopped:
            break

    summary = summarise(read_jsonl(out_path, skip_bad_lines=True))
    summary.update({"bucket": bucket, "target": int(target), "band_targets": band_targets,
                    "path": str(out_path), "stopped_by_time_budget": stopped, "pulls": pulls})
    return summary


def print_summary(summary: dict[str, Any]) -> None:
    print(f"\nbucket={summary['bucket']}  file={summary['path']}  rows={summary['rows']}  "
          f"target={summary['target']}  mean_words={summary['mean_words']:.1f}")
    print(f"{'writer_L1_band':<16}{'rows':>8}{'target':>8}{'mean_words':>12}")
    for band, stats in summary["bands"].items():
        print(f"{band:<16}{stats['rows']:>8}{summary['band_targets'].get(band, 0):>8}{stats['mean_words']:>12.1f}")
    print(f"\n{'band':<10}{'source':<16}{'domain':<18}{'rows':>8}{'mean_words':>12}")
    for (band, source, domain), stats in summary["sources"].items():
        print(f"{band:<10}{source:<16}{domain:<18}{stats['rows']:>8}{stats['mean_words']:>12.1f}")
    pulls = summary.get("pulls") or {}
    if pulls:
        print(f"\nthis run - documents read vs rejected\n{'band':<10}{'source':<16}{'seen':>8}{'kept':>8}"
              f"{'rej%':>8}  reasons")
        for (band, source), stats in pulls.items():
            seen, kept = stats["seen"], stats["accepted"]
            rejected = stats["rejected"]
            total_rej = sum(rejected.values())
            pct = 100.0 * total_rej / seen if seen else 0.0
            reasons = ", ".join(f"{k} {v} ({100.0 * v / seen:.1f}%)" for k, v in
                                sorted(rejected.items(), key=lambda kv: -kv[1])) or "none"
            print(f"{band:<10}{source:<16}{seen:>8}{kept:>8}{pct:>7.1f}%  {reasons}")
    else:
        print("\nthis run - nothing pulled (already at target)")
    short = {b: t - summary["bands"].get(b, {}).get("rows", 0) for b, t in summary["band_targets"].items()}
    short = {b: n for b, n in short.items() if n > 0}
    if summary.get("stopped_by_time_budget"):
        print("\nstopped at --max-minutes; resume with the same command")
    elif short:
        print(f"\nshort of target by {short}; sources exhausted - add sources in configs/data.yaml and re-run")
    else:
        print("\ncomplete")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bucket", required=True, choices=LANGUAGE_BUCKETS)
    parser.add_argument("--target", type=int, default=2200, help="total rows for the bucket (split by band share)")
    parser.add_argument("--config", default="configs/data.yaml", help="path to configs/data.yaml")
    parser.add_argument("--seed", type=int, default=42, help="visiting order for shuffled sources")
    parser.add_argument("--max-minutes", type=float, default=0, help="stop cleanly after this long (0 = no limit)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    from src.utils.logging import configure_logging

    configure_logging(args.log_level)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    summary = build(args.bucket, args.target, load_config(args.config), seed=args.seed, max_minutes=args.max_minutes)
    print_summary(summary)


if __name__ == "__main__":
    main()
