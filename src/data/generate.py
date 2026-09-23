"""Generate prompt-matched machine text for one generator (resumable CLI).

Usage::

    python -m src.data.generate --generator qwen7b --buckets en,hi,te,cm --max-minutes 110
        [--backend ollama|hf] [--model TAG_OR_HF_ID] [--fraction 1.0]
        [--config configs/data.yaml] [--limit N] [--seed 42]

``--generator`` is an alias from ``machine_corpus.generators`` in
configs/data.yaml: ``qwen7b``, ``gemma``, ``mistral`` (seen) and ``llama``,
``phi`` (held-out, test only).

Prompt matching
    Every machine passage is matched to one human passage: the prompt is that
    passage's **first sentence**, and the requested length is its **length
    bin**, so human and machine text share topic and length distribution by
    construction. Without that, a detector can separate the classes on topic or
    length alone and the reported AUROC means nothing
    (docs/project-context-master.md §6).

Assignment
    Each human passage gets exactly ONE seen generator, ``sha256(prompt_id) mod
    3``, so each seen generator covers ~1/3 of every bucket and the seen machine
    side is 1:1 with the human side. Each held-out generator takes a disjoint
    30 % slice from a second, independent hash. Python's ``hash()`` is salted
    per process and would reshuffle the assignment on every run, so it is not
    used.

Token budget
    ``num_predict = human_words x fertility(bucket) x 1.5 + 128``, capped at
    3600, with fertility read from results/tokenizer_fertility.csv for the
    generator's own tokenizer (the Qwen2.5 row when that is missing). A flat
    words-x-4 budget cut every Hindi and Telugu passage short: Qwen needs 4.8
    tokens per Hindi word and 11.9 per Telugu word. Rows record ``done_reason``
    and ``truncated`` so cleaning can drop the ones the cap still cuts.

Backends
    ``ollama`` (default) runs a persistent pool of
    ``machine_corpus.ollama.parallel`` worker threads
    (:func:`src.utils.resumable.run_concurrent`). Each pulls the next prompt and
    posts it to ``/api/generate`` (``stream=false``) on its own, so a server
    slot is refilled the moment its request finishes rather than waiting for a
    batch's slowest member. Each row is written the moment it completes. The
    server only runs requests concurrently if it was started with as many
    slots. On Windows, set it once and restart the Ollama app::

        setx OLLAMA_NUM_PARALLEL 4      # then Quit Ollama from the tray and reopen it

    ``%LOCALAPPDATA%/Ollama/server.log`` prints ``OLLAMA_NUM_PARALLEL:4`` at
    startup when it took effect. Without it the server queues requests one at a
    time: same output, slower. The progress line (every 8 completed requests)
    shows the effective concurrency, so a queueing server is visible at once.

    ``hf`` is the fallback: transformers + bitsandbytes 4-bit via
    src/utils/modelload.py, in batches. One output file never mixes backends or
    models, since that would mix quantisations; the run refuses to append to a
    file written by a different one.

Length gate
    Each bucket is taken to its first ``machine_corpus.gate.probe_rows`` rows and
    judged before the rest of that bucket is generated: the machine/human length
    ratio must sit inside 0.75-1.25 and at most 20 % of rows may have hit the
    token cap, or the run aborts. A prompt regime that does not transfer between
    buckets is therefore worth 24 rows, not a corpus (:func:`run_with_gates`).
    cm is also held to its human passages' romanised-Hindi share and to zero
    emoji rows.

Output is ``data/raw/machine/<generator>.jsonl``, one row per assigned human
passage, carrying ``prompt_id`` back to it. Ids already present are skipped, so
re-running the same command resumes.

Serves docs/parts-plan.md Stage 3.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from src.data.clean_artifacts import strip_boilerplate
from src.data.schema import LABEL_MACHINE, LANGUAGE_BUCKETS
from src.features.language_id import romanised_hindi_share
from src.utils.config import load_config
from src.utils.io import read_jsonl
from src.utils.resumable import (DEFAULT_BATCH_SIZE, RunReport, read_done_ids, run_concurrent,
                                 run_resumable)

log = logging.getLogger("generate")

BACKENDS: tuple[str, ...] = ("ollama", "hf")

#: Length bins (word counts) the prompt asks for. The human passage's own length
#: picks the bin, so the machine length distribution tracks the human one.
LENGTH_BINS: tuple[int, ...] = (25, 50, 100, 150, 200, 250, 300)

LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
    "te": "Telugu",
    "cm": "Hinglish",
}

#: Per-bucket instruction. cm needs its own: asking for "Hinglish" alone tends to
#: produce Devanagari Hindi or formal prose, neither of which matches the human
#: cm bucket (Romanised, informal). te and cm carry an explicit "do not stop
#: early" framing: measured 2026-09-18, the plain instruction makes Qwen stop at
#: 0.31 of the requested Telugu length with done_reason="stop" on 12 of 12
#: passages — under-production, not truncation.
#:
#: en and hi both stay on the plain template, for different reasons. en lands at
#: 0.90 as-is. hi *over*-produces under the insistent framing (1.40 with the
#: sentence clause, 1.21 without) but under-produces at 0.57 on the plain one —
#: and compensation can only inflate the ask, so the plain template is the one
#: the existing mechanism can correct, exactly as it corrects te. Every bucket's
#: behaviour was measured on its own probe — none of it transfers by analogy
#: (decisions.md 2026-09-18).
PROMPT_TEMPLATES: dict[str, str] = {
    "default": ('{first_sentence}\n\nWrite about {n_words} words in {language} '
                'on this topic, continuing naturally.'),
    "indic": ('{first_sentence}\n\nContinue this topic in {language} as a complete '
              'article of at least {n_words} words, written as {n_sentences} or more '
              'full sentences. Do not stop early and do not summarise; develop the '
              'topic in detail.'),
    # cm from 2026-09-21. The template it replaces wrote English in all but name
    # (romanised-Hindi share 0.02-0.04 against a human 0.30-0.33) and ran long on
    # short passages. Probed 3 variants x 24 stratified passages, then the long
    # bin's ask: this one lands 0-40 at 1.19 and 40+ at 1.09, Hindi 0.37 against
    # 0.29, no emoji (scripts/probes/, docs/progress.md 2026-09-21). {n_words} is
    # the ask from ASK_FROM_HUMAN, not the bin.
    "cm": ('{first_sentence}\n\nWrite about {n_words} words on this topic in Hinglish: mix Hindi '
           'and English words within every sentence, with the Hindi written in Roman script, '
           'casual, like students texting each other, as {n_sentences} or more full sentences. '
           'No emoji. Continue naturally and do not stop early.'),
}

#: Which template each bucket uses. Changing this changes the corpus: rows
#: generated under different templates are not comparable within a bucket.
BUCKET_TEMPLATE: dict[str, str] = {"en": "default", "hi": "default", "te": "indic", "cm": "cm"}

_SENT_END_RE = re.compile(r"(?<=[.!?।॥])\s+")
_SLUG_RE = re.compile(r"[^A-Za-z0-9]+")


def generator_slug(generator: str) -> str:
    """Filesystem-safe name: ``Qwen/Qwen2.5-7B-Instruct`` -> ``qwen-qwen2-5-7b-instruct``."""
    return _SLUG_RE.sub("-", generator).strip("-").lower()


def first_sentence(text: str, max_words: int = 40) -> str:
    """Leading sentence of a passage, truncated to ``max_words``.

    This is the topic seed. Truncation matters: a very long first sentence would
    hand the model most of the human passage to copy rather than a topic.
    """
    parts = _SENT_END_RE.split(text.strip(), maxsplit=1)
    sentence = (parts[0] if parts else text).strip()
    words = sentence.split()
    return " ".join(words[:max_words]) if len(words) > max_words else sentence


def length_bin(n_words: int, bins: Sequence[int] = LENGTH_BINS) -> int:
    """Nearest length bin to ``n_words``."""
    return min(bins, key=lambda b: abs(b - n_words))


def sentence_target(bucket: str, n_words: int) -> int:
    """Sentences to ask for alongside the word target.

    cm needs its own rule. Its human passages average 57 words, and a floor of 8
    sentences implies ~88 words whatever the word target says — measured
    2026-09-18, that drove cm passages of 21 human words to 84 machine words
    (4.0x), because the sentence clause silently overrides the word clause.
    ``max(2, n/15)`` lands cm at 1.08 of human length.

    hi/te keep ``max(8, n/12)``: their bins are all >= 100, so the floor never
    binds there and this leaves their prompts byte-identical.
    """
    if BUCKET_TEMPLATE.get(bucket) == "cm":
        return max(2, n_words // 15)
    return max(8, n_words // 12)


def build_prompt(text: str, bucket: str, n_words: int) -> str:
    """Prompt-matched instruction for one human passage."""
    template = PROMPT_TEMPLATES[BUCKET_TEMPLATE.get(bucket, "default")]
    return template.format(
        first_sentence=first_sentence(text),
        n_words=n_words,
        n_sentences=sentence_target(bucket, n_words),
        language=LANGUAGE_NAMES.get(bucket, bucket),
    )


# --- assignment -------------------------------------------------------------

def _hash64(key: str) -> int:
    """Stable 64-bit hash of ``key`` (sha256), identical across runs and machines."""
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")


def seen_generator_for(prompt_id: str, seen: Sequence[str]) -> str:
    """The one seen generator for a human passage: ``sha256(prompt_id) mod len(seen)``."""
    return seen[_hash64(prompt_id) % len(seen)]


def heldout_generator_for(prompt_id: str, heldout: Sequence[str], share: float) -> str | None:
    """The held-out generator for a passage, or None: disjoint ``share`` slices of a second hash.

    The key is salted, so this hash is independent of the seen assignment and
    each held-out slice cuts evenly across all three seen generators.
    """
    if share * len(heldout) > 1:
        raise ValueError(f"{len(heldout)} held-out generators x {share} share exceeds 100 %")
    slot = int((_hash64("heldout:" + prompt_id) / 2**64) // share)
    return heldout[slot] if slot < len(heldout) else None


def generator_roster(generators: dict[str, dict[str, Any]]) -> tuple[list[str], list[str]]:
    """``(seen, heldout)`` aliases, each ordered by its ``slot``."""
    def by_role(role: str) -> list[str]:
        members = sorted((spec["slot"], alias) for alias, spec in generators.items()
                         if spec["role"] == role)
        slots = [slot for slot, _ in members]
        if slots != list(range(len(slots))):
            raise ValueError(f"{role} generator slots must be 0..{len(slots) - 1}, got {slots}")
        return [alias for _, alias in members]

    return by_role("seen"), by_role("heldout")


def make_assignment(generator: str, generators: dict[str, dict[str, Any]],
                    heldout_share: float) -> Callable[[str], bool]:
    """Predicate: is this human passage (by id) assigned to ``generator``?"""
    seen, heldout = generator_roster(generators)
    if generator in seen:
        return lambda prompt_id: seen_generator_for(prompt_id, seen) == generator
    if generator in heldout:
        return lambda prompt_id: heldout_generator_for(prompt_id, heldout, heldout_share) == generator
    raise KeyError(f"generator '{generator}' has no seen/heldout role")


# --- per-bucket caps --------------------------------------------------------

def stratified_cap(prompts: Sequence[dict[str, Any]], limit: int) -> list[str]:
    """Ids of ``limit`` prompts, keeping the length-bin distribution of the input.

    Largest-remainder allocation across the bins, then a deterministic pick
    inside each bin ordered by ``sha256(id)``. Stable across runs and machines,
    and independent of whatever order the human corpus happens to be in, so the
    same generator always selects the same passages.
    """
    if limit >= len(prompts):
        return [p["id"] for p in prompts]
    by_bin: dict[int, list[dict[str, Any]]] = {}
    for prompt in prompts:
        by_bin.setdefault(prompt["n_words"], []).append(prompt)
    exact = {b: limit * len(ps) / len(prompts) for b, ps in by_bin.items()}
    take = {b: int(v) for b, v in exact.items()}
    order = sorted(by_bin, key=lambda b: (exact[b] - take[b], b), reverse=True)
    index = 0
    while sum(take.values()) < limit:          # largest remainder first, then by bin
        take[order[index % len(order)]] += 1
        index += 1
    chosen: list[str] = []
    for bin_words, in_bin in by_bin.items():
        ranked = sorted(in_bin, key=lambda p: _hash64(p["id"]))
        chosen.extend(p["id"] for p in ranked[:take[bin_words]])
    return sorted(chosen)


def load_or_record_selection(path: Path, prompts: Sequence[dict[str, Any]], limit: int,
                             bucket: str, generator: str) -> set[str]:
    """Frozen ids for a capped bucket: computed and written once, reused after.

    The selection is deterministic, so this file is a record rather than the
    source of truth — but recording it means a later edit to the selection rule
    cannot silently change a corpus that is already half generated.
    """
    if path.exists():
        record = json.loads(path.read_text(encoding="utf-8"))
        recorded = int(record.get("limit", len(record["ids"])))
        if recorded != limit:
            # A frozen record outranks the config, so a recut cap would otherwise
            # be read, logged as "reusing", and silently ignored — the same class
            # of silent-config bug as a compliance ratio keyed on the bucket alone
            # (2026-09-23). Recutting is a decision, so it is made by hand.
            raise SystemExit(
                f"\nSELECTION CAP CHANGED — bucket '{bucket}', generator '{generator}'\n"
                f"  {path} was frozen at limit {recorded}, the config now asks for {limit}.\n\n"
                f"A frozen selection is reused verbatim, so this run would have generated the OLD\n"
                f"{recorded} passages and ignored the new cap. Archive the record and re-run to cut a\n"
                f"fresh one — but only if no rows for this bucket are on disk yet, because a recut\n"
                f"changes WHICH passages are chosen, not just how many:\n"
                f"  mkdir -p data/processed/selection/superseded\n"
                f"  mv {path} data/processed/selection/superseded/\n")
        log.info("bucket %s: reusing frozen selection of %d ids from %s",
                 bucket, len(record["ids"]), path)
        return set(record["ids"])
    ids = stratified_cap(prompts, limit)
    chosen = set(ids)
    bins: dict[int, int] = {}
    for prompt in prompts:
        if prompt["id"] in chosen:
            bins[prompt["n_words"]] = bins.get(prompt["n_words"], 0) + 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"generator": generator, "bucket": bucket, "limit": limit,
                                "selected_from": len(prompts), "n": len(ids),
                                "bins": dict(sorted(bins.items())), "ids": ids},
                               indent=1), encoding="utf-8")
    return chosen


# --- token budget -----------------------------------------------------------

def load_fertility(csv_path: str | Path, tokenizer: str | None, fallback: str,
                   buckets: Sequence[str] = LANGUAGE_BUCKETS) -> tuple[dict[str, float], str]:
    """Tokens per word by bucket for ``tokenizer``, else ``fallback``.

    Returns ``(table, model_used)``. A model counts only if it has an ``ok`` row
    for every requested bucket.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run: python scripts/tokenizer_fertility.py")
    by_model: dict[str, dict[str, float]] = {}
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("status") == "ok" and row.get("tokens_per_word"):
                by_model.setdefault(row["model"], {})[row["bucket"]] = float(row["tokens_per_word"])
    for model in (tokenizer, fallback):
        if model and all(b in by_model.get(model, {}) for b in buckets):
            return by_model[model], model
    raise ValueError(f"{path} has no complete fertility rows for {tokenizer!r} or fallback {fallback!r}")


def token_budget(n_words: int, fertility: float, overshoot: float = 1.3,
                 pad: int = 64, cap: int = 2048) -> int:
    """Generation budget in tokens: ``n_words x fertility x overshoot + pad``, capped."""
    return min(cap, int(n_words * fertility * overshoot + pad))


#: Measured machine/human word ratio under the template each bucket actually uses
#: (probe 2026-09-18: 12 te passages x 4 prompt variants; en from 730 finished rows).
#: te's 0.62 is the ratio under the "indic" template, NOT the 0.31 measured under
#: the plain one — compensating a control-template ratio on top of the new framing
#: would double-count and drive every passage into the cap.
#:
#: Measured per bucket on its own 12-passage probe, 2026-09-18:
#:   en 1.0   plain template, 0.90 as-is, no correction needed
#:   hi 0.55  plain template, under which it writes 0.55-0.57x the ask - so the
#:            ask is inflated x1.8, the same correction te gets. Two independent
#:            measurements agree: the probe (0.55) and the real 731-row hi
#:            corpus generated under this template (0.57)
#:   te 0.62  insistent template, measured under that template (not the 0.31 the
#:            plain one gave) - compensating a control ratio would double-count
#:   cm 1.0   unused from 2026-09-21: cm's ask comes from ASK_FROM_HUMAN instead
COMPLIANCE_RATIO: dict[str, float] = {"en": 1.0, "hi": 0.55, "te": 0.62, "cm": 1.0}

#: Per-generator overrides of :data:`COMPLIANCE_RATIO`, keyed generator alias
#: then bucket. The flat map above stays every generator's default, so a
#: generator absent from here resolves exactly as it did before this key
#: existed — qwen7b's 6,284 frozen rows included.
#:
#: A compliance ratio is a property of the generator AND the bucket, never the
#: bucket alone: it measures what fraction of the ask THIS model writes in THIS
#: language. The values above were all measured on qwen and then applied to
#: gemma unchanged, which is how gemma's hi ask came out inflated x1.8 — gemma
#: writes longer on Indic and shorter on English (en 0.80 against qwen's 0.87),
#: so its hi probe landed at 1.51 of human length and failed the gate
#: (2026-09-23). decisions.md 2026-09-18 already required every regime to be
#: measured on its own probe and never carried across by analogy; keying this on
#: the bucket alone is how the config carried it across silently anyway.
COMPLIANCE_BY_GENERATOR: dict[str, dict[str, float]] = {}


def excluded_buckets(generator: str, role: str, heldout_drop: Sequence[str],
                     by_generator: dict[str, Sequence[str]]) -> dict[str, str]:
    """Buckets ``generator`` does not generate, mapped to why.

    Two independent exclusions, both reported so a missing bucket is always
    traceable to the rule that removed it:

    * by ROLE — a held-out generator skips ``heldout_drop`` (en), because
      held-out models are test-only and exist to prove generalisation, which
      hi/te/cm already test (decisions.md 2026-09-23).
    * by GENERATOR — ``by_generator[generator]``, for a model whose tokenizer
      makes a bucket structurally unrunnable rather than merely badly tuned.
      mistral reads Telugu at 13.309 tokens/word, so every te ask pins at the
      3600-token cap whatever the passage length: the budget sets the length
      instead of the passage, which no template or ratio can reach
      (decisions.md 2026-09-24).

    A bucket excluded by both reports the per-generator reason, the more
    specific of the two.
    """
    excluded: dict[str, str] = {}
    if role == "heldout":
        for bucket in heldout_drop:
            excluded[bucket] = "held-out role, machine_corpus.heldout_drop_buckets"
    for bucket in by_generator.get(generator) or ():
        excluded[bucket] = "machine_corpus.drop_buckets_by_generator"
    return excluded


def compliance_for(generator: str, bucket: str, flat: dict[str, float],
                   by_generator: dict[str, dict[str, float]]) -> float:
    """The compliance ratio for one generator in one bucket.

    ``by_generator[generator][bucket]`` when that generator names that bucket,
    else the bucket's entry in ``flat``, else 1.0 (ask for the human length and
    do not compensate). A generator with an entry for *some* buckets still
    falls back per bucket, so overriding hi alone leaves en and cm on the
    shared defaults.
    """
    per_generator = by_generator.get(generator) or {}
    if bucket in per_generator:
        return float(per_generator[bucket])
    return float(flat.get(bucket, 1.0))


#: Buckets whose ask is the human passage's OWN length rather than its bin, with
#: passages of ``long_from_words`` or more asked for ``human / long_ratio``.
#: cm needs both. Its bins are coarse where its passages are short: 59 % of cm
#: sits in the 0-40 bin, where "25" is 1.25x a 20-word passage before the model
#: writes a word. And one ask ratio does not fit both ends: at ask = human the
#: probe read 0-40 at 1.13 and 40+ at 0.73, so only the long bin is inflated
#: (probed 2026-09-21: 1.19 and 1.09). Only ever inflates, like compensated_words.
ASK_FROM_HUMAN: dict[str, dict[str, float]] = {"cm": {"long_from_words": 40, "long_ratio": 0.75}}


def ask_words_for(prompt: dict[str, Any], fertility: float, ratio: float, cap: int,
                  rule: dict[str, float] | None = None) -> int:
    """Words to ask for one passage.

    Without ``rule``: the passage's bin, compensated by the bucket's compliance
    ``ratio`` — every bucket's behaviour before 2026-09-21, unchanged. With an
    :data:`ASK_FROM_HUMAN` rule: the human length, divided by ``long_ratio`` for
    passages of ``long_from_words`` or more; the compliance ratio is not applied.
    """
    if rule is None:
        return compensated_words(prompt["n_words"], fertility, ratio, cap)
    human = int(prompt["human_length_words"])
    long_ratio = float(rule["long_ratio"]) if human >= int(rule["long_from_words"]) else 1.0
    return compensated_words(human, fertility, long_ratio, cap)


def compensated_words(n_words: int, fertility: float, ratio: float,
                      cap: int = 2048, floor_ratio: float = 0.25) -> int:
    """Words to ASK for so the model actually writes about ``n_words``.

    ``n_words / ratio``, clamped to ``cap / fertility``: asking for more words
    than the token cap can hold only guarantees a truncated passage, which Part
    13 drops. Never asks for fewer words than the human passage has.

    This only ever *inflates* the ask, so a ratio above 1 has no effect. A bucket
    that over-produces is therefore corrected by choosing a template it
    under-produces under, not by a ratio — see hi, decisions.md 2026-09-18.
    """
    ratio = min(1.0, max(floor_ratio, ratio))
    return max(n_words, min(int(round(n_words / ratio)), int(cap / fertility)))


# --- prompts and output -----------------------------------------------------

def load_prompts(human_dir: Path, buckets: Sequence[str], fraction: float,
                 seed: int, generator: str, limit: int | None = None,
                 keep: Callable[[str], bool] | None = None) -> list[dict[str, Any]]:
    """One prompt per human passage assigned to ``generator``, across the requested buckets.

    ``fraction`` samples deterministically from each bucket before assignment:
    the same fraction and seed always select the same passages, so a
    partly-generated corpus can be extended without reshuffling what is done.
    """
    slug = generator_slug(generator)
    prompts: list[dict[str, Any]] = []
    for bucket in buckets:
        path = human_dir / f"{bucket}.jsonl"
        if not path.exists():
            log.warning("no human corpus for bucket '%s' at %s — skipping", bucket, path)
            continue
        rows = read_jsonl(path, skip_bad_lines=True)
        rows.sort(key=lambda r: r["id"])  # stable order regardless of file order
        if fraction < 1.0:
            keep_n = max(1, int(round(len(rows) * fraction)))
            rows = random.Random(seed).sample(rows, keep_n)
            rows.sort(key=lambda r: r["id"])
        n_human = len(rows)
        if keep is not None:
            rows = [row for row in rows if keep(row["id"])]
        log.info("bucket %s: %d human passages, %d assigned to %s", bucket, n_human, len(rows), generator)
        for row in rows:
            n_words = length_bin(int(row["length_words"]))
            prompts.append({
                "id": f"{slug}__{row['id']}",
                "prompt_id": row["id"],
                "bucket": bucket,
                "n_words": n_words,
                "text": row["text"],        # kept so main() can rebuild the prompt once it knows fertility
                "prompt": build_prompt(row["text"], bucket, n_words),
                "domain": row.get("domain"),
                "human_length_words": int(row["length_words"]),
            })
    if limit is not None:
        prompts = prompts[:limit]
    return prompts


def check_output_compatible(out_path: Path, backend: str, model: str) -> None:
    """Refuse to append to a file that another backend or model wrote (it would mix quantisations)."""
    if not out_path.exists():
        return
    with open(out_path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            found = (row.get("backend"), row.get("generator_model"))
            if found != (backend, model):
                raise SystemExit(
                    f"{out_path} was written by backend={found[0]} model={found[1]}, but this run is "
                    f"backend={backend} model={model}. One file must not mix quantisations: move the "
                    f"file aside, or rerun with the original backend and model.")
            return


def machine_row(item: dict[str, Any], text: str, *, generator: str, role: str, model: str,
                backend: str, gen_tokens: int, done_reason: str | None,
                decoding: dict[str, Any]) -> dict[str, Any]:
    """One output row in the corpus schema, plus generation provenance."""
    return {
        "id": item["id"],
        "text": text,
        "label": LABEL_MACHINE,
        "language": item["bucket"],
        "code_mix_ratio": 0.0,          # measured in Part 13, after cleaning
        "generator": generator,
        "generator_role": role,
        "generator_model": model,
        "backend": backend,
        "domain": item["domain"],
        "length_words": len(text.split()),
        "attack_type": "clean",
        "writer_L1_band": None,
        "split": None,
        "prompt_id": item["prompt_id"],
        "prompt": item["prompt"],
        "requested_words": item["n_words"],
        "human_length_words": item["human_length_words"],
        "num_predict": item["num_predict"],
        "gen_tokens": gen_tokens,
        "done_reason": done_reason,
        "truncated": done_reason == "length",
        "decoding": decoding,
    }


class ThroughputMeter:
    """Thread-safe token counter that logs tokens/sec every ``log_every`` completed requests.

    With ``concurrent=True`` it also logs per-stream speed and the effective
    concurrency: summed per-request decode time over wall time in the window.
    About 4 means every server slot stayed busy; about 1 means the server is
    queueing requests.
    """

    def __init__(self, log_every: int = DEFAULT_BATCH_SIZE, concurrent: bool = True) -> None:
        self.log_every = max(1, log_every)
        self.concurrent = concurrent
        self.started = time.monotonic()
        self.tokens = 0
        self._lock = threading.Lock()
        self._reset_window(self.started)

    def _reset_window(self, now: float) -> None:
        self._window_start = now
        self._window_tokens = 0
        self._window_decode = 0.0
        self._window_requests = 0
        self._window_truncated = 0

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def record(self, tokens: int, decode_seconds: float, truncated: int, requests: int = 1) -> None:
        with self._lock:
            self.tokens += tokens
            self._window_tokens += tokens
            self._window_decode += decode_seconds
            self._window_requests += requests
            self._window_truncated += truncated
            if self._window_requests < self.log_every:
                return
            now = time.monotonic()
            wall = now - self._window_start
            extra = ""
            if self.concurrent and self._window_decode and wall:
                extra = (f", per-stream {self._window_tokens / self._window_decode:.1f} tok/s, "
                         f"concurrency x{self._window_decode / wall:.1f}")
            log.info("%d requests: %d tokens in %.1fs = %.1f tok/s%s, %d hit num_predict (run %.1f tok/s)",
                     self._window_requests, self._window_tokens, wall,
                     self._window_tokens / wall if wall else 0.0, extra, self._window_truncated,
                     self.tokens / max(now - self.started, 1e-6))
            self._reset_window(now)


# --- ollama backend ---------------------------------------------------------

class OllamaClient:
    """Minimal Ollama HTTP client on the standard library (no new dependency)."""

    def __init__(self, host: str, timeout_s: float = 900) -> None:
        self.host = host.rstrip("/")
        self.timeout_s = timeout_s

    def _request(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(self.host + path, data=data,
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise RuntimeError(f"ollama {path} HTTP {exc.code}: {detail}") from exc

    def version(self) -> str:
        return self._request("/api/version")["version"]

    def show(self, model: str) -> dict[str, Any]:
        return self._request("/api/show", {"model": model})

    def tags(self) -> list[dict[str, Any]]:
        return self._request("/api/tags").get("models", [])

    def load(self, model: str, num_ctx: int, keep_alive: str) -> None:
        """Load ``model`` with the run's context size; an empty prompt loads without generating."""
        self._request("/api/generate", {"model": model, "prompt": "", "stream": False,
                                        "options": {"num_ctx": num_ctx}, "keep_alive": keep_alive})

    def generate(self, model: str, prompt: str, options: dict[str, Any], keep_alive: str) -> dict[str, Any]:
        return self._request("/api/generate", {"model": model, "prompt": prompt, "stream": False,
                                               "options": options, "keep_alive": keep_alive})


def describe_ollama_model(client: OllamaClient, model: str) -> dict[str, Any]:
    """Quantisation and digest of a pulled model; exits with a pull hint if it is missing."""
    try:
        version = client.version()
    except (urllib.error.URLError, OSError) as exc:
        raise SystemExit(f"no Ollama server at {client.host} ({exc}). Start the Ollama app.") from exc
    try:
        details = client.show(model).get("details") or {}
    except RuntimeError as exc:
        raise SystemExit(f"{exc}\nmodel not available - run: ollama pull {model}") from exc
    digest = next((m.get("digest", "") for m in client.tags()
                   if model in (m.get("name"), m.get("model"))), "")
    return {"backend": "ollama", "server_version": version, "quantization": details.get("quantization_level"),
            "parameter_size": details.get("parameter_size"), "digest": digest[:12]}


#: Substrings Ollama/llama.cpp use when a load does not fit in VRAM.
_OOM_MARKERS: tuple[str, ...] = ("out of memory", "insufficient memory", "unable to allocate",
                                 "failed to allocate", "cudamalloc", "no available slots")


def load_with_fallback(client: OllamaClient, model: str, num_ctx: int, keep_alive: str,
                       parallel: int) -> int:
    """Load ``model``, halving ``parallel`` while the server reports out of memory.

    A 9B model at Q4_K_M plus KV cache for four slots at this ``num_ctx`` does not
    fit an 8 GB card — decisions.md 2026-09-13 section 2 flags ``gemma2:9b``
    specifically — so drop concurrency rather than ending a multi-hour run.

    Caveat: the server's own ``OLLAMA_NUM_PARALLEL`` decides how many slots it
    allocates. Lowering the client thread count reduces pressure but cannot
    shrink an allocation the server has already made; if this keeps firing, lower
    ``OLLAMA_NUM_PARALLEL`` or ``machine_corpus.ollama.num_ctx`` and restart it.
    """
    while True:
        try:
            client.load(model, num_ctx, keep_alive)
            return parallel
        except Exception as exc:  # noqa: BLE001 - the server returns HTTP 500 with a text body
            if parallel <= 1 or not any(m in str(exc).lower() for m in _OOM_MARKERS):
                raise
            parallel = max(1, parallel // 2)
            log.warning("load hit an out-of-memory error — retrying with parallel=%d (%s)",
                        parallel, str(exc)[:160])


def make_ollama_worker(client: OllamaClient, generator: str, role: str, model: str,
                       info: dict[str, Any], *, temperature: float, top_p: float,
                       num_ctx: int, keep_alive: str, seed: int, retries: int = 1,
                       retry_wait_s: float = 2.0, log_every: int = DEFAULT_BATCH_SIZE,
                       ) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Return a thread-safe ``process_item`` that generates one passage with one Ollama request.

    Built for :func:`src.utils.resumable.run_concurrent`, whose worker threads
    each call it independently. A request that still fails after ``retries``
    raises; the runner logs it and leaves the passage for the next resume.
    """
    meter = ThroughputMeter(log_every, concurrent=True)
    decoding = {"temperature": temperature, "top_p": top_p, "num_ctx": num_ctx,
                "quantization": info["quantization"], "digest": info["digest"]}
    warned = threading.Event()

    def process_item(item: dict[str, Any]) -> dict[str, Any]:
        item_seed = _hash64(f"{seed}:{item['id']}") % 2**31
        options = {"num_predict": item["num_predict"], "num_ctx": num_ctx,
                   "temperature": temperature, "top_p": top_p, "seed": item_seed}
        for attempt in range(retries + 1):
            try:
                response = client.generate(model, item["prompt"], options, keep_alive)
                break
            except Exception as exc:  # noqa: BLE001 - timeouts, dropped connections, server restarts
                if attempt == retries:
                    raise
                log.warning("%s: %s — retrying", item["id"], exc)
                time.sleep(retry_wait_s)
        prompt_tokens = int(response.get("prompt_eval_count") or 0)
        if prompt_tokens + item["num_predict"] > num_ctx and not warned.is_set():
            warned.set()
            log.warning("%s: prompt %d + num_predict %d exceeds num_ctx %d — raise machine_corpus.ollama.num_ctx",
                        item["id"], prompt_tokens, item["num_predict"], num_ctx)
        row = machine_row(item, (response.get("response") or "").strip(), generator=generator, role=role,
                          model=model, backend="ollama", gen_tokens=int(response.get("eval_count") or 0),
                          done_reason=response.get("done_reason"), decoding={**decoding, "seed": item_seed})
        meter.record(row["gen_tokens"], int(response.get("eval_duration") or 0) / 1e9, int(row["truncated"]))
        return row

    process_item.meter = meter  # type: ignore[attr-defined]
    return process_item


# --- hf backend (fallback) --------------------------------------------------

def make_hf_processor(model, tokenizer, generator: str, role: str, model_id: str, info: dict[str, Any], *,
                      temperature: float, top_p: float, max_prompt_tokens: int = 1024):
    """Return a ``process_batch`` that generates a batch locally with transformers."""
    import torch

    meter = ThroughputMeter(log_every=1, concurrent=False)   # one log line per batch

    def process_batch(batch: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        started = time.monotonic()
        texts = [item["prompt"] for item in batch]
        chat_template = getattr(tokenizer, "chat_template", None)
        if chat_template:
            # Instruct models need their chat template or they continue the
            # instruction text instead of answering it.
            texts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": t}], tokenize=False, add_generation_prompt=True
                )
                for t in texts
            ]
        encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                            max_length=max_prompt_tokens, add_special_tokens=not chat_template)
        encoded = {k: v.to(model.device) for k, v in encoded.items()}
        # A batch runs until its longest request is done, so budget by the largest.
        max_new = max(item["num_predict"] for item in batch)
        with torch.no_grad():
            output = model.generate(
                **encoded, max_new_tokens=max_new, do_sample=True,
                temperature=temperature, top_p=top_p,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )
        generated = output[:, encoded["input_ids"].shape[1]:]
        completions = tokenizer.batch_decode(generated, skip_special_tokens=True)
        decoding = {"temperature": temperature, "top_p": top_p, "max_new_tokens": max_new,
                    "quantization": info["dtype"]}

        rows: list[dict[str, Any]] = []
        for item, sequence, completion in zip(batch, generated, completions):
            n_tokens = int((sequence != tokenizer.pad_token_id).sum())
            rows.append(machine_row(item, completion.strip(), generator=generator, role=role, model=model_id,
                                    backend="hf", gen_tokens=n_tokens,
                                    done_reason="length" if n_tokens >= max_new else "stop",
                                    decoding=decoding))
        meter.record(sum(r["gen_tokens"] for r in rows), time.monotonic() - started,
                     sum(r["truncated"] for r in rows), requests=len(rows))
        return rows

    process_batch.meter = meter  # type: ignore[attr-defined]
    return process_batch


# --- length gate ------------------------------------------------------------

#: The gate: a bucket's first ``PROBE_ROWS`` rows must have a machine/human
#: length ratio inside ``GATE_RATIO_RANGE``, with at most ``GATE_MAX_CAP_PCT``
#: of them at the token cap, or the run aborts before the rest of that bucket
#: is generated. Overridable under ``machine_corpus.gate`` in configs/data.yaml.
#:
#: From 2026-09-18 this check was a line in decisions.md and an ad-hoc watcher
#: run by hand in a second terminal, which is precisely how cm reached 470 rows
#: at 1.88 of human length: a check that is a habit rather than code runs only
#: when someone remembers to run it, and its first version tested only the lower
#: bound and waved a 1.72 through. Both bounds now live here (decisions.md
#: 2026-09-19).
#:
#: The probe rows are drawn stratified across the bucket's length bins from
#: 2026-09-20: taking the head of an id-sorted list let cm's short-passage
#: defect sit outside the probe entirely (decisions.md 2026-09-20).
PROBE_ROWS = 24
GATE_RATIO_RANGE: tuple[float, float] = (0.75, 1.25)
GATE_MAX_CAP_PCT = 20.0

#: Code-mix check (2026-09-21). For these buckets the probe rows' median
#: romanised-Hindi function-word share must be at least ``GATE_CODEMIX_MIN_RATIO``
#: of the median for the same passages' human text. Length alone let 701 cm rows
#: through at 0.02-0.04 against a human 0.30-0.33: English in all but name, and
#: separable from the human bucket on that alone (docs/progress.md 2026-09-21).
GATE_CODEMIX_BUCKETS: tuple[str, ...] = ("cm",)
GATE_CODEMIX_MIN_RATIO = 2 / 3

#: Emoji check (2026-09-22). For these buckets a single probe row containing an
#: emoji fails the gate. The human cm bucket has emoji in 0 of 2,200 rows, so one
#: in a machine row is a class shortcut rather than style. The template says "No
#: emoji" and that is not enough on its own: two of the three probed variants
#: still put one in (docs/progress.md 2026-09-21), and 28 of the 701 stale cm
#: rows carry one.
GATE_EMOJI_BUCKETS: tuple[str, ...] = ("cm",)

#: Pictographs, dingbats, misc symbols/arrows, the watch/hourglass/media glyphs
#: and the emoji presentation selector. A lone ZWJ (U+200D) is deliberately
#: absent: it joins Devanagari conjuncts, and counting it scored 118 of the 2,200
#: human hi passages as emoji. Inside an emoji sequence the pictographs match
#: anyway.
_EMOJI_RE = re.compile("[\U0001F000-\U0001FAFF☀-➿⬀-⯿"
                       "⌚⌛⏩-⏺️]")


def has_emoji(text: str) -> bool:
    """True if ``text`` contains an emoji (see :data:`_EMOJI_RE` for what counts)."""
    return bool(_EMOJI_RE.search(text))


def _percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile of an already-sorted sequence."""
    if not values:
        return 0.0
    position = (len(values) - 1) * q
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


@dataclass
class GateStats:
    """Machine/human length statistics for one bucket's first rows.

    ``ratio`` is the **median** of the per-row machine/human word ratios, and the
    median is the gate's statistic deliberately. cm's failure was concentrated in
    the short passages — 21 human words answered with 84 machine words — so a
    length-weighted ratio of totals dilutes exactly the rows that are wrong,
    while a plain mean is dragged around by one runaway row. Measured on the 470
    discarded cm rows: median 1.98 against a mean of 1.88.
    """

    bucket: str
    n: int
    ratio: float = 0.0
    mean: float = 0.0
    p90: float = 0.0
    cap_pct: float = 0.0
    over_pct: float = 0.0
    #: Median romanised-Hindi share of the rows and of their own human passages;
    #: None when the bucket is not code-mix checked or no human text was paired.
    codemix: float | None = None
    human_codemix: float | None = None
    #: Rows containing an emoji; None when the bucket is not emoji checked.
    emoji_rows: int | None = None

    def describe(self) -> str:
        text = (f"{self.bucket}: {self.n} rows, machine/human length median {self.ratio:.2f} "
                f"(mean {self.mean:.2f}, p90 {self.p90:.2f}), {self.cap_pct:.0f} % at the token cap")
        if self.codemix is not None:
            text += f", Hindi share {self.codemix:.2f} vs human {self.human_codemix:.2f}"
        if self.emoji_rows is not None:
            text += f", {self.emoji_rows} with emoji"
        return text


def gate_stats(bucket: str, rows: Sequence[dict[str, Any]],
               ratio_max: float = GATE_RATIO_RANGE[1],
               human_texts: dict[str, str] | None = None,
               check_emoji: bool = False) -> GateStats:
    """Length statistics for ``rows``, which must already be one bucket's rows.

    With ``human_texts`` (row id -> the human passage it was prompted from), also
    the median romanised-Hindi share of the rows and of those same passages. The
    comparison is paired on purpose: the probe is 24 rows, so measuring it against
    the whole bucket's human share would mix a sample with a population.

    With ``check_emoji``, also the number of rows containing an emoji.

    **Every check reads the text after :func:`strip_boilerplate`**, including the
    length one, which recounts the words rather than trusting ``length_words``.
    Part 13 drops rows only after stripping, so a gate that judged raw text would
    abort a run over chatter that cleaning removes — which is exactly what gemma's
    cm probe did, failing on 2 emoji that were both inside "Let me know if you'd
    like me to continue! 😊" (decisions.md 2026-09-22). The stored row keeps the
    raw text; only the checks see the stripped version. Rows without ``text`` fall
    back to ``length_words``, so callers with bare rows are unaffected.
    """
    judged = [(row, strip_boilerplate(row.get("text") or "")) for row in rows]
    ratios: list[float] = []
    for row, text in judged:
        human = float(row.get("human_length_words") or 0)
        if human > 0:
            machine = float(len(text.split())) if text else float(row.get("length_words") or 0)
            ratios.append(machine / human)
    if not ratios:
        return GateStats(bucket, 0)
    ratios.sort()
    codemix = human_codemix = None
    if human_texts:
        paired = [(text, human_texts[row["id"]]) for row, text in judged
                  if row.get("id") in human_texts]
        if paired:
            codemix = _percentile(sorted(romanised_hindi_share(m) for m, _ in paired), 0.5)
            human_codemix = _percentile(sorted(romanised_hindi_share(h) for _, h in paired), 0.5)
    return GateStats(
        bucket=bucket,
        n=len(ratios),
        ratio=_percentile(ratios, 0.5),
        mean=sum(ratios) / len(ratios),
        p90=_percentile(ratios, 0.9),
        cap_pct=100.0 * sum(1 for row in rows if row.get("truncated")) / len(rows),
        over_pct=100.0 * sum(1 for r in ratios if r > ratio_max) / len(ratios),
        codemix=codemix,
        human_codemix=human_codemix,
        emoji_rows=sum(has_emoji(text) for _, text in judged) if check_emoji else None,
    )


def gate_reasons(stats: GateStats, ratio_range: tuple[float, float] = GATE_RATIO_RANGE,
                 max_cap_pct: float = GATE_MAX_CAP_PCT,
                 codemix_min_ratio: float = GATE_CODEMIX_MIN_RATIO) -> list[str]:
    """Why this bucket fails the gate; empty means it passes.

    Both bounds are tested. Failing only below ``ratio_range[0]`` is what let the
    cm run reach 470 rows: the watcher saw 1.72 and passed it. The code-mix check
    runs only when :func:`gate_stats` measured it and the human passages carry
    any Hindi at all — a zero human share gives nothing to be a fraction of. The
    emoji check, likewise only when measured, allows none.
    """
    low, high = ratio_range
    reasons: list[str] = []
    if stats.n == 0:
        return reasons
    if not low <= stats.ratio <= high:
        direction = "under" if stats.ratio < low else "over"
        reasons.append(f"machine/human length ratio {stats.ratio:.2f} is outside {low}-{high} "
                       f"({direction}-production; mean {stats.mean:.2f}, p90 {stats.p90:.2f}, "
                       f"{stats.over_pct:.0f} % of rows above {high})")
    if stats.cap_pct > max_cap_pct:
        reasons.append(f"{stats.cap_pct:.0f} % of rows hit the token cap, limit {max_cap_pct:.0f} % "
                       f"— they are truncated, not written short")
    if stats.codemix is not None and stats.human_codemix:
        floor = codemix_min_ratio * stats.human_codemix
        if stats.codemix < floor:
            reasons.append(f"code-mix: romanised-Hindi share median {stats.codemix:.2f} against "
                           f"{stats.human_codemix:.2f} for the same passages' human text, minimum "
                           f"{floor:.2f} ({codemix_min_ratio:.2f} of human) — the rows are not Hinglish")
    if stats.emoji_rows:
        reasons.append(f"emoji: {stats.emoji_rows} of {stats.n} rows contain emoji, limit 0 — the "
                       f"human '{stats.bucket}' passages carry none, so any is a class shortcut")
    return reasons


def gate_failure_message(stats: GateStats, reasons: Sequence[str], out_path: Path,
                         remaining: int) -> str:
    """The abort message: what failed, what it cost, and what to do about it."""
    bullets = "\n".join(f"  - {reason}" for reason in reasons)
    return (
        f"\nGATE FAILED — bucket '{stats.bucket}' after {stats.n} rows\n"
        f"{bullets}\n\n"
        f"Aborted before the remaining {remaining} '{stats.bucket}' passages were generated.\n"
        f"Those {stats.n} rows are still in {out_path} and are NOT usable — a resume reads them\n"
        f"back and fails this gate again until they are removed, which is deliberate.\n\n"
        f"Fix the bucket's prompt template or compliance ratio first (both are per-bucket and\n"
        f"measured on a 12-passage probe, never carried across by analogy — docs/decisions.md\n"
        f"2026-09-18), then drop the bucket's rows and re-run the same command:\n"
        f"  python -c \"import json,pathlib; p=pathlib.Path(r'{out_path}'); "
        f"p.write_text(''.join(l for l in p.read_text(encoding='utf-8').splitlines(keepends=True) "
        f"if l.strip() and json.loads(l)['language']!='{stats.bucket}'), encoding='utf-8')\"\n"
    )


def bucket_rows(out_path: Path, bucket: str, limit: int) -> list[dict[str, Any]]:
    """The first ``limit`` rows of ``bucket`` in an output file, in written order."""
    rows: list[dict[str, Any]] = []
    if not Path(out_path).exists():
        return rows
    with open(out_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("language") == bucket:
                rows.append(row)
                if len(rows) >= limit:
                    break
    return rows


def stratified_probe(items: Sequence[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    """``k`` of ``items``, drawn across their length bins instead of off the head.

    The gate judges a bucket's first rows, so which rows those are decides what
    it can see. Taking them from the head of an id-sorted list makes the probe
    inherit whatever length mix that ordering happens to front-load, and a
    defect that lives in one bin is then invisible in proportion to how badly
    the head misrepresents the bucket.

    Measured 2026-09-20 on the finished cm bucket: its first 24 rows had a
    median human length of 82 words against the bucket's 26, and 12.5 % of them
    in the 0-40 bin against 59 % across all 701. cm's over-production was
    entirely bin-local — 413 short rows at median 1.75 while every bin from 40
    words up sat inside 0.75-1.25 — so the probe read 1.02, passed, and the
    bucket settled at 1.39. A proportional draw puts ~14 short rows in the same
    24 and fails it at the probe, which is the whole point of the gate.

    Reuses :func:`stratified_cap`'s largest-remainder allocation, so the probe
    and the bucket caps stratify by the same rule and a resume draws the same
    passages. Items without ``n_words`` keep the previous head order, which is
    what the unit tests and any caller with bare prompts rely on.
    """
    if k <= 0:
        return []
    if k >= len(items):
        return list(items)
    if not all("n_words" in item for item in items):
        return list(items[:k])
    chosen = set(stratified_cap(items, k))
    return [item for item in items if item["id"] in chosen]


def run_with_gates(buckets: Sequence[str], prompts: Sequence[dict[str, Any]], *,
                   done_ids: set[str], out_path: Path,
                   run_phase: Callable[..., RunReport],
                   probe_rows: int = PROBE_ROWS,
                   ratio_range: tuple[float, float] = GATE_RATIO_RANGE,
                   max_cap_pct: float = GATE_MAX_CAP_PCT,
                   codemix_buckets: Sequence[str] = GATE_CODEMIX_BUCKETS,
                   codemix_min_ratio: float = GATE_CODEMIX_MIN_RATIO,
                   emoji_buckets: Sequence[str] = GATE_EMOJI_BUCKETS) -> RunReport:
    """Probe each bucket, gate it, then generate the rest. Aborts if a gate fails.

    Every bucket is taken to ``probe_rows`` rows and judged before the next one
    is touched, so a prompt regime that does not transfer costs 24 rows rather
    than a bucket. The probe is drawn across the bucket's length bins rather
    than off the head of the list (:func:`stratified_probe`), because a bin-local
    defect is otherwise invisible to it — that is how cm passed at 1.02 and
    settled at 1.39 over 701 rows (decisions.md 2026-09-20). Rows already on disk
    from an earlier sitting count towards the probe, so a resume judges what is
    there instead of generating a fresh 24 — and a bucket whose bad rows were
    never cleaned up keeps failing, by design.

    Buckets in ``codemix_buckets`` must also keep at least ``codemix_min_ratio``
    of their own human passages' romanised-Hindi share: length alone passed cm
    rows that were English in all but name (docs/progress.md 2026-09-21).
    Buckets in ``emoji_buckets`` fail on a single emoji row.

    The probe rows are ordinary corpus rows written to the same file, and the
    bulk phase receives the full prompt list with ``done_ids`` already updated,
    so a passing probe is never regenerated.

    Raises:
        SystemExit: a bucket's first rows failed the gate.
    """
    for bucket in buckets:
        in_bucket = [p for p in prompts if p["bucket"] == bucket]
        if not in_bucket:
            continue
        pending = [p for p in in_bucket if p["id"] not in done_ids]
        on_disk = len(in_bucket) - len(pending)
        shortfall = min(max(0, probe_rows - on_disk), len(pending))
        if shortfall:
            probe = stratified_probe(pending, shortfall)
            log.info("bucket %s: generating %d rows for the gate probe (%d already on disk), "
                     "stratified across %d length bins",
                     bucket, shortfall, on_disk, len({p.get("n_words") for p in probe}))
            phase = run_phase(probe, f"{bucket} gate-probe")
            if phase.stopped_by_time:
                log.warning("bucket %s: gate probe stopped at --max-minutes — not starting the bulk run",
                            bucket)
                return phase
        human_texts = ({p["id"]: p["text"] for p in in_bucket if p.get("text")}
                       if bucket in codemix_buckets else None)
        stats = gate_stats(bucket, bucket_rows(out_path, bucket, probe_rows), ratio_range[1],
                           human_texts, check_emoji=bucket in emoji_buckets)
        if stats.n == 0:
            log.warning("bucket %s: no rows to judge (every probe request failed) — "
                        "not starting the bulk run", bucket)
            return RunReport(total=len(prompts), already_done=len(prompts) - len(pending))
        reasons = gate_reasons(stats, ratio_range, max_cap_pct, codemix_min_ratio)
        if reasons:
            remaining = len([p for p in in_bucket if p["id"] not in done_ids])
            raise SystemExit(gate_failure_message(stats, reasons, out_path, remaining))
        log.info("gate passed — %s", stats.describe())
    return run_phase(prompts, "passages")


# --- CLI --------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generator", required=True,
                        help="alias from machine_corpus.generators: qwen7b, gemma, mistral, llama, phi")
    parser.add_argument("--backend", choices=BACKENDS, default=None, help="default: machine_corpus.backend")
    parser.add_argument("--model", default=None, help="override the alias's Ollama tag / HF id (fallbacks)")
    parser.add_argument("--buckets", default="en,hi,te,cm", help="comma-separated")
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="fraction of human passages considered, before assignment")
    parser.add_argument("--max-minutes", type=float, default=0, help="stop cleanly after this long (0 = no limit)")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--out-dir", default=None, help="default: machine_corpus.output_dir")
    parser.add_argument("--human-dir", default=None, help="default: human_corpus.output_dir")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="hf backend: prompts per batch")
    parser.add_argument("--parallel", type=int, default=None,
                        help="ollama backend: worker threads (default: machine_corpus.ollama.parallel)")
    parser.add_argument("--host", default=None, help="default: machine_corpus.ollama.host")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None, help="cap prompts (smoke tests)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="hf backend: cuda | cpu (default: auto)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    from src.utils.logging import configure_logging

    configure_logging(args.log_level)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    config = load_config(args.config)
    mc = config.get("machine_corpus") or {}
    generators = mc.get("generators") or {}
    if args.generator not in generators:
        parser.error(f"unknown generator '{args.generator}'; valid: {list(generators)}")
    spec = generators[args.generator]
    backend = args.backend or mc.get("backend", "ollama")
    model = args.model or spec[backend]
    decoding_cfg = mc.get("decoding") or {}
    temperature = args.temperature if args.temperature is not None else decoding_cfg.get("temperature", 0.8)
    top_p = args.top_p if args.top_p is not None else decoding_cfg.get("top_p", 0.95)

    human_dir = Path(args.human_dir or (config.get("human_corpus") or {}).get("output_dir", "data/raw/human"))
    buckets = [b.strip() for b in args.buckets.split(",") if b.strip()]
    unknown = [b for b in buckets if b not in LANGUAGE_BUCKETS]
    if unknown:
        parser.error(f"unknown bucket(s) {unknown}; valid: {list(LANGUAGE_BUCKETS)}")

    # Buckets this generator does not generate at all, from two independent
    # exclusions. Both name their source in the log, so a missing bucket is
    # always traceable to the line that removed it.
    #
    # By ROLE: held-out generators are test-only and exist to prove
    # generalisation (non-negotiable #4), which hi/te/cm already test, and en is
    # the best-covered bucket (decisions.md 2026-09-23). Keyed on the role so a
    # generator whose role changes carries the behaviour with it.
    #
    # By GENERATOR: a model whose tokenizer makes a bucket structurally
    # unrunnable. mistral reads Telugu at 13.309 tokens/word, so every te ask
    # pins at the 3600-token cap whatever the passage length — not a template
    # problem and not a ratio problem (decisions.md 2026-09-24).
    excluded = excluded_buckets(args.generator, spec.get("role", "seen"),
                                mc.get("heldout_drop_buckets") or (),
                                mc.get("drop_buckets_by_generator") or {})
    dropped = [b for b in buckets if b in excluded]
    if dropped:
        buckets = [b for b in buckets if b not in dropped]
        for bucket in dropped:
            log.info("generator %s: bucket %s excluded (%s)", args.generator, bucket, excluded[bucket])
        if not buckets:
            print(f"every requested bucket is excluded for generator '{args.generator}'")
            return

    keep = make_assignment(args.generator, generators, float((mc.get("assignment") or {}).get("heldout_share", 0.30)))
    prompts = load_prompts(human_dir, buckets, args.fraction, args.seed, args.generator, args.limit, keep)
    if not prompts:
        print("no prompts — is the human corpus built?")
        return

    # Per-bucket caps (te: Telugu decode cost, decisions.md 2026-09-18). The kept
    # ids are frozen to disk so every later run of this generator selects the same
    # passages, and so the selection is auditable rather than implicit.
    selection_dir = Path(mc.get("selection_dir", "data/processed/selection"))
    # Per-generator caps override the shared ones. They are a COMPUTE constraint,
    # not a statistical one: gemma and mistral decode ~4x slower than qwen on this
    # card, so a full hi/te assignment for each would cost days (decisions.md
    # 2026-09-23). A generator absent here keeps the shared cap.
    limits = {**(mc.get("bucket_limits") or {}),
              **((mc.get("bucket_limits_by_generator") or {}).get(args.generator) or {})}
    for bucket, raw_limit in limits.items():
        in_bucket = [p for p in prompts if p["bucket"] == bucket]
        limit = int(raw_limit)
        if not in_bucket or len(in_bucket) <= limit:
            continue
        record = selection_dir / f"{generator_slug(args.generator)}__{bucket}.json"
        keep_ids = load_or_record_selection(record, in_bucket, limit, bucket, args.generator)
        prompts = [p for p in prompts if p["bucket"] != bucket or p["id"] in keep_ids]
        log.info("bucket %s capped to %d of %d passages (%s)",
                 bucket, len(keep_ids), len(in_bucket), record)

    budget = mc.get("budget") or {}
    fertility, fertility_source = load_fertility(budget.get("fertility_csv", "results/tokenizer_fertility.csv"),
                                                 spec.get("tokenizer"),
                                                 budget.get("fallback_tokenizer", "Qwen/Qwen2.5-0.5B"), buckets)
    cap = int(budget.get("cap", 2048))
    ratios = {**COMPLIANCE_RATIO, **(budget.get("compliance_ratio") or {})}
    by_generator = {**COMPLIANCE_BY_GENERATOR, **(budget.get("compliance_ratio_by_generator") or {})}
    ask_rules = {**ASK_FROM_HUMAN, **(budget.get("ask_from_human") or {})}
    for prompt in prompts:
        bucket = prompt["bucket"]
        # Ask for more words than we want, because the model writes short (ratio < 1).
        prompt["ask_words"] = ask_words_for(prompt, fertility[bucket],
                                            compliance_for(args.generator, bucket, ratios, by_generator),
                                            cap, ask_rules.get(bucket))
        if prompt["ask_words"] != prompt["n_words"]:
            prompt["prompt"] = build_prompt(prompt["text"], bucket, prompt["ask_words"])
        # Budget the passage we WANT (the human length), not the inflated ask.
        prompt["num_predict"] = token_budget(prompt["human_length_words"], fertility[bucket],
                                             float(budget.get("overshoot", 1.3)),
                                             int(budget.get("pad_tokens", 64)), cap)
    for bucket in buckets:
        budgets = [p["num_predict"] for p in prompts if p["bucket"] == bucket]
        if budgets:
            log.info("budget %s: %.3f tokens/word (%s) -> num_predict %d..%d, %d of %d at the %d cap",
                     bucket, fertility[bucket], fertility_source, min(budgets), max(budgets),
                     sum(b >= cap for b in budgets), len(budgets), cap)
            # Say which compliance regime this bucket ran under and where it came
            # from. gemma inherited qwen's hi 0.55 for a whole run without one
            # line of the log naming it (2026-09-23).
            if bucket in ask_rules:
                source = f"ask_from_human {ask_rules[bucket]}"
            elif bucket in (by_generator.get(args.generator) or {}):
                source = f"{compliance_for(args.generator, bucket, ratios, by_generator):.2f} measured on {args.generator}"
            else:
                source = f"{compliance_for(args.generator, bucket, ratios, by_generator):.2f} shared default"
            log.info("compliance %s: %s", bucket, source)

    out_path = Path(args.out_dir or mc.get("output_dir", "data/raw/machine")) / f"{generator_slug(args.generator)}.jsonl"
    check_output_compatible(out_path, backend, model)
    done_ids = read_done_ids(out_path)
    if len(done_ids) >= len(prompts) and all(p["id"] in done_ids for p in prompts):
        print(f"\ncomplete — {len(prompts)} of {len(prompts)} (nothing to generate)\n{out_path}")
        return

    # One wall deadline across the gate probes and the bulk run, so --max-minutes
    # still means what it says now that a run is several phases.
    deadline = time.monotonic() + args.max_minutes * 60 if args.max_minutes and args.max_minutes > 0 else None

    def minutes_left() -> float | None:
        """Minutes of budget left, or None when there is no limit."""
        return None if deadline is None else (deadline - time.monotonic()) / 60

    if backend == "ollama":
        ollama_cfg = mc.get("ollama") or {}
        client = OllamaClient(args.host or ollama_cfg.get("host", "http://127.0.0.1:11434"),
                              float(ollama_cfg.get("request_timeout_s", 900)))
        info = describe_ollama_model(client, model)
        parallel = args.parallel or int(ollama_cfg.get("parallel", 4))
        num_ctx = int(ollama_cfg.get("num_ctx", 3072))
        keep_alive = str(ollama_cfg.get("keep_alive", "30m"))
        log.info("ollama %s: %s (%s, %s, digest %s), %d worker threads",
                 info["server_version"], model, info["parameter_size"], info["quantization"], info["digest"], parallel)
        load_started = time.monotonic()
        # Load up front so the first requests do not queue behind it, and so an
        # out-of-memory model is caught here rather than hours into a run.
        parallel = load_with_fallback(client, model, num_ctx, keep_alive, parallel)
        log.info("model loaded in %.1fs", time.monotonic() - load_started)
        process_item = make_ollama_worker(client, args.generator, spec["role"], model, info,
                                          temperature=temperature, top_p=top_p, num_ctx=num_ctx,
                                          keep_alive=keep_alive, seed=args.seed)
        meter = process_item.meter  # type: ignore[attr-defined]
        precision = info["quantization"]

        def run_phase(items: Sequence[dict[str, Any]], label: str = "passages") -> RunReport:
            left = minutes_left()
            if left is not None and left <= 0:
                return RunReport(total=len(items), stopped_by_time=True)
            return run_concurrent(items, id_of=lambda p: p["id"], process_item=process_item,
                                  out_path=out_path, done_ids=done_ids, workers=parallel,
                                  max_minutes=left or 0, label=label)
    else:
        from src.utils.modelload import load_causal_lm

        hf_model, tokenizer, info = load_causal_lm(model, device=args.device)
        process_batch = make_hf_processor(hf_model, tokenizer, args.generator, spec["role"], model, info,
                                          temperature=temperature, top_p=top_p)
        meter = process_batch.meter  # type: ignore[attr-defined]
        precision = f"{info['dtype']} on {info['device']}"

        def run_phase(items: Sequence[dict[str, Any]], label: str = "passages") -> RunReport:
            left = minutes_left()
            if left is not None and left <= 0:
                return RunReport(total=len(items), stopped_by_time=True)
            return run_resumable(items, id_of=lambda p: p["id"], process_batch=process_batch,
                                 out_path=out_path, done_ids=done_ids, batch_size=args.batch_size,
                                 max_minutes=left or 0, label=label)

    # Each bucket is judged on its first rows before the rest of it is generated:
    # a prompt regime that does not transfer costs 24 rows, not a bucket.
    gate_cfg = mc.get("gate") or {}
    report = run_with_gates(
        buckets, prompts, done_ids=done_ids, out_path=out_path, run_phase=run_phase,
        probe_rows=int(gate_cfg.get("probe_rows", PROBE_ROWS)),
        ratio_range=(float(gate_cfg.get("ratio_min", GATE_RATIO_RANGE[0])),
                     float(gate_cfg.get("ratio_max", GATE_RATIO_RANGE[1]))),
        max_cap_pct=float(gate_cfg.get("max_cap_pct", GATE_MAX_CAP_PCT)),
        codemix_buckets=tuple(gate_cfg.get("codemix_buckets", GATE_CODEMIX_BUCKETS)),
        codemix_min_ratio=float(gate_cfg.get("codemix_min_ratio", GATE_CODEMIX_MIN_RATIO)),
        emoji_buckets=tuple(gate_cfg.get("emoji_buckets", GATE_EMOJI_BUCKETS)))

    resume = f"python -m src.data.generate --generator {args.generator} --buckets {args.buckets}"
    if args.fraction != 1.0:
        resume += f" --fraction {args.fraction}"
    if args.backend:
        resume += f" --backend {backend}"
    if args.model:
        resume += f" --model {model}"
    print(f"\ngenerator={args.generator} ({spec['role']}: {model}, {precision}, {backend})  out={out_path}")
    print(report.summary(f"resume with: {resume}"))
    if meter.tokens:
        print(f"throughput: {meter.tokens:,} tokens in {meter.elapsed / 60:.1f} min = "
              f"{meter.tokens / meter.elapsed:.1f} tokens/sec")
    on_disk = len(read_done_ids(out_path) & {p["id"] for p in prompts})
    if on_disk != report.done:
        print(f"on disk: {on_disk} of {len(prompts)}")
    if report.errors:
        print(f"first error: {report.errors[0]}")


if __name__ == "__main__":
    main()
