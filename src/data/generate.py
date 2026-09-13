"""Generate prompt-matched machine text for one generator (resumable CLI).

Usage::

    python -m src.data.generate --generator Qwen/Qwen2.5-7B-Instruct
        --buckets en,hi,te,cm --fraction 1.0 --max-minutes 110
        [--config configs/data.yaml] [--limit N] [--seed 42]

Every machine passage is matched to one human passage: the prompt is that
passage's **first sentence**, and the requested length is its **length bin**, so
human and machine text share topic and length distribution by construction.
Without that, a detector can separate the classes on topic or length alone and
the reported AUROC means nothing (docs/project-context-master.md §6).

Output is ``data/raw/machine/<generator-slug>.jsonl``, one row per human
passage, carrying ``prompt_id`` back to it. Rows are appended after every batch
and ids already present are skipped, so re-running the same command resumes.

Serves docs/parts-plan.md Part 4(a) and Stage 3.
"""
from __future__ import annotations

import argparse
import logging
import random
import re
import sys
from pathlib import Path
from typing import Any, Sequence

from src.data.schema import LABEL_MACHINE, LANGUAGE_BUCKETS
from src.utils.config import load_config
from src.utils.io import read_jsonl
from src.utils.resumable import DEFAULT_BATCH_SIZE, read_done_ids, run_resumable

log = logging.getLogger("generate")

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
#: cm bucket (Romanised, informal).
PROMPT_TEMPLATES: dict[str, str] = {
    "default": ('{first_sentence}\n\nWrite about {n_words} words in {language} '
                'on this topic, continuing naturally.'),
    "cm": ('{first_sentence}\n\nWrite about {n_words} words on this topic in casual '
           'Hindi-English Hinglish, Roman script, the way students text. '
           'Continue naturally.'),
}

_SENT_END_RE = re.compile(r"(?<=[.!?।॥])\s+")
_SLUG_RE = re.compile(r"[^A-Za-z0-9]+")


def generator_slug(generator: str) -> str:
    """Filesystem-safe name for a HF model id: ``Qwen/Qwen2.5-7B-Instruct`` -> ``qwen-qwen2-5-7b-instruct``."""
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


def build_prompt(text: str, bucket: str, n_words: int) -> str:
    """Prompt-matched instruction for one human passage."""
    template = PROMPT_TEMPLATES.get(bucket, PROMPT_TEMPLATES["default"])
    return template.format(
        first_sentence=first_sentence(text),
        n_words=n_words,
        language=LANGUAGE_NAMES.get(bucket, bucket),
    )


def load_prompts(human_dir: Path, buckets: Sequence[str], fraction: float,
                 seed: int, generator: str, limit: int | None = None) -> list[dict[str, Any]]:
    """One prompt per human passage, across the requested buckets.

    ``fraction`` samples deterministically from each bucket: the same fraction
    and seed always select the same passages, so a partly-generated corpus can
    be extended without reshuffling what is already done.
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
            keep = max(1, int(round(len(rows) * fraction)))
            rows = random.Random(seed).sample(rows, keep)
            rows.sort(key=lambda r: r["id"])
        log.info("bucket %s: %d human passages", bucket, len(rows))
        for row in rows:
            n_words = length_bin(int(row["length_words"]))
            prompts.append({
                "id": f"{slug}__{row['id']}",
                "prompt_id": row["id"],
                "bucket": bucket,
                "n_words": n_words,
                "prompt": build_prompt(row["text"], bucket, n_words),
                "domain": row.get("domain"),
                "human_length_words": int(row["length_words"]),
            })
    if limit is not None:
        prompts = prompts[:limit]
    return prompts


def make_batch_processor(model, tokenizer, generator: str, info: dict[str, Any],
                         temperature: float, top_p: float, max_prompt_tokens: int = 512):
    """Return a ``process_batch`` that generates continuations for a batch of prompts."""
    import torch

    def process_batch(batch: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
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
        # Budget tokens by the largest request in the batch, scaled for the
        # tokenizer's fertility on this script (Indic words cost 3-6 tokens).
        want_words = max(item["n_words"] for item in batch)
        max_new = min(1024, int(want_words * 4) + 64)
        with torch.no_grad():
            output = model.generate(
                **encoded, max_new_tokens=max_new, do_sample=True,
                temperature=temperature, top_p=top_p,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )
        prompt_len = encoded["input_ids"].shape[1]
        completions = tokenizer.batch_decode(output[:, prompt_len:], skip_special_tokens=True)

        rows: list[dict[str, Any]] = []
        for item, completion in zip(batch, completions):
            text = completion.strip()
            rows.append({
                "id": item["id"],
                "text": text,
                "label": LABEL_MACHINE,
                "language": item["bucket"],
                "code_mix_ratio": 0.0,          # measured in Part 13, after cleaning
                "generator": generator,
                "domain": item["domain"],
                "length_words": len(text.split()),
                "attack_type": "clean",
                "writer_L1_band": None,
                "split": None,
                "prompt_id": item["prompt_id"],
                "prompt": item["prompt"],
                "requested_words": item["n_words"],
                "human_length_words": item["human_length_words"],
                "decoding": {"temperature": temperature, "top_p": top_p, "dtype": info["dtype"]},
            })
        return rows

    return process_batch


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generator", required=True, help="HF model id")
    parser.add_argument("--buckets", default="en,hi,te,cm", help="comma-separated")
    parser.add_argument("--fraction", type=float, default=1.0, help="fraction of human passages to match")
    parser.add_argument("--max-minutes", type=float, default=0, help="stop cleanly after this long (0 = no limit)")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--out-dir", default="data/raw/machine")
    parser.add_argument("--human-dir", default=None, help="defaults to the config's human_corpus.output_dir")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--limit", type=int, default=None, help="cap prompts (smoke tests)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    from src.utils.logging import configure_logging

    configure_logging(args.log_level)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    config = load_config(args.config)
    human_dir = Path(args.human_dir or (config.get("human_corpus") or {}).get("output_dir", "data/raw/human"))
    buckets = [b.strip() for b in args.buckets.split(",") if b.strip()]
    unknown = [b for b in buckets if b not in LANGUAGE_BUCKETS]
    if unknown:
        parser.error(f"unknown bucket(s) {unknown}; valid: {list(LANGUAGE_BUCKETS)}")

    out_path = Path(args.out_dir) / f"{generator_slug(args.generator)}.jsonl"
    prompts = load_prompts(human_dir, buckets, args.fraction, args.seed, args.generator, args.limit)
    if not prompts:
        print("no prompts — is the human corpus built?")
        return

    done_ids = read_done_ids(out_path)
    if len(done_ids) >= len(prompts) and all(p["id"] in done_ids for p in prompts):
        print(f"\ncomplete — {len(prompts)} of {len(prompts)} (nothing to generate)\n{out_path}")
        return

    from src.utils.modelload import load_causal_lm

    model, tokenizer, info = load_causal_lm(args.generator, device=args.device)
    process_batch = make_batch_processor(model, tokenizer, args.generator, info,
                                         args.temperature, args.top_p)
    report = run_resumable(
        prompts, id_of=lambda p: p["id"], process_batch=process_batch, out_path=out_path,
        done_ids=done_ids, batch_size=args.batch_size, max_minutes=args.max_minutes,
        label="passages",
    )
    print(f"\ngenerator={args.generator} ({info['dtype']} on {info['device']})  out={out_path}")
    print(report.summary(f"resume with: python -m src.data.generate --generator {args.generator} "
                         f"--buckets {args.buckets} --fraction {args.fraction}"))
    if report.errors:
        print(f"first error: {report.errors[0]}")


if __name__ == "__main__":
    main()
