"""Measure ONE generator's raw compliance ratio in ONE bucket, before generating it.

A compliance ratio is the fraction of the asked-for length that a model actually
writes: ``machine_words / ask_words``. :mod:`src.data.generate` inflates every
ask by ``1 / ratio`` so the model lands on the human length. The ratio is
therefore a property of the generator AND the bucket, and a value measured on
one model says nothing about another.

That is not how the values got there. ``compliance_ratio {hi 0.55, te 0.62}``
was measured on qwen and applied to gemma unchanged. gemma writes longer on
Indic, so its hi ask was inflated x1.8 on top of a model that did not need it
and its 24 gate-probe rows came out at 1.49 of human length, against a gate
bound of 1.25 (2026-09-23).

This script measures the raw ratio instead of assuming it:

  * the bucket's CURRENT template, unchanged — a ratio is only valid under the
    template it was measured on (decisions.md 2026-09-18, and te's 0.62-vs-0.31
    split is exactly this trap);
  * ``ask = the passage's length bin``, with NO compensation, so what comes back
    is the model's own behaviour and not a correction of someone else's;
  * ``k`` passages drawn stratified across the bucket's length bins, like the
    gate's probe, because a head-of-list sample hid cm's short-passage defect
    (decisions.md 2026-09-20);
  * generous headroom on ``num_predict`` (``--overshoot``, default 3.0) so a
    model that over-produces is measured rather than truncated. A censored row
    would drag the ratio down and re-create the very bug this measures.

Rows go to ``results/probes/`` and never to the corpus: a probe runs at the
wrong ask by construction, so its rows are not corpus rows. Words are counted
after :func:`strip_boilerplate`, the way the gate counts them.

Usage:
  python -m src.data.probe_compliance --generator gemma --buckets hi,te
  python -m src.data.probe_compliance --generator gemma --buckets hi --n 12
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from src.data.clean_artifacts import strip_boilerplate
from src.data.generate import (OllamaClient, build_prompt, compensated_words, describe_ollama_model,
                               generator_slug, load_fertility, load_with_fallback, load_prompts,
                               make_assignment, make_ollama_worker, stratified_probe, token_budget)
from src.data.schema import LANGUAGE_BUCKETS
from src.utils.config import load_config
from src.utils.io import read_jsonl
from src.utils.resumable import run_concurrent

log = logging.getLogger("probe_compliance")

#: Probe size. The gate uses 24; a ratio needs fewer rows than a pass/fail
#: verdict does, and 12 is what every earlier regime was measured on
#: (decisions.md 2026-09-18, 2026-09-21).
DEFAULT_PROBE_N = 12


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile of an already-sorted list."""
    if not values:
        return 0.0
    position = (len(values) - 1) * q
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def summarise(bucket: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Raw compliance statistics for one bucket's probe rows.

    ``ratio_to_ask`` is the compliance ratio to put in the config; the median is
    the statistic, matching the gate (a mean is dragged by one runaway row).
    ``ratio_to_human`` is what the gate would see if the bucket ran at this ask.
    """
    to_ask: list[float] = []
    to_human: list[float] = []
    truncated = 0
    for row in rows:
        words = len(strip_boilerplate(row.get("text") or "").split())
        ask = float(row.get("requested_words") or 0)
        human = float(row.get("human_length_words") or 0)
        if ask > 0:
            to_ask.append(words / ask)
        if human > 0:
            to_human.append(words / human)
        truncated += int(bool(row.get("truncated")))
    to_ask.sort()
    to_human.sort()
    return {
        "bucket": bucket,
        "n": len(rows),
        "ratio_to_ask": percentile(to_ask, 0.5),
        "mean_to_ask": sum(to_ask) / len(to_ask) if to_ask else 0.0,
        "p10_to_ask": percentile(to_ask, 0.1),
        "p90_to_ask": percentile(to_ask, 0.9),
        "ratio_to_human": percentile(to_human, 0.5),
        "truncated": truncated,
    }


def report(stats: dict[str, Any]) -> str:
    """One bucket's verdict, and what it means for the config."""
    ratio = stats["ratio_to_ask"]
    lines = [
        f"\n  {stats['bucket']}: {stats['n']} rows at ask = bin, no compensation",
        f"    machine/ask    median {ratio:.2f}  (mean {stats['mean_to_ask']:.2f}, "
        f"p10 {stats['p10_to_ask']:.2f}, p90 {stats['p90_to_ask']:.2f})",
        f"    machine/human  median {stats['ratio_to_human']:.2f}   <- what the gate would see",
    ]
    if stats["truncated"]:
        lines.append(f"    WARNING {stats['truncated']} of {stats['n']} rows hit num_predict — the ratio "
                     f"is a LOWER BOUND; re-run with a larger --overshoot")
    if ratio >= 1.0:
        lines.append(f"    -> writes AT OR ABOVE the ask. A ratio only ever inflates, so there is no")
        lines.append(f"       value that fixes this: set 1.0 (no compensation) and if machine/human is")
        lines.append(f"       still outside [0.75, 1.25], the TEMPLATE is what has to change.")
    else:
        lines.append(f"    -> set compliance_ratio_by_generator for this generator to {ratio:.2f}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generator", required=True, help="alias from machine_corpus.generators")
    parser.add_argument("--buckets", required=True, help="comma-separated")
    parser.add_argument("--n", type=int, default=DEFAULT_PROBE_N, help=f"passages per bucket (default {DEFAULT_PROBE_N})")
    parser.add_argument("--overshoot", type=float, default=3.0,
                        help="num_predict headroom; higher than the run's, so over-production is measured not clipped")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--out-dir", default="results/probes")
    parser.add_argument("--human-dir", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--parallel", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
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
    model = args.model or spec["ollama"]
    buckets = [b.strip() for b in args.buckets.split(",") if b.strip()]
    unknown = [b for b in buckets if b not in LANGUAGE_BUCKETS]
    if unknown:
        parser.error(f"unknown bucket(s) {unknown}; valid: {list(LANGUAGE_BUCKETS)}")

    human_dir = Path(args.human_dir or (config.get("human_corpus") or {}).get("output_dir", "data/raw/human"))
    keep = make_assignment(args.generator, generators,
                           float((mc.get("assignment") or {}).get("heldout_share", 0.30)))
    prompts = load_prompts(human_dir, buckets, 1.0, args.seed, args.generator, None, keep)
    if not prompts:
        print("no prompts — is the human corpus built?")
        return

    budget = mc.get("budget") or {}
    fertility, fertility_source = load_fertility(budget.get("fertility_csv", "results/tokenizer_fertility.csv"),
                                                 spec.get("tokenizer"),
                                                 budget.get("fallback_tokenizer", "Qwen/Qwen2.5-0.5B"), buckets)
    cap = int(budget.get("cap", 2048))

    selected: list[dict[str, Any]] = []
    for bucket in buckets:
        in_bucket = [p for p in prompts if p["bucket"] == bucket]
        chosen = stratified_probe(in_bucket, args.n)
        for prompt in chosen:
            # The ask is the bin at ratio 1.0: compensated_words never shrinks, so
            # this is the bin itself. Going through it keeps the cap clamp.
            prompt["ask_words"] = compensated_words(prompt["n_words"], fertility[bucket], 1.0, cap)
            prompt["prompt"] = build_prompt(prompt["text"], bucket, prompt["ask_words"])
            # Headroom, not the run's overshoot: a clipped row understates the ratio.
            prompt["num_predict"] = token_budget(prompt["ask_words"], fertility[bucket], args.overshoot,
                                                 int(budget.get("pad_tokens", 64)), cap)
        log.info("probe %s: %d of %d passages, ask = bin (no compensation), %.3f tokens/word (%s), "
                 "num_predict %d..%d", bucket, len(chosen), len(in_bucket), fertility[bucket], fertility_source,
                 min((p["num_predict"] for p in chosen), default=0),
                 max((p["num_predict"] for p in chosen), default=0))
        selected.extend(chosen)

    ollama_cfg = mc.get("ollama") or {}
    client = OllamaClient(args.host or ollama_cfg.get("host", "http://127.0.0.1:11434"),
                          float(ollama_cfg.get("request_timeout_s", 900)))
    info = describe_ollama_model(client, model)
    parallel = args.parallel or int(ollama_cfg.get("parallel", 4))
    num_ctx = int(ollama_cfg.get("num_ctx", 3072))
    keep_alive = str(ollama_cfg.get("keep_alive", "30m"))
    decoding_cfg = mc.get("decoding") or {}
    log.info("ollama %s: %s (%s, %s), %d worker threads", info["server_version"], model,
             info["parameter_size"], info["quantization"], parallel)
    parallel = load_with_fallback(client, model, num_ctx, keep_alive, parallel)

    process_item = make_ollama_worker(
        client, args.generator, spec["role"], model, info,
        temperature=args.temperature if args.temperature is not None else decoding_cfg.get("temperature", 0.8),
        top_p=args.top_p if args.top_p is not None else decoding_cfg.get("top_p", 0.95),
        num_ctx=num_ctx, keep_alive=keep_alive, seed=args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{generator_slug(args.generator)}__{'-'.join(buckets)}__askbin.jsonl"
    if out_path.exists():
        out_path.unlink()   # a probe is a fresh measurement, never a resume

    started = time.monotonic()
    run_concurrent(selected, id_of=lambda item: item["id"], process_item=process_item,
                   out_path=out_path, done_ids=set(), workers=parallel, label="probe rows")
    rows = read_jsonl(out_path, skip_bad_lines=True)
    log.info("%d rows in %.1f min -> %s", len(rows), (time.monotonic() - started) / 60, out_path)

    print(f"\nRAW COMPLIANCE — {args.generator} ({model}), ask = bin, current templates")
    summaries = []
    for bucket in buckets:
        in_bucket = [r for r in rows if r.get("language") == bucket]
        if not in_bucket:
            print(f"\n  {bucket}: no rows")
            continue
        stats = summarise(bucket, in_bucket)
        summaries.append(stats)
        print(report(stats))
    if summaries:
        print("\n  configs/data.yaml -> machine_corpus.budget.compliance_ratio_by_generator:")
        print(f"    {args.generator}: {{"
              + ", ".join(f"{s['bucket']}: {min(1.0, s['ratio_to_ask']):.2f}" for s in summaries) + "}")
    print(f"\n  rows: {out_path}")


if __name__ == "__main__":
    main()
