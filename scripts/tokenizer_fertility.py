"""Measure tokenizer fertility per language bucket and lock the Head B scorer.

Fertility = tokens emitted per whitespace word. A tokenizer trained mostly on
English shatters Devanagari and Telugu into many more pieces per word, which
inflates sequence length, wastes the context window, and - the reason this
matters here - changes the per-token log-probabilities that Fast-DetectGPT
curvature is computed from. Reporting fertility alongside per-bucket AUROC is
what turns "curvature is weaker on Telugu" into an explanation rather than an
observation (F2 in docs/project-context-master.md §7).

Tokenizers only: nothing is downloaded beyond the tokenizer files, and no model
weights are loaded, so this runs on CPU in seconds. Gated repos are skipped with
a note rather than failing the run.

Run::

    python scripts/tokenizer_fertility.py [--config configs/models.yaml]
        [--n 200] [--out results/tokenizer_fertility.csv]

Writes ``results/tokenizer_fertility.csv`` with one row per (model, bucket).
Serves docs/parts-plan.md Part 4 and docs/master-execution-plan.md Phase 2.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# Tokenizer-only: gemma-2-2b and Llama-3.1-8B are gated and skipped without a
# HF token, which is expected and not an error.
DEFAULT_MODELS: tuple[str, ...] = (
    "ai-forever/mGPT",
    "Qwen/Qwen2.5-0.5B",
    "google/gemma-2-2b",
    "meta-llama/Llama-3.1-8B",
)
BUCKETS: tuple[str, ...] = ("en", "hi", "te", "cm")


def load_passages(human_dir: Path, bucket: str, n: int, seed: int) -> list[str]:
    """Return up to ``n`` human passages for one bucket, sampled deterministically."""
    path = human_dir / f"{bucket}.jsonl"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        texts = [json.loads(line)["text"] for line in fh if line.strip()]
    random.Random(seed).shuffle(texts)
    return texts[:n]


def measure(tokenizer, texts: list[str]) -> dict[str, float]:
    """Mean tokens per word and per character over ``texts``.

    Ratios are computed over the pooled totals, not as a mean of per-text
    ratios, so long and short passages contribute in proportion to their size.
    """
    total_tokens = total_words = total_chars = 0
    for text in texts:
        total_tokens += len(tokenizer(text, add_special_tokens=False)["input_ids"])
        total_words += len(text.split())
        total_chars += len(text)
    if not total_words:
        return {"tokens_per_word": float("nan"), "tokens_per_char": float("nan"),
                "n_texts": 0, "n_words": 0, "n_tokens": 0}
    return {
        "tokens_per_word": total_tokens / total_words,
        "tokens_per_char": total_tokens / total_chars,
        "n_texts": len(texts),
        "n_words": total_words,
        "n_tokens": total_tokens,
    }


def run(models: list[str], human_dir: Path, out_path: Path, n: int, seed: int) -> list[dict]:
    from transformers import AutoTokenizer

    passages = {b: load_passages(human_dir, b, n, seed) for b in BUCKETS}
    for bucket, texts in passages.items():
        if not texts:
            print(f"  WARNING: no passages for bucket '{bucket}' in {human_dir}")
        else:
            print(f"  {bucket}: {len(texts)} passages, {sum(len(t.split()) for t in texts):,} words")

    rows: list[dict] = []
    for model in models:
        print(f"\n[{model}]")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model)
        except Exception as exc:  # noqa: BLE001 - gated/offline repos must not fail the run
            reason = type(exc).__name__
            gated = any(k in str(exc).lower() for k in ("gated", "awaiting", "401", "403", "authorized"))
            print(f"  SKIPPED ({'gated - accept the licence and huggingface-cli login' if gated else reason})")
            rows.append({"model": model, "bucket": "-", "tokens_per_word": "", "tokens_per_char": "",
                         "n_texts": 0, "n_words": 0, "n_tokens": 0, "vocab_size": "",
                         "status": "gated" if gated else f"error:{reason}"})
            continue
        vocab = getattr(tokenizer, "vocab_size", "")
        for bucket in BUCKETS:
            if not passages[bucket]:
                continue
            stats = measure(tokenizer, passages[bucket])
            rows.append({"model": model, "bucket": bucket, **stats, "vocab_size": vocab, "status": "ok"})
            print(f"  {bucket}: {stats['tokens_per_word']:.3f} tokens/word, "
                  f"{stats['tokens_per_char']:.3f} tokens/char")
    write_csv(rows, out_path)
    return rows


def write_csv(rows: list[dict], out_path: Path) -> None:
    import csv

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "bucket", "tokens_per_word", "tokens_per_char",
              "n_texts", "n_words", "n_tokens", "vocab_size", "status"]
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})
    print(f"\nwrote {out_path}")


def print_table(rows: list[dict]) -> None:
    """Print tokens/word as a model x bucket grid, with each bucket's ratio to en."""
    ok = [r for r in rows if r.get("status") == "ok"]
    if not ok:
        print("no successful measurements")
        return
    models = list(dict.fromkeys(r["model"] for r in ok))
    print(f"\ntokens per word (lower is better; 'xEN' = this bucket / en)\n"
          f"{'model':<28}" + "".join(f"{b:>18}" for b in BUCKETS))
    for model in models:
        by_bucket = {r["bucket"]: r for r in ok if r["model"] == model}
        en = by_bucket.get("en", {}).get("tokens_per_word")
        cells = ""
        for bucket in BUCKETS:
            row = by_bucket.get(bucket)
            if not row:
                cells += f"{'-':>18}"
                continue
            tpw = row["tokens_per_word"]
            ratio = f" x{tpw / en:.2f}" if en else ""
            cells += f"{tpw:>12.3f}{ratio:>6}"
        print(f"{model:<28}{cells}")
    skipped = [r for r in rows if r.get("status") != "ok"]
    for row in skipped:
        print(f"  skipped: {row['model']} ({row['status']})")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/models.yaml")
    parser.add_argument("--human-dir", default="data/raw/human")
    parser.add_argument("--out", default="results/tokenizer_fertility.csv")
    parser.add_argument("--n", type=int, default=200, help="passages sampled per bucket")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", nargs="*", default=None, help="override the model list")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    models = args.models if args.models else list(DEFAULT_MODELS)
    print(f"fertility on {args.n} passages/bucket from {args.human_dir}")
    rows = run(models, Path(args.human_dir), Path(args.out), args.n, args.seed)
    print_table(rows)


if __name__ == "__main__":
    main()
