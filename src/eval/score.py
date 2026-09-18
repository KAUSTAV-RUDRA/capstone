"""Score every corpus row with one method, into one column of results/scores.parquet.

Usage::

    python -m src.eval.score --column headB --max-minutes 110
        [--only-attacked] [--config configs/models.yaml] [--limit N]

Every text is scored once by every method into a single wide table, and every
results table in the paper is then a pandas operation over that file. That is
what makes the two-week plan fit: the expensive work happens once, and T1-T6 are
cheap re-reads (docs/sprint-plan.md).

**Convention: higher = more machine-like, for every column.** Baselines that
are naturally the other way round (perplexity) are negated inside their scorer,
so AUROC is computed identically for all of them.

Columns are a registry. Each is a function ``(texts, rows, ctx) -> list[float]``
registered under a name; adding a method is one decorated function, with no
change to the CLI. Progress is journalled to
``results/.score_<column>.jsonl`` and folded into the parquet at the end, so a
killed run resumes from the last completed batch.

Serves docs/parts-plan.md Part 4(b) and Stage 4.
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.utils.config import load_config
from src.utils.io import read_jsonl
from src.utils.resumable import DEFAULT_BATCH_SIZE, read_done_ids, run_resumable

log = logging.getLogger("score")

DEFAULT_CORPUS = "data/processed/corpus.jsonl"
DEFAULT_ATTACKED = "data/processed/attacked.jsonl"
DEFAULT_SCORES = "results/scores.parquet"


@dataclass
class ScoreContext:
    """What a column function may need beyond the texts themselves."""

    config: dict[str, Any] = field(default_factory=dict)
    device: str | None = None
    models_config: dict[str, Any] = field(default_factory=dict)
    cache: dict[str, Any] = field(default_factory=dict)   # loaded models, reused across batches


ColumnFn = Callable[[Sequence[str], Sequence[dict[str, Any]], ScoreContext], Sequence[float]]

#: name -> scoring function. Populated by @register.
COLUMN_REGISTRY: dict[str, ColumnFn] = {}
#: name -> one-line description, for --list.
COLUMN_DOCS: dict[str, str] = {}


def register(name: str, doc: str) -> Callable[[ColumnFn], ColumnFn]:
    """Register a column scorer under ``name``."""

    def decorator(fn: ColumnFn) -> ColumnFn:
        if name in COLUMN_REGISTRY:
            raise KeyError(f"column '{name}' is already registered")
        COLUMN_REGISTRY[name] = fn
        COLUMN_DOCS[name] = doc
        return fn

    return decorator


# ---------------------------------------------------------------------------
# The eight columns. All are stubs until their parts; each raises with the part
# that fills it in, so running one early fails with a pointer rather than a
# silent column of zeros.
# ---------------------------------------------------------------------------
@register("headA", "Head A: stylometric features -> per-bucket logistic regression (Part 14)")
def score_headA(texts, rows, ctx):
    raise NotImplementedError(
        "headA lands in parts-plan Part 14: extend src/features/stylometric.py with real "
        "POS n-grams and dependency depth (stanza for hi/te, spaCy for en/cm), fit a "
        "per-bucket logistic regression on the train split, score all rows."
    )


@register("headB", "Head B: Fast-DetectGPT curvature, scorer ai-forever/mGPT (Part 15)")
def score_headB(texts, rows, ctx):
    raise NotImplementedError(
        "headB lands in parts-plan Part 15: Fast-DetectGPT analytic curvature with "
        "ai-forever/mGPT (LOCKED 2026-09-13 on tokenizer fertility) fp16 on CUDA. "
        "The mechanism already exists in src/baselines/fast_detectgpt.py."
    )


@register("headC", "Head C: MuRIL mean-pooled embeddings -> per-bucket logistic head (Part 16)")
def score_headC(texts, rows, ctx):
    raise NotImplementedError(
        "headC lands in parts-plan Part 16: google/muril-base-cased mean-pooled embeddings "
        "into a per-bucket logistic head. MuRIL is encoder-only — embeddings ONLY, never "
        "curvature (non-negotiable #9)."
    )


@register("ppl", "Baseline: negated mean log-perplexity, Qwen2.5-0.5B (Part 17)")
def score_ppl(texts, rows, ctx):
    raise NotImplementedError(
        "ppl lands in parts-plan Part 17. Negate it so that, like every other column, "
        "higher means more machine-like."
    )


@register("fastdetectgpt_en", "Baseline: Fast-DetectGPT with the English-default scorer (Part 17)")
def score_fastdetectgpt_en(texts, rows, ctx):
    raise NotImplementedError(
        "fastdetectgpt_en lands in parts-plan Part 17: Fast-DetectGPT with Qwen2.5-0.5B. "
        "Its gap to headB on hi/te versus en is the tokenizer-fragmentation result (F2)."
    )


@register("binoculars", "Baseline: Binoculars, observer Qwen2.5-0.5B / performer -0.5B-Instruct (Part 17)")
def score_binoculars(texts, rows, ctx):
    raise NotImplementedError("binoculars lands in parts-plan Part 17.")


@register("xlmr", "Baseline: fine-tuned xlm-roberta-base, supervised (Part 18)")
def score_xlmr(texts, rows, ctx):
    raise NotImplementedError(
        "xlmr lands in parts-plan Part 18: fine-tune xlm-roberta-base for 2 epochs on the "
        "train split, max_len 256, batch 16, fp16; checkpoint to results/models/xlmr/."
    )


# NOTE: the "detectgpt" column was dropped on 2026-09-18 (decisions.md). Vanilla
# DetectGPT is superseded by Fast-DetectGPT, which is retained above with an
# identical statistic; running both cost two GPU sittings for no extra evidence.
# The paper still cites Mitchell et al. 2023 as the origin of the curvature
# hypothesis. src/baselines/detectgpt.py is left in place but unregistered.


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def load_rows(corpus_path: Path, attacked_path: Path, only_attacked: bool) -> list[dict[str, Any]]:
    """Load corpus rows, plus attacked rows when that file exists."""
    rows: list[dict[str, Any]] = []
    if not only_attacked:
        if not corpus_path.exists():
            raise FileNotFoundError(
                f"{corpus_path} does not exist. It is written by src/data/freeze_splits.py "
                f"(parts-plan Part 13); until then there is nothing to score."
            )
        rows.extend(read_jsonl(corpus_path, skip_bad_lines=True))
    if attacked_path.exists():
        attacked = read_jsonl(attacked_path, skip_bad_lines=True)
        log.info("including %d attacked rows from %s", len(attacked), attacked_path)
        rows.extend(attacked)
    elif only_attacked:
        raise FileNotFoundError(f"--only-attacked given but {attacked_path} does not exist")
    return rows


def journal_path(scores_path: Path, column: str) -> Path:
    """Where partial results for one column are journalled between batches."""
    return scores_path.parent / f".score_{column}.jsonl"


def merge_into_parquet(scores_path: Path, column: str, journal: Path) -> int:
    """Fold the journal into ``scores.parquet`` as one column keyed by row id."""
    import pandas as pd

    entries = read_jsonl(journal, skip_bad_lines=True)
    if not entries:
        return 0
    scored = pd.DataFrame(entries).drop_duplicates(subset="id", keep="last").set_index("id")
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    if scores_path.exists():
        table = pd.read_parquet(scores_path).set_index("id")
        table[column] = scored["score"]           # aligns on id; missing rows stay NaN
    else:
        table = scored[["score"]].rename(columns={"score": column})
    table.reset_index().to_parquet(scores_path, index=False)
    return len(scored)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--column", help=f"one of: {', '.join(COLUMN_REGISTRY)}")
    parser.add_argument("--list", action="store_true", help="list registered columns and exit")
    parser.add_argument("--max-minutes", type=float, default=0)
    parser.add_argument("--only-attacked", action="store_true", help="score only data/processed/attacked.jsonl")
    parser.add_argument("--config", default="configs/models.yaml")
    parser.add_argument("--corpus", default=DEFAULT_CORPUS)
    parser.add_argument("--attacked", default=DEFAULT_ATTACKED)
    parser.add_argument("--scores", default=DEFAULT_SCORES)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    from src.utils.logging import configure_logging

    configure_logging(args.log_level)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if args.list or not args.column:
        print(f"{'column':<20}description")
        for name, doc in COLUMN_DOCS.items():
            print(f"{name:<20}{doc}")
        if not args.column and not args.list:
            parser.error("--column is required")
        return
    if args.column not in COLUMN_REGISTRY:
        parser.error(f"unknown column '{args.column}'; registered: {', '.join(COLUMN_REGISTRY)}")

    rows = load_rows(Path(args.corpus), Path(args.attacked), args.only_attacked)
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        print("no rows to score")
        return

    scores_path = Path(args.scores)
    journal = journal_path(scores_path, args.column)
    ctx = ScoreContext(config=load_config(args.config), device=args.device,
                       models_config=load_config(args.config))
    column_fn = COLUMN_REGISTRY[args.column]

    def process_batch(batch: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        values = column_fn([r["text"] for r in batch], batch, ctx)
        if len(values) != len(batch):
            raise ValueError(f"column '{args.column}' returned {len(values)} scores for {len(batch)} rows")
        return [{"id": r["id"], "score": float(v)} for r, v in zip(batch, values)]

    report = run_resumable(
        rows, id_of=lambda r: r["id"], process_batch=process_batch, out_path=journal,
        done_ids=read_done_ids(journal), batch_size=args.batch_size,
        max_minutes=args.max_minutes, label="rows",
    )
    merged = merge_into_parquet(scores_path, args.column, journal)
    print(f"\ncolumn={args.column}  rows={len(rows)}  merged={merged}  out={scores_path}")
    print(report.summary(f"resume with: python -m src.eval.score --column {args.column}"))
    if report.errors:
        print(f"first error: {report.errors[0]}")


if __name__ == "__main__":
    main()
