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
from src.utils.text import normalise_rows

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
    corpus_path: Path = Path(DEFAULT_CORPUS)               # supervised columns fit on its train split
    max_minutes: float = 0
    normalise: bool = False                                # collapse whitespace before any feature extraction


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
def stylometric_features(rows: Sequence[dict[str, Any]], cfg: dict[str, Any],
                         max_minutes: float = 0) -> dict[str, list[float]]:
    """Head A's 40 raw features for ``rows``, id -> vector, cached on disk.

    Parsing is the slow part (stanza ~0.5 s/passage on CPU for hi/te), so vectors
    are journalled next to ``features_cache`` as they are computed and a killed run
    resumes; the finished set is folded into the parquet. Rows are batched by
    bucket so each batch uses one parser.
    """
    import pandas as pd

    from src.features.stylometric import StylometricExtractor

    cache = Path(cfg["features_cache"])
    journal = cache.parent / f".{cache.stem}.jsonl"
    have: dict[str, list[float]] = {}
    if cache.exists():
        table = pd.read_parquet(cache)
        have = dict(zip(table["id"], table["features"].map(list)))
    missing = [r for r in rows if r["id"] not in have]
    if missing:
        extractor = StylometricExtractor(config=cfg, syntax=True)
        missing.sort(key=lambda r: (r["language"], r["id"]))

        def process(batch: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
            X = extractor.raw_features([r["text"] for r in batch], [r["language"] for r in batch])
            return [{"id": r["id"], "features": x.tolist()} for r, x in zip(batch, X)]

        report = run_resumable(missing, id_of=lambda r: r["id"], process_batch=process,
                               out_path=journal, batch_size=32, max_minutes=max_minutes,
                               label="rows (features)")
        if report.errors:
            raise RuntimeError(f"feature extraction failed: {report.errors[0]}")
        for entry in read_jsonl(journal, skip_bad_lines=True):
            have[entry["id"]] = entry["features"]
        still = [r["id"] for r in rows if r["id"] not in have]
        if still:
            raise RuntimeError(f"{len(still)} rows still lack features (time budget hit?); re-run to resume")
        cache.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"id": list(have), "features": list(have.values())}).to_parquet(cache, index=False)
        journal.unlink(missing_ok=True)
    return {r["id"]: have[r["id"]] for r in rows}


def _corpus_rows(ctx: ScoreContext) -> list[dict[str, Any]]:
    """The corpus as scoring sees it: whitespace-collapsed when ``text_normalisation`` is on."""
    rows = read_jsonl(ctx.corpus_path)
    return normalise_rows(rows) if ctx.normalise else rows


def fit_headA(ctx: ScoreContext) -> Any:
    """Fit Head A on the corpus's train split, save it, log train CV AUROC."""
    import numpy as np

    from src.features.stylometric import SYNTAX_FEATURE_NAMES, HeadA, StylometricExtractor

    cfg = ctx.models_config["head_a"]
    corpus = _corpus_rows(ctx)
    feats = stylometric_features(corpus, cfg, ctx.max_minutes)
    train = [r for r in corpus if r["split"] == "train"]
    X = np.asarray([feats[r["id"]] for r in train], dtype=float)
    y = np.asarray([r["label"] for r in train])
    buckets = [r["language"] for r in train]
    names = StylometricExtractor(syntax=True).feature_names()
    assert len(names) == X.shape[1] and names[-len(SYNTAX_FEATURE_NAMES):] == SYNTAX_FEATURE_NAMES
    head = HeadA(names, C=float(cfg.get("C", 1.0)), class_weight=cfg.get("class_weight", "balanced"))
    head.cv_ = head.cv_auroc(X, y, buckets, folds=int(cfg.get("cv_folds", 5)))
    head.fit(X, y, buckets)
    head.train_counts_ = {b: {"human": int(sum(1 for r in train if r["language"] == b and r["label"] == 0)),
                              "machine": int(sum(1 for r in train if r["language"] == b and r["label"] == 1))}
                          for b in sorted(set(buckets))}
    head.save(cfg["model_path"])
    log.info("headA fitted on %d train rows -> %s", len(train), cfg["model_path"])
    return head, feats


@register("headA", "Head A: 40 stylometric + parser features -> per-bucket logistic regression (Part 14)")
def score_headA(texts, rows, ctx):
    import numpy as np

    if "headA" not in ctx.cache:
        ctx.cache["headA"] = fit_headA(ctx)
    head, feats = ctx.cache["headA"]
    missing = [r for r in rows if r["id"] not in feats]            # e.g. attacked rows
    if missing:
        feats.update(stylometric_features(missing, ctx.models_config["head_a"]))
    X = np.asarray([feats[r["id"]] for r in rows], dtype=float)
    return head.predict(X, [r["language"] for r in rows]).tolist()


#: Columns served by one mGPT pass, in the order CurvatureScorer.full_stats returns them.
CURVATURE_COLUMNS = ("headB", "headB_word")
#: Columns served by one Qwen2.5-0.5B pass (same machinery, different scorer + columns).
QWEN_BASELINE_COLUMNS = ("fastdetectgpt_en", "ppl")


def _causal_lm_cache(rows: Sequence[dict[str, Any]], cfg: dict[str, Any], columns: tuple[str, ...],
                     extract: Callable[["np.ndarray"], Sequence[float]], label: str,
                     max_minutes: float = 0, device: str | None = None) -> dict[str, list[float]]:
    """Shared cache/journal machinery for any CurvatureScorer-based column group.

    One forward pass of ``cfg["scorer"]`` yields ``CurvatureScorer.full_stats()``'s
    three numbers per row; ``extract`` picks and orders the ones this group of
    columns wants (e.g. ``[d_tok, d_word]`` for headB/headB_word, ``[d_tok,
    ll_mean]`` for fastdetectgpt_en/ppl). Journalled as it completes, resumable,
    folded into ``cfg["stats_cache"]``, same pattern as ``stylometric_features``.
    """
    import pandas as pd

    from src.features.curvature import CurvatureScorer

    cache = Path(cfg["stats_cache"])
    journal = cache.parent / f".{cache.stem}.jsonl"
    have: dict[str, list[float]] = {}
    if cache.exists():
        table = pd.read_parquet(cache)
        have = dict(zip(table["id"], table[list(columns)].to_numpy().tolist()))
    missing = [r for r in rows if r["id"] not in have]
    if missing:
        scorer = CurvatureScorer(cfg["scorer"], device=device or cfg.get("device", "cuda"),
                                 load_in_8bit=bool(cfg.get("load_in_8bit", False)), config=cfg)
        scorer.load()
        log.info("%s scorer %s on %s, max_length %d", label, cfg["scorer"], scorer.model.device, scorer.max_length)
        missing.sort(key=lambda r: len(r["text"]))

        def process(batch: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
            S = scorer.full_stats([r["text"] for r in batch])
            return [{"id": r["id"], **dict(zip(columns, map(float, extract(s))))} for r, s in zip(batch, S)]

        report = run_resumable(missing, id_of=lambda r: r["id"], process_batch=process,
                               out_path=journal, batch_size=int(cfg.get("batch_size", 8)),
                               max_minutes=max_minutes, label=f"rows ({label})")
        if report.errors:
            raise RuntimeError(f"{label} scoring failed: {report.errors[0]}")
        for entry in read_jsonl(journal, skip_bad_lines=True):
            have[entry["id"]] = [entry[c] for c in columns]
        still = [r["id"] for r in rows if r["id"] not in have]
        if still:
            raise RuntimeError(f"{len(still)} rows still lack {label} (time budget hit?); re-run to resume")
        cache.parent.mkdir(parents=True, exist_ok=True)
        table = pd.DataFrame(list(have.values()), columns=list(columns))
        table.insert(0, "id", list(have))
        table.to_parquet(cache, index=False)
        journal.unlink(missing_ok=True)
    return {r["id"]: have[r["id"]] for r in rows}


def curvature_features(rows: Sequence[dict[str, Any]], cfg: dict[str, Any],
                       max_minutes: float = 0, device: str | None = None) -> dict[str, list[float]]:
    """Head B's two statistics for ``rows``, id -> [headB, headB_word], cached on disk."""
    return _causal_lm_cache(rows, cfg, CURVATURE_COLUMNS, lambda s: s[:2], "headB", max_minutes, device)


def qwen_baseline_features(rows: Sequence[dict[str, Any]], cfg: dict[str, Any],
                           max_minutes: float = 0, device: str | None = None) -> dict[str, list[float]]:
    """Qwen2.5-0.5B's two baseline statistics, id -> [fastdetectgpt_en, ppl], cached on disk."""
    return _causal_lm_cache(rows, cfg, QWEN_BASELINE_COLUMNS, lambda s: [s[0], s[2]],
                            "qwen_baseline", max_minutes, device)


def _cached_column(cache_key: str, columns: tuple[str, ...], name: str,
                   fetch: Callable[[Sequence[dict[str, Any]], ScoreContext], dict[str, list[float]]]) -> ColumnFn:
    """A column function that reads/fills one shared, keyed cache in ``ctx.cache``.

    Used for any column group computed together in one model pass (Head B's two
    columns, the Qwen-baseline's two columns): the first of the pair to run for a
    batch of rows fills the cache for both, the second is a free dict lookup.
    """
    k = columns.index(name)

    def fn(texts, rows, ctx):
        cache = ctx.cache.setdefault(cache_key, {})
        missing = [r for r in rows if r["id"] not in cache]          # rows not prepared up front
        if missing:
            cache.update(fetch(missing, ctx))
        return [cache[r["id"]][k] for r in rows]

    return fn


register("headB", "Head B: Fast-DetectGPT token-level curvature, scorer ai-forever/mGPT (Part 15)")(
    _cached_column("curvature", CURVATURE_COLUMNS, "headB",
                   lambda rows, ctx: curvature_features(rows, ctx.models_config["head_b"], ctx.max_minutes, ctx.device)))
register("headB_word", "Head B, word level: per-word standardised curvature, mGPT (Patent 2)")(
    _cached_column("curvature", CURVATURE_COLUMNS, "headB_word",
                   lambda rows, ctx: curvature_features(rows, ctx.models_config["head_b"], ctx.max_minutes, ctx.device)))
register("fastdetectgpt_en", "Baseline: Fast-DetectGPT with the English-default scorer Qwen2.5-0.5B (Part 17)")(
    _cached_column("qwen_baseline", QWEN_BASELINE_COLUMNS, "fastdetectgpt_en",
                   lambda rows, ctx: qwen_baseline_features(rows, ctx.models_config["qwen_baseline"],
                                                            ctx.max_minutes, ctx.device)))
register("ppl", "Baseline: negated mean log-perplexity, Qwen2.5-0.5B (Part 17)")(
    _cached_column("qwen_baseline", QWEN_BASELINE_COLUMNS, "ppl",
                   lambda rows, ctx: qwen_baseline_features(rows, ctx.models_config["qwen_baseline"],
                                                            ctx.max_minutes, ctx.device)))


@register("binoculars", "Baseline: Binoculars, observer Qwen2.5-0.5B / performer -0.5B-Instruct (Part 17)")
def score_binoculars(texts, rows, ctx):
    from src.baselines.binoculars import BinocularsDetector

    cfg = ctx.models_config["binoculars"]
    if "binoculars" not in ctx.cache:
        det = BinocularsDetector(cfg["observer"], cfg["performer"],
                                 device=ctx.device or cfg.get("device", "cuda"), config=cfg)
        det.load()
        log.info("binoculars observer=%s performer=%s on %s, max_length %d",
                 cfg["observer"], cfg["performer"], det.observer.device, det.max_length)
        ctx.cache["binoculars"] = det
    return ctx.cache["binoculars"].score(list(texts)).tolist()


def muril_embeddings(rows: Sequence[dict[str, Any]], cfg: dict[str, Any],
                     max_minutes: float = 0) -> dict[str, list[float]]:
    """Head C's mean-pooled MuRIL embeddings for ``rows``, id -> vector, cached on disk.

    Same pattern as ``stylometric_features``: the encoder pass is the slow part,
    so vectors are journalled as they're computed and a killed run resumes; the
    finished set is folded into the parquet.
    """
    import pandas as pd

    from src.features.semantic import MurilEmbedder

    cache = Path(cfg["embeddings_cache"])
    journal = cache.parent / f".{cache.stem}.jsonl"
    have: dict[str, list[float]] = {}
    if cache.exists():
        table = pd.read_parquet(cache)
        have = dict(zip(table["id"], table["embedding"].map(list)))
    missing = [r for r in rows if r["id"] not in have]
    if missing:
        embedder = MurilEmbedder(cfg["model"], device=cfg.get("device", "cuda"), config=cfg)
        embedder.load()
        log.info("headC embedder %s on %s, max_length %d", cfg["model"], embedder.model.device, embedder.max_length)
        missing.sort(key=lambda r: len(r["text"]))

        def process(batch: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
            X = embedder.embed([r["text"] for r in batch])
            return [{"id": r["id"], "embedding": x.tolist()} for r, x in zip(batch, X)]

        report = run_resumable(missing, id_of=lambda r: r["id"], process_batch=process,
                               out_path=journal, batch_size=int(cfg.get("batch_size", 16)),
                               max_minutes=max_minutes, label="rows (headC embeddings)")
        if report.errors:
            raise RuntimeError(f"headC embedding failed: {report.errors[0]}")
        for entry in read_jsonl(journal, skip_bad_lines=True):
            have[entry["id"]] = entry["embedding"]
        still = [r["id"] for r in rows if r["id"] not in have]
        if still:
            raise RuntimeError(f"{len(still)} rows still lack embeddings (time budget hit?); re-run to resume")
        cache.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"id": list(have), "embedding": list(have.values())}).to_parquet(cache, index=False)
        journal.unlink(missing_ok=True)
    return {r["id"]: have[r["id"]] for r in rows}


def fit_headC(ctx: ScoreContext) -> Any:
    """Fit Head C on the corpus's train split, save it, log train counts."""
    import numpy as np

    from src.features.semantic import HeadC

    cfg = ctx.models_config["head_c"]
    corpus = _corpus_rows(ctx)
    embs = muril_embeddings(corpus, cfg, ctx.max_minutes)
    train = [r for r in corpus if r["split"] == "train"]
    X = np.asarray([embs[r["id"]] for r in train], dtype=float)
    y = np.asarray([r["label"] for r in train])
    buckets = [r["language"] for r in train]
    head = HeadC(X.shape[1], C=float(cfg.get("C", 1.0)), class_weight=cfg.get("class_weight", "balanced"))
    head.fit(X, y, buckets)
    head.save(cfg["model_path"])
    log.info("headC fitted on %d train rows -> %s", len(train), cfg["model_path"])
    return head, embs


@register("headC", "Head C: MuRIL mean-pooled embeddings -> per-bucket logistic head (Part 16)")
def score_headC(texts, rows, ctx):
    import numpy as np

    if "headC" not in ctx.cache:
        ctx.cache["headC"] = fit_headC(ctx)
    head, embs = ctx.cache["headC"]
    missing = [r for r in rows if r["id"] not in embs]              # e.g. attacked rows
    if missing:
        embs.update(muril_embeddings(missing, ctx.models_config["head_c"]))
    X = np.asarray([embs[r["id"]] for r in rows], dtype=float)
    return head.predict(X, [r["language"] for r in rows]).tolist()


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
    parser.add_argument("--report", metavar="COLS",
                        help="comma-separated scored columns: print their per-bucket test AUROC side by side and exit")
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

    if args.report:
        cols = [c.strip() for c in args.report.split(",") if c.strip()]
        print(format_comparison({c: auroc_by_bucket(Path(args.scores), Path(args.corpus), c) for c in cols}))
        return
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
                       models_config=load_config(args.config), corpus_path=Path(args.corpus),
                       max_minutes=args.max_minutes)
    ctx.normalise = ctx.config.get("text_normalisation") == "whitespace"
    if ctx.normalise:
        rows = normalise_rows(rows)
        log.info("text_normalisation=whitespace: newlines/whitespace runs collapsed before scoring")
    column_fn = COLUMN_REGISTRY[args.column]
    if args.column == "headA":            # fit once, up front, not inside the first batch
        ctx.cache["headA"] = fit_headA(ctx)
    elif args.column == "headC":
        ctx.cache["headC"] = fit_headC(ctx)
    elif args.column in CURVATURE_COLUMNS:  # one mGPT pass serves both columns, cached
        ctx.cache["curvature"] = curvature_features(rows, ctx.models_config["head_b"],
                                                    args.max_minutes, args.device)
    elif args.column in QWEN_BASELINE_COLUMNS:  # one Qwen2.5-0.5B pass serves both columns, cached
        ctx.cache["qwen_baseline"] = qwen_baseline_features(rows, ctx.models_config["qwen_baseline"],
                                                             args.max_minutes, args.device)

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
    if Path(args.corpus).exists() and scores_path.exists():
        # Only headA/headC's cache is the (head, feats) shape format_auroc wants
        # (for the train-CV-spread columns); every other column's cache entry
        # (a loaded scorer/detector, or a curvature-cache dict) isn't, so it's
        # not a fitted head to report CV for.
        cached = ctx.cache.get(args.column)
        head = cached[0] if isinstance(cached, tuple) else None
        print(format_auroc(auroc_by_bucket(scores_path, Path(args.corpus), args.column), head))


def auroc_by_bucket(scores_path: Path, corpus_path: Path, column: str, split: str = "test",
                    n_boot: int = 1000, seed: int = 0) -> list[dict[str, Any]]:
    """Per-bucket AUROC of ``column`` on ``split``, with a 95 % bootstrap interval.

    The interval resamples human and machine rows separately (stratified), so each
    replicate keeps the split's class counts.
    """
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    scores = pd.read_parquet(scores_path).set_index("id")[column]
    rows = [r for r in read_jsonl(corpus_path) if r["split"] == split and r["id"] in scores.index]
    rng = np.random.default_rng(seed)
    out = []
    for bucket in sorted({r["language"] for r in rows}):
        sub = [r for r in rows if r["language"] == bucket]
        y = np.asarray([r["label"] for r in sub])
        s = scores.loc[[r["id"] for r in sub]].to_numpy(dtype=float)
        h, m = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
        boots = []
        for _ in range(n_boot):
            idx = np.concatenate([rng.choice(h, len(h)), rng.choice(m, len(m))])
            boots.append(roc_auc_score(y[idx], s[idx]))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out.append({"bucket": bucket, "split": split, "human": len(h), "machine": len(m),
                    "auroc": float(roc_auc_score(y, s)), "ci_lo": float(lo), "ci_hi": float(hi)})
    return out


def format_auroc(table: list[dict[str, Any]], head: Any = None) -> str:
    """The AUROC table; with a fitted supervised head, also its train CV spread."""
    cv = getattr(head, "cv_", None) or {}
    counts = getattr(head, "train_counts_", None) or {}
    lines = [f"\n{'bucket':6} {'test h/m':>9} {'AUROC':>6}  {'95% CI':>13}"
             + (f"  {'train h/m':>9}  {'train CV AUROC':>15}" if cv else "")]
    for t in table:
        line = (f"{t['bucket']:6} {t['human']:>4}/{t['machine']:<4} {t['auroc']:6.3f}  "
                f"[{t['ci_lo']:.3f}, {t['ci_hi']:.3f}]")
        if t["bucket"] in cv:
            c = counts.get(t["bucket"], {})
            mean, std = cv[t["bucket"]]
            line += f"  {c.get('human', 0):>4}/{c.get('machine', 0):<4}  {mean:.3f} ± {std:.3f}"
        lines.append(line)
    return "\n".join(lines)


def format_comparison(tables: dict[str, list[dict[str, Any]]]) -> str:
    """Several columns' per-bucket AUROC side by side, one row per bucket."""
    cols = list(tables)
    by = {c: {t["bucket"]: t for t in tables[c]} for c in cols}
    buckets = sorted({b for c in cols for b in by[c]})
    lines = [f"\n{'bucket':6} {'test h/m':>9}  " + "  ".join(f"{c + ' AUROC [95% CI]':>27}" for c in cols)]
    for b in buckets:
        first = next(by[c][b] for c in cols if b in by[c])
        cells = [f"{by[c][b]['auroc']:.3f} [{by[c][b]['ci_lo']:.3f}, {by[c][b]['ci_hi']:.3f}]"
                 if b in by[c] else "-" for c in cols]
        lines.append(f"{b:6} {first['human']:>4}/{first['machine']:<4}  " + "  ".join(f"{x:>27}" for x in cells))
    return "\n".join(lines)


if __name__ == "__main__":
    main()
