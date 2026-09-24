"""Build the frozen corpus: clean, length-match, split, and write splits.json once.

NON-NEGOTIABLE #6: ``data/processed/splits.json`` is frozen once written and
must never be modified or regenerated. This module writes it exactly once and
refuses to overwrite an existing file (the "overwrite guard").

Usage::

    python -m src.data.freeze_splits --config configs/data.yaml --dry-run
    python -m src.data.freeze_splits --config configs/data.yaml

``--dry-run`` runs every stage and prints every table but writes nothing.

The pipeline, per bucket (``corpus`` in ``configs/data.yaml``)
-------------------------------------------------------------
1. **Clean** human and machine rows identically (:mod:`src.data.clean_artifacts`).
   Dropped rows are copied, with their reasons, to ``discarded_dir``; the raw
   files are never edited.
2. **Calibration draw**: ``cal_human_per_bucket`` human rows, only from
   calibration-eligible sources (cm: cmu_hinglish_dog + hinge, decisions.md
   2026-09-13), stratified by source x length bin. Drawn before matching and
   never trimmed by it: cal is human-only, so matching it buys nothing.
3. **Length-match** the classes where they meet, train/test: ``length_bins``
   quantile bins of the bucket's train/test human word counts, and the side named
   in ``length_match_trim`` is downsampled to the other side's bin distribution
   (en/hi/te trim human, cm trims machine; decisions.md 2026-09-24 Part 13).
4. **Split** the rest:

   - Every prompt (= human passage id) gets a **prompt group** split, train or
     test, balanced ``train_share`` within four strata: {has a machine row, does
     not} x {human in cal, not}. That makes human train/test and machine
     train/test both exact halves, and keeps a human passage and its machine
     continuation in the same split (decisions.md 2026-09-13 §3 carry-over).
   - Non-cal human rows and seen-generator rows take their prompt group's split;
     held-out generators go to ``test`` (non-negotiable #4).

   Groups are recorded for all prompts, including those whose human row is in
   ``cal``, so rows from generators added later have a split already decided.

Serves docs/master-execution-plan.md Phase 2 §2.1.1 and docs/parts-plan.md Part 13.
"""
from __future__ import annotations

import argparse
import bisect
import json
import logging
import os
import random
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

from src.data.schema import LANGUAGE_BUCKETS, SCHEMA_FIELDS, SPLITS

log = logging.getLogger("freeze_splits")


def calibration_eligible_sources(config: dict[str, Any]) -> dict[str, bool]:
    """Map ``source`` name -> may its rows enter the ``cal`` split?

    Reads ``human_corpus`` from ``configs/data.yaml``: every source is eligible
    unless its spec sets ``calibration_eligible: false`` (the default comes from
    ``calibration_eligible_default``).

    The conformal guarantee is only as strong as the certainty that the
    calibration text is human, so a source whose collection date cannot rule out
    machine text is barred from ``cal`` and routed to train/test instead. As of
    2026-09-13 that is ``comi_lingua`` (53 % of the ``cm`` bucket), which leaves
    cm calibration to cmu_hinglish_dog + hinge = 1,025 rows against a 1,000-row
    floor. See docs/decisions.md.
    """
    human_corpus = config.get("human_corpus") or {}
    default = bool(human_corpus.get("calibration_eligible_default", True))
    eligible: dict[str, bool] = {}
    for bucket_cfg in (human_corpus.get("buckets") or {}).values():
        for band_cfg in (bucket_cfg.get("bands") or {}).values():
            for spec in band_cfg.get("sources") or []:
                eligible[spec["name"]] = bool(spec.get("calibration_eligible", default))
    return eligible


# ---------------------------------------------------------------------------
# Length matching
# ---------------------------------------------------------------------------
def length_edges(words: list[int], n_bins: int) -> list[int]:
    """Inner cut points of ``n_bins`` quantile bins over ``words``.

    A row with ``n`` words falls in bin ``bisect_right(edges, n)``. Tied cut
    points (cm has a heavy short mode) are collapsed, so a bucket can end up with
    fewer than ``n_bins`` bins; the caller reports how many it got.
    """
    ordered = sorted(words)
    cuts = [ordered[int(len(ordered) * k / n_bins)] for k in range(1, n_bins)]
    return sorted(set(cuts))


def bin_of(n_words: int, edges: list[int]) -> int:
    """Index of the length bin ``n_words`` falls in."""
    return bisect.bisect_right(edges, n_words)


def length_match(reference: list[dict[str, Any]], pool: list[dict[str, Any]], edges: list[int],
                 rng: random.Random) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Downsample ``pool`` so its length-bin distribution equals ``reference``'s.

    Either side can be the pool: machine cut to the human distribution, or human
    cut to the machine one — what must hold is that the two match. The largest
    total ``T`` whose per-bin share equals the reference share without any bin
    running out is ``T = min_b pool_b / reference_share_b``; each bin then keeps
    ``floor(T * reference_share_b)`` rows, drawn at random.

    Returns:
        ``(kept, excluded, bins)``; ``bins`` has per-bin counts for the report.
    """
    n_real = len(edges) + 1
    ref_counts = Counter(bin_of(r["length_words"], edges) for r in reference)
    by_bin: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in sorted(pool, key=lambda r: r["id"]):
        by_bin[bin_of(r["length_words"], edges)].append(r)
    share = {b: ref_counts[b] / len(reference) for b in range(n_real)}
    total = min([len(pool)] + [len(by_bin[b]) / share[b] for b in range(n_real) if share[b] > 0])
    kept: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    bins = []
    for b in range(n_real):
        target = int(total * share[b])
        rows = by_bin[b]
        chosen = {r["id"] for r in (rng.sample(rows, target) if target < len(rows) else rows)}
        kept.extend(r for r in rows if r["id"] in chosen)
        excluded.extend(r for r in rows if r["id"] not in chosen)
        lo = edges[b - 1] if b > 0 else 0
        hi = edges[b] - 1 if b < len(edges) else None
        bins.append({"bin": f"{lo}-{hi if hi is not None else ''}", "reference": ref_counts[b],
                     "share": round(share[b], 3), "pool_in": len(rows), "pool_kept": len(chosen)})
    return kept, excluded, bins


# ---------------------------------------------------------------------------
# Split assignment
# ---------------------------------------------------------------------------
def _stratified_pick(rows: list[dict[str, Any]], k: int, key, rng: random.Random) -> list[dict[str, Any]]:
    """``k`` rows, proportional across ``key(row)`` strata (largest remainder)."""
    strata: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for r in sorted(rows, key=lambda r: r["id"]):
        strata[key(r)].append(r)
    quotas = {s: k * len(v) / len(rows) for s, v in strata.items()}
    take = {s: int(q) for s, q in quotas.items()}
    for s in sorted(quotas, key=lambda s: (quotas[s] - take[s], str(s)), reverse=True)[: k - sum(take.values())]:
        take[s] += 1
    picked: list[dict[str, Any]] = []
    for s in sorted(strata, key=str):
        picked.extend(rng.sample(strata[s], take[s]))
    return picked


def _halve(ids: list[str], train_share: float, rng: random.Random) -> dict[str, str]:
    ordered = sorted(ids)
    rng.shuffle(ordered)
    n_train = round(len(ordered) * train_share)
    return {pid: ("train" if i < n_train else "test") for i, pid in enumerate(ordered)}


def pick_cal(human: list[dict[str, Any]], config: dict[str, Any], rng: random.Random) -> set[str]:
    """Ids of one bucket's ``cal`` rows: calibration-eligible sources only,
    stratified by source x length bin.

    Raises:
        ValueError: If the calibration-eligible human rows cannot fill ``cal``.
    """
    corpus = config["corpus"]
    cal_n = int(corpus["cal_human_per_bucket"])
    floor = int(config.get("min_human_calibration_per_bucket", cal_n))
    eligible = calibration_eligible_sources(config)
    pool = [h for h in human if eligible.get(h.get("source"), True)]
    if len(pool) < max(cal_n, floor):
        raise ValueError(
            f"{human[0]['language']}: {len(pool)} calibration-eligible human rows after cleaning, "
            f"need {max(cal_n, floor)}. The conformal guarantee needs this floor; revisit the "
            f"source restriction (decisions.md 2026-09-13) rather than filling cal from "
            f"ineligible sources.")
    edges = length_edges([h["length_words"] for h in human], int(corpus["length_bins"]))
    cal = _stratified_pick(pool, cal_n, lambda r: (r.get("source"), bin_of(r["length_words"], edges)), rng)
    return {r["id"] for r in cal}


def assign_splits(human: list[dict[str, Any]], machine: list[dict[str, Any]], *,
                  config: dict[str, Any], rng: random.Random, cal_ids: set[str] | None = None,
                  extra_prompts: set[str] | frozenset[str] = frozenset(),
                  ) -> tuple[dict[str, str], dict[str, str]]:
    """Split one bucket.

    Args:
        cal_ids: The bucket's ``cal`` rows when already drawn (the freeze draws
            them before length matching); drawn here by :func:`pick_cal` if None.
        extra_prompts: Prompts whose human row is not in the corpus (trimmed by
            length matching). They still get a group, for later generators, in a
            stratum of their own so the kept rows' halves stay exact.

    Returns:
        ``(row_split, prompt_group)``: row id -> split, and prompt id -> train/test.
    """
    train_share = float(config["corpus"]["train_share"])
    heldout = set(config.get("heldout_generators") or [])
    if cal_ids is None:
        cal_ids = pick_cal(human, config, rng)

    seen_prompts = {m["prompt_id"] for m in machine if m["generator"] not in heldout}
    all_prompts = {h["id"] for h in human} | {m["prompt_id"] for m in machine} | set(extra_prompts)
    group: dict[str, str] = {}
    for has_machine in (True, False):
        for in_cal in (True, False):
            for extra in (True, False):
                stratum = [p for p in all_prompts if (p in seen_prompts) == has_machine
                           and (p in cal_ids) == in_cal and (p in extra_prompts) == extra]
                group.update(_halve(stratum, train_share, rng))

    split = {h["id"]: ("cal" if h["id"] in cal_ids else group[h["id"]]) for h in human}
    for m in machine:
        split[m["id"]] = "test" if m["generator"] in heldout else group[m["prompt_id"]]
    return split, group


def freeze_splits(
    samples: list[dict[str, Any]],
    output_path: str | Path,
    config: dict | None = None,
    force: bool = False,
    extra: dict[str, Any] | None = None,
    cal_ids: dict[str, set[str]] | None = None,
    extra_prompts: dict[str, set[str]] | None = None,
) -> dict[str, Any]:
    """Assign and persist immutable train/cal/test splits.

    ``samples`` are the cleaned, length-matched rows of every bucket (human rows
    carry ``source``; machine rows carry ``generator`` and ``prompt_id``).
    Held-out generators are routed to ``test`` only (non-negotiable #4). Each
    row's ``split`` is set in place.

    Args:
        samples: All corpus rows to partition.
        output_path: Destination ``splits.json`` path.
        config: Loaded ``configs/data.yaml`` (needs ``corpus`` and ``human_corpus``).
        force: Must stay False in normal use; the guard raises if the file
            already exists. Only tests pass True, on a temporary path.
        extra: Additional top-level keys to record (exclusions, counts).
        cal_ids: bucket -> ``cal`` ids already drawn (see :func:`assign_splits`).
        extra_prompts: bucket -> prompts with no row in ``samples``.

    Returns:
        The payload written to ``output_path``.

    Raises:
        FileExistsError: If ``output_path`` already exists (the freeze guard).
    """
    output_path = Path(output_path)
    if output_path.exists() and not force:
        raise FileExistsError(f"{output_path} is frozen (non-negotiable #6); refusing to overwrite it.")
    payload = build_split_payload(samples, config or {}, cal_ids, extra_prompts)
    payload.update(extra or {})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "x" if not force else "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
    return payload


def build_split_payload(samples: list[dict[str, Any]], config: dict[str, Any],
                        cal_ids: dict[str, set[str]] | None = None,
                        extra_prompts: dict[str, set[str]] | None = None) -> dict[str, Any]:
    """Assign splits (setting ``row["split"]``) and return the splits.json payload."""
    rng = random.Random(int(config["corpus"]["seed"]) + 1)
    splits: dict[str, list[str]] = {s: [] for s in SPLITS}
    groups: dict[str, str] = {}
    for bucket in LANGUAGE_BUCKETS:
        human = [r for r in samples if r["language"] == bucket and r["label"] == 0]
        machine = [r for r in samples if r["language"] == bucket and r["label"] == 1]
        if not human:
            continue
        row_split, group = assign_splits(
            human, machine, config=config, rng=rng,
            cal_ids=(cal_ids or {}).get(bucket), extra_prompts=(extra_prompts or {}).get(bucket, set()))
        groups.update(group)
        for r in human + machine:
            r["split"] = row_split[r["id"]]
            splits[r["split"]].append(r["id"])
    return {
        "version": config["corpus"].get("version"),
        "created": date.today().isoformat(),
        "seed": config["corpus"]["seed"],
        "generators": sorted({r["generator"] for r in samples if r.get("generator")}),
        "splits": {s: sorted(ids) for s, ids in splits.items()},
        "prompt_groups": dict(sorted(groups.items())),
    }


def load_frozen_splits(path: str | Path) -> dict[str, list[str]]:
    """Load the frozen split assignment (mapping split -> list of sample ids)."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["splits"]


def assert_splits_frozen(path: str | Path) -> None:
    """Raise if ``splits.json`` is missing; used as a precondition by experiments."""
    if not Path(path).is_file():
        raise FileNotFoundError(
            f"{path} does not exist: run `python -m src.data.freeze_splits --config "
            f"configs/data.yaml` (parts-plan Part 13) first.")


# ---------------------------------------------------------------------------
# Corpus rows
# ---------------------------------------------------------------------------
def to_corpus_row(row: dict[str, Any], length_tokens: int | None) -> dict[str, Any]:
    """The 11 schema fields, then provenance kept for traceability.

    ``code_mix_ratio`` is recomputed from the cleaned text for every row with the
    function that built the human cm bucket. Machine rows arrive with 0.0, which
    would separate the classes by metadata alone.
    """
    from src.data.build_human_corpus import hindi_word_ratio

    out = {
        "id": row["id"],
        "text": row["text"],
        "label": row["label"],
        "language": row["language"],
        "code_mix_ratio": round(hindi_word_ratio(row["text"]), 4) if row["language"] == "cm" else 0.0,
        "generator": row.get("generator"),
        "domain": row.get("domain"),
        "length_tokens": length_tokens,
        "attack_type": row.get("attack_type") or "clean",
        "writer_L1_band": row.get("writer_L1_band"),
        "split": row["split"],
    }
    assert tuple(out) == SCHEMA_FIELDS
    out.update({
        "length_words": row["length_words"],
        "source": row.get("source") or row.get("generator"),
        "prompt_id": row.get("prompt_id") or row["id"],
    })
    return out


def count_tokens(texts: list[str], tokenizer_name: str) -> list[int]:
    """Token counts under the Head B scorer's tokenizer (local cache only)."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    return [len(ids) for ids in tok(texts, add_special_tokens=False)["input_ids"]]


def count_table(rows: list[dict[str, Any]]) -> str:
    """bucket x source x split counts, with per-bucket human/machine totals."""
    c = Counter((r["language"], r.get("source") or r.get("generator"), r["split"]) for r in rows)
    lines = [f"{'bucket':6} {'source':18} {'cal':>6} {'train':>6} {'test':>6} {'total':>6}"]
    for bucket in LANGUAGE_BUCKETS:
        sources = sorted({s for b, s, _ in c if b == bucket})
        for s in sources:
            n = [c[(bucket, s, sp)] for sp in ("cal", "train", "test")]
            lines.append(f"{bucket:6} {s:18} {n[0]:>6} {n[1]:>6} {n[2]:>6} {sum(n):>6}")
        for label, name in ((0, "= human"), (1, "= machine")):
            sub = [r for r in rows if r["language"] == bucket and r["label"] == label]
            n = [sum(1 for r in sub if r["split"] == sp) for sp in ("cal", "train", "test")]
            lines.append(f"{'':6} {name:18} {n[0]:>6} {n[1]:>6} {n[2]:>6} {sum(n):>6}")
    return "\n".join(lines)


def load_raw(config: dict[str, Any], generators: list[str]) -> list[dict[str, Any]]:
    """Human rows of every bucket plus the named generators' machine rows."""
    from src.utils.io import read_jsonl

    human_dir = Path(config["human_corpus"]["output_dir"])
    machine_dir = Path(config["machine_corpus"]["output_dir"])
    rows: list[dict[str, Any]] = []
    for bucket in config.get("buckets", LANGUAGE_BUCKETS):
        rows.extend(read_jsonl(human_dir / f"{bucket}.jsonl"))
    for gen in generators:
        rows.extend(read_jsonl(machine_dir / f"{gen}.jsonl"))
    return rows


def main(argv: list[str] | None = None) -> int:
    from src.data.clean_artifacts import clean_rows, format_report
    from src.utils.config import load_config
    from src.utils.io import write_jsonl

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--dry-run", action="store_true", help="run every stage, write nothing")
    parser.add_argument("--allow-small-buckets", action="store_true",
                        help="freeze even if a bucket is below corpus.min_rows_per_side")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = load_config(args.config)
    corpus = config["corpus"]
    splits_path = Path(config["splits_file"])
    if splits_path.exists() and not args.dry_run:
        raise FileExistsError(f"{splits_path} is frozen (non-negotiable #6); refusing to rebuild the corpus.")

    rows = load_raw(config, list(corpus["generators"]))
    kept, dropped, report = clean_rows(rows)
    print("== Cleaning: % dropped per bucket x source ==")
    print(format_report(report))

    rng = random.Random(int(corpus["seed"]))
    matched: list[dict[str, Any]] = []
    excluded_len: list[str] = []
    cal_ids: dict[str, set[str]] = {}
    extra_prompts: dict[str, set[str]] = {}
    trim_side = corpus.get("length_match_trim") or {}
    print("\n== Length match: train/test human vs machine, cal drawn first and never trimmed ==")
    for bucket in LANGUAGE_BUCKETS:
        human = [r for r in kept if r["language"] == bucket and r["label"] == 0]
        machine = [r for r in kept if r["language"] == bucket and r["label"] == 1]
        if not human:
            continue
        cal_ids[bucket] = pick_cal(human, config, rng)
        cal = [h for h in human if h["id"] in cal_ids[bucket]]
        tt_human = [h for h in human if h["id"] not in cal_ids[bucket]]
        edges = length_edges([h["length_words"] for h in tt_human], int(corpus["length_bins"]))
        side = trim_side.get(bucket, "machine")
        if side == "human":
            tt_kept, out, bins = length_match(machine, tt_human, edges, rng)
            matched.extend(cal + tt_kept + machine)
            extra_prompts[bucket] = {h["id"] for h in out}
            ref_name, pool_name = "machine", "human"
        elif side == "machine":
            m_kept, out, bins = length_match(tt_human, machine, edges, rng)
            matched.extend(human + m_kept)
            ref_name, pool_name = "human", "machine"
        else:
            raise ValueError(f"corpus.length_match_trim.{bucket}: {side!r} (want human | machine)")
        excluded_len.extend(r["id"] for r in out)
        print(f"{bucket}: trim {pool_name} to the {ref_name} distribution; edges {edges}; "
              f"{pool_name} {len(out) + sum(b['pool_kept'] for b in bins)} -> "
              f"{sum(b['pool_kept'] for b in bins)}")
        for b in bins:
            print(f"   {b['bin']:>9}  {ref_name} {b['reference']:>5} ({b['share']:.3f})  "
                  f"{pool_name} {b['pool_in']:>5} -> {b['pool_kept']:>5}")

    payload = build_split_payload(matched, config, cal_ids, extra_prompts)
    print("\n== bucket x source x split ==")
    print(count_table(matched))

    floor = int(corpus["min_rows_per_side"])
    small = [f"{b} {side} {n}" for b in LANGUAGE_BUCKETS
             for side, label in (("human", 0), ("machine", 1))
             if (n := sum(1 for r in matched if r["language"] == b and r["label"] == label)) < floor]
    if small:
        print(f"\nBelow min_rows_per_side ({floor}): " + ", ".join(small))
    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return 0
    if small and not args.allow_small_buckets:
        print("Not freezing. Re-run with --allow-small-buckets once that is decided.")
        return 2

    texts = [r["text"] for r in matched]
    lengths = count_tokens(texts, corpus["length_tokenizer"])
    corpus_rows = [to_corpus_row(r, n) for r, n in zip(matched, lengths)]
    extra = {
        "excluded": {
            "clean": {r["id"]: r["drop_reasons"] for r in dropped},
            "length_match": sorted(excluded_len),
        },
    }
    freeze_splits(matched, splits_path, config, extra=extra, cal_ids=cal_ids, extra_prompts=extra_prompts)
    write_jsonl(sorted(corpus_rows, key=lambda r: r["id"]), corpus["path"])
    discarded = Path(corpus["discarded_dir"]) / f"part13_clean_{corpus['version']}.jsonl"
    write_jsonl(dropped, discarded)
    print(f"\nwrote {splits_path} ({sum(len(v) for v in payload['splits'].values())} ids), "
          f"{corpus['path']} ({len(corpus_rows)} rows), {discarded} ({len(dropped)} dropped rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
