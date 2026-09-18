"""Adversarial attacks on machine text (resumable CLI). STUB — filled in Stage 6.

Usage (once implemented)::

    python -m src.data.attack --attack paraphrase --max-minutes 110

Robustness under paraphrase is the first way commercial detectors fail
(docs/project-context-master.md §2), so T4 is what shows the stylometry-curvature
fusion holding where either head alone collapses. Three families, matching
RAID/DetectRL practice:

``hybrid``
    20 % of sentences get a synonym swap, clause reorder, or split/merge. No GPU.
    Models the realistic case of a student editing generated text. Part 23.
``paraphrase``
    Qwen2.5-7B-Instruct 4-bit: "Rewrite in your own words, same language, same
    length". Part 24.
``backtranslation``
    IndicTrans2 en<->indic: en->hi->en, hi->en->hi, te->en->te; for cm,
    Devanagari-normalise via indic-nlp-library, hi->en->hi, then re-romanise.
    Part 25.

Input is TEST-split rows only, sampled per bucket per attack: at most
``MACHINE_ROWS_PER_BUCKET`` machine rows and ``HUMAN_ROWS_PER_BUCKET`` human
rows, stratified by length bin and by generator so no generator drops out of the
table. Human rows are attacked too, or a rise in false positives under attack
would be invisible. T4 needs a precise estimate of degradation, not the whole
test set: 400 rows per cell holds the standard error on AUROC under 0.02
(decisions.md 2026-09-18). Output rows keep the source id with a suffix (``_hyb``,
``_para``, ``_bt``), set ``attack_type``, and are appended to
``data/processed/attacked.jsonl``.

This module is a stub so the CLI surface, the output contract and the resume
behaviour are fixed now, in Part 4, alongside generate and score.

Serves docs/parts-plan.md Part 4(c) and Stage 6.
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.data.schema import ATTACK_TYPES
from src.utils.config import load_config
from src.utils.resumable import DEFAULT_BATCH_SIZE

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


def select_rows(corpus_path: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    """Sampled TEST rows per bucket: <= MACHINE_ROWS_PER_BUCKET machine + HUMAN_ROWS_PER_BUCKET human."""
    # TODO(stage-6 part-23): read corpus.jsonl, filter split == "test", then per
    # bucket take a seeded stratified sample — machine rows balanced across length
    # bin AND generator so every generator stays represented, human rows across
    # length bin. Never attack train or cal rows: T4 measures degradation on test,
    # and attacking cal rows would contaminate the conformal thresholds.
    raise NotImplementedError("select_rows lands in parts-plan Part 23")


def apply_hybrid(texts: Sequence[str], buckets: Sequence[str]) -> list[str]:
    """Edit HYBRID_EDIT_FRACTION of sentences: synonym swap, clause reorder, or split/merge."""
    # TODO(stage-6 part-23): WordNet synonyms for en; small hi/te synonym lists.
    raise NotImplementedError("apply_hybrid lands in parts-plan Part 23")


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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    from src.utils.logging import configure_logging

    configure_logging(args.log_level)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    attack_type, suffix, part = ATTACKS[args.attack]
    assert attack_type in ATTACK_TYPES, f"{attack_type} missing from schema.ATTACK_TYPES"
    config = load_config(args.config)
    log.info("attack=%s attack_type=%s suffix=%s out=%s", args.attack, attack_type, suffix, args.out)
    raise NotImplementedError(
        f"src/data/attack.py is a stub: the '{args.attack}' attack lands in {part}. "
        f"The CLI surface, output contract (ids suffixed '{suffix}', attack_type "
        f"'{attack_type}', appended to {args.out}) and resume behaviour are fixed now."
    )


if __name__ == "__main__":
    main()
