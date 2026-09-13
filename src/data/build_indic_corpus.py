"""Assemble the IndicStudentMGT corpus (10-15K prompt/length-matched pairs).

Pulls human text from Indic/code-mixed sources (IndicCorp, AI4Bharat catalogue,
HinGE, COMI-LINGUA, L3Cube, Indian student essays, TOEFL-style L2 corpora),
pairs it with seen-generator machine text, and emits schema-conformant records
across the four buckets (en / hi / te / cm).

Serves docs/master-execution-plan.md Phase 2 §2.1.2 and §2.1.8 (corpus card).
"""
from __future__ import annotations

from src.data.schema import Sample


def load_human_sources(config: dict | None = None) -> list[Sample]:
    """Load and normalise human text from all configured Indic/L2 sources."""
    # TODO(phase-2 step-2.1.2): read configured human sources into Samples.
    raise NotImplementedError


def match_prompts_and_lengths(
    human: list[Sample],
    machine: list[Sample],
    config: dict | None = None,
) -> tuple[list[Sample], list[Sample]]:
    """Prompt-match and length-match human vs machine per bucket.

    Length distributions per bucket must be indistinguishable (KS-test
    p > 0.05) — see §2.1.3 done-when criterion.
    """
    # TODO(phase-2 step-2.1.2): balance prompts and length bins per bucket.
    raise NotImplementedError


def build_corpus(config: dict | None = None) -> list[Sample]:
    """Produce the full IndicStudentMGT corpus as schema-conformant samples."""
    # TODO(phase-2 step-2.1.2): assemble human+machine into one corpus.
    raise NotImplementedError


def write_corpus_card(samples: list[Sample], output_path: str, config: dict | None = None) -> None:
    """Write docs/data/IndicStudentMGT_card.md (sources, licences, counts)."""
    # TODO(phase-2 step-2.1.8): emit datasheet / corpus card.
    raise NotImplementedError
