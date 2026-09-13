"""Per-sample corpus schema (docs/project-context-master.md §6).

Canonical record shape shared by every loader, generator, and experiment:

    id | text | label | language | code_mix_ratio | generator | domain |
    length_tokens | attack_type | writer_L1_band | split

Serves docs/master-execution-plan.md Phase 2 §2.1 (Corpus construction).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# --- Controlled vocabularies (real constants, not stubs) ---------------------
LANGUAGE_BUCKETS: tuple[str, ...] = ("en", "hi", "te", "cm")  # non-negotiable #1
SPLITS: tuple[str, ...] = ("train", "cal", "test")
ATTACK_TYPES: tuple[str, ...] = ("clean", "paraphrase", "back_translation", "hybrid")
WRITER_L1_BANDS: tuple[str, ...] = ("general", "indian", "unknown")  # en: general (L1 proxy) vs indian (Indian-English, L2 proxy); hi/te/cm bands set in parts-plan Part 2-3

LABEL_HUMAN: int = 0
LABEL_MACHINE: int = 1

SCHEMA_FIELDS: tuple[str, ...] = (
    "id",
    "text",
    "label",
    "language",
    "code_mix_ratio",
    "generator",
    "domain",
    "length_tokens",
    "attack_type",
    "writer_L1_band",
    "split",
)


@dataclass
class Sample:
    """A single corpus record. ``generator`` is None for human text."""

    id: str
    text: str
    label: int  # LABEL_HUMAN (0) or LABEL_MACHINE (1)
    language: str  # one of LANGUAGE_BUCKETS
    code_mix_ratio: float
    generator: str | None
    domain: str
    length_tokens: int
    attack_type: str | None  # None or one of ATTACK_TYPES
    writer_L1_band: str | None  # None or one of WRITER_L1_BANDS
    split: str  # one of SPLITS


def sample_from_dict(row: dict[str, Any]) -> Sample:
    """Build a :class:`Sample` from a raw dict/CSV row."""
    # TODO(phase-2 step-2.1.2): map and coerce fields into Sample.
    raise NotImplementedError


def sample_to_dict(sample: Sample) -> dict[str, Any]:
    """Serialise a :class:`Sample` to a dict with SCHEMA_FIELDS keys."""
    # TODO(phase-2 step-2.1.2): serialise Sample.
    raise NotImplementedError


def validate_sample(sample: Sample) -> None:
    """Raise ``ValueError`` if any field violates the controlled vocabulary."""
    # TODO(phase-2 step-2.1.2): enforce bucket/label/split/attack vocab.
    raise NotImplementedError
