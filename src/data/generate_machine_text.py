"""Generate machine text via local Ollama for the seen generators.

Three seen open models (e.g. Llama-3-8B, Gemma-2-9B, Mistral-7B) produce
prompt-matched, length-binned machine text against the human corpus
(locked decision §5). Held-out generators are produced separately and used
for test only (non-negotiable #4).

Serves docs/master-execution-plan.md Phase 2 §2.1.3.
"""
from __future__ import annotations

from src.data.schema import Sample


def generate_for_prompts(
    prompts: list[str],
    generator: str,
    config: dict | None = None,
    length_bin_tokens: int | None = None,
) -> list[Sample]:
    """Generate machine samples for a list of prompts with one generator.

    Args:
        prompts: Prompts matched to the human corpus.
        generator: Ollama model tag (e.g. ``"llama3:8b"``).
        config: Loaded ``configs/models.yaml``.
        length_bin_tokens: Optional target length to match human length dist.

    Returns:
        Generated machine samples (label = MACHINE, generator set).
    """
    # TODO(phase-2 step-2.1.3): call Ollama and wrap outputs as Samples.
    raise NotImplementedError


def build_seen_generator_corpus(config: dict | None = None) -> list[Sample]:
    """Generate the full seen-generator machine corpus across all buckets."""
    # TODO(phase-2 step-2.1.3): iterate seen_generators from models.yaml.
    raise NotImplementedError


def build_heldout_generator_corpus(config: dict | None = None) -> list[Sample]:
    """Generate the held-out-generator machine corpus (test only)."""
    # TODO(phase-2 step-2.1.5): iterate heldout_generators; route to test.
    raise NotImplementedError
