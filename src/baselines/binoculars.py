"""Baseline: Binoculars (Hans, ICML 2024) - cross-perplexity ratio.

Scores the ratio of a text's perplexity under one LM (the observer) to its
cross-perplexity under a second, closely related LM (the performer). Included
as a strong zero-shot T1 baseline.

Shared contract: ``score(texts) -> np.ndarray``.

Serves docs/master-execution-plan.md Phase 2 §2.2.1 and review2-sprint Day 3.

Score, and the sign
--------------------
Per position ``i`` the observer gives ``l_i = log P_observer(x_i | x_<i)`` (the
observed token's log-prob), and the performer gives a distribution ``q_i`` over
the vocabulary; the cross term is ``m_i = E_{v~q_i}[log P_observer(v | x_<i)]``.
The paper's raw statistic is a ratio of two perplexities::

    B(x) = PPL(x) / X-PPL(x)
    log PPL(x)   = -mean_i(l_i)
    log X-PPL(x) = -mean_i(m_i)

with **lower** ``B`` indicating machine-generated text. This module reports
``-log B(x) = log X-PPL(x) - log PPL(x) = mean_i(l_i - m_i)`` instead: a
strictly decreasing reparametrisation of ``B``, so it ranks texts identically
(same AUROC) while matching every other column's convention of higher = more
machine-like.

Reuses Head B's token-budget batching and OOM-safe retry
(:mod:`src.features.curvature`) rather than re-deriving them, since observer
and performer are each a full causal-LM forward pass over the same batch.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from src.features.curvature import run_group_with_oom_retry, token_budget_groups

if TYPE_CHECKING:
    import torch


class BinocularsDetector:
    """Cross-perplexity (observer/performer) detector."""

    def __init__(
        self,
        observer_model: str = "ai-forever/mGPT",
        performer_model: str = "Qwen/Qwen2.5-0.5B",
        device: str = "cpu",
        config: dict | None = None,
    ) -> None:
        """Args:
        observer_model: HF id scored for its own perplexity.
        performer_model: HF id whose distribution weights the cross term. Must
            share a tokenizer/vocab with ``observer_model``.
        device: ``"cpu"`` or ``"cuda"``.
        config: the ``binoculars`` section of ``configs/models.yaml``.
        """
        self.config = config or {}
        self.observer_model = observer_model
        self.performer_model = performer_model
        self.device = device
        self.max_length = int(self.config.get("max_length", 2048))
        self.vocab_chunk = int(self.config.get("vocab_chunk_positions", 512))
        self.max_batch_tokens = int(self.config.get("max_batch_tokens", 4096))
        self.max_rows_per_batch = int(self.config.get("batch_size", 8))
        self.tokenizer: Any = None
        self.observer: Any = None
        self.performer: Any = None

    def load(self) -> None:
        """Load the observer and performer models + a shared tokenizer."""
        if self.observer is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.observer_model)
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.observer = AutoModelForCausalLM.from_pretrained(
            self.observer_model, torch_dtype=dtype).to(self.device).eval()
        self.performer = AutoModelForCausalLM.from_pretrained(
            self.performer_model, torch_dtype=dtype).to(self.device).eval()
        if self.observer.config.vocab_size != self.performer.config.vocab_size:
            raise ValueError(
                "observer_model and performer_model must share a tokenizer/vocab "
                "for the cross-perplexity term.")
        self.max_length = min(self.max_length, int(getattr(
            self.observer.config, "max_position_embeddings", self.max_length)))

    def _cross_stats(self, input_ids: "torch.Tensor", attention_mask: "torch.Tensor") -> list[float]:
        """Per row, ``mean_i(l_i - m_i)`` over its scored positions."""
        import torch

        with torch.no_grad():
            h_obs = self.observer.base_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            h_perf = self.performer.base_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            head_obs = self.observer.get_output_embeddings()
            head_perf = self.performer.get_output_embeddings()
            out = []
            for row in range(input_ids.size(0)):
                n = int(attention_mask[row].sum())
                labels = input_ids[row, 1:n]
                parts = []
                for s in range(0, n - 1, self.vocab_chunk):
                    e = min(s + self.vocab_chunk, n - 1)
                    logp_obs = torch.log_softmax(head_obs(h_obs[row, s:e]).float(), dim=-1)
                    p_perf = torch.softmax(head_perf(h_perf[row, s:e]).float(), dim=-1)
                    l = logp_obs.gather(-1, labels[s:e, None]).squeeze(-1)
                    m = (p_perf * logp_obs).sum(-1)
                    parts.append(l - m)
                dev = torch.cat(parts) if parts else torch.zeros(0)
                out.append(float(dev.double().mean().item()) if dev.numel() else float("nan"))
        return out

    def score(self, texts: list[str]) -> np.ndarray:
        """Return per-text ``mean_i(l_i - m_i)``. Higher = more machine-like."""
        self.load()
        texts = list(texts)
        lengths = [min(len(ids), self.max_length)
                  for ids in self.tokenizer(texts, add_special_tokens=False)["input_ids"]]
        result = np.full(len(texts), np.nan)

        def do(group: list[int]) -> None:
            enc = self.tokenizer([texts[i] for i in group], truncation=True, max_length=self.max_length,
                                 padding=True, return_tensors="pt")
            vals = self._cross_stats(enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device))
            for gi, v in zip(group, vals):
                result[gi] = v

        for group in token_budget_groups(lengths, self.max_batch_tokens, self.max_rows_per_batch):
            run_group_with_oom_retry(do, group)
        return result
