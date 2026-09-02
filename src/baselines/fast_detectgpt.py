"""Baseline: Fast-DetectGPT (Bao, ICLR 2024) - conditional-probability curvature.

The reference zero-shot baseline and the mechanism behind Head B. Used first as
a Review-1 smoke test on HC3 English (Phase 1, Day 3-4) and later as a T1
baseline. Head B (:mod:`src.features.curvature`) extends this with script
normalisation and fertility-aware segmentation.

Shared contract: ``score(texts) -> np.ndarray``.

Serves docs/master-execution-plan.md Phase 1 (Day 3-4) and Phase 2 §2.2.1.

Method
------
The analytic (single-model) estimator from Fast-DetectGPT: with the reference
distribution set to the scorer itself, the sampling discrepancy has a closed
form and needs a single forward pass. Per text, at each position ``i`` the
scorer gives a conditional distribution ``p(. | x_<i)`` with log-probs
``lp_i``; let the observed token be ``x_i``. Then::

    log p(x)  = Σ_i lp_i(x_i)                       (observed log-likelihood)
    μ         = Σ_i  E_{v~p_i}[lp_i(v)]             (= -Σ entropy)
    σ²        = Σ_i  Var_{v~p_i}[lp_i(v)]
    d(x)      = (log p(x) - μ) / sqrt(σ²)

Higher ``d(x)`` = more machine-like. Weights load in fp16 on CUDA; the softmax
math is done in fp32 for numerical stability over the large vocab.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch


class FastDetectGPTDetector:
    """Conditional-probability curvature detector."""

    def __init__(
        self,
        scorer_model: str = "ai-forever/mGPT",
        reference_model: str | None = None,
        device: str = "cpu",
        config: dict | None = None,
    ) -> None:
        """Args:
        scorer_model: Causal LM used for scoring (fallback ``Qwen/Qwen2.5-0.5B``).
        reference_model: Optional separate sampling model; defaults to scorer.
        device: ``"cpu"`` or ``"cuda"``.
        config: Loaded ``configs/models.yaml``.
        """
        self.scorer_model = scorer_model
        self.reference_model = reference_model or scorer_model
        self.device = device
        self.config = config or {}
        self.max_length = int(self.config.get("max_length", 512))
        # fp16 on CUDA, fp32 on CPU (resolved at load()).
        self._tokenizer = None
        self._scorer = None
        self._reference = None
        self._loaded = False

    def load(self) -> None:
        """Load the scorer (and reference) model + tokenizer."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._tokenizer = AutoTokenizer.from_pretrained(self.scorer_model)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._scorer = (
            AutoModelForCausalLM.from_pretrained(self.scorer_model, torch_dtype=dtype)
            .to(self.device)
            .eval()
        )
        if self.reference_model == self.scorer_model:
            self._reference = self._scorer  # analytic single-model estimator
        else:
            self._reference = (
                AutoModelForCausalLM.from_pretrained(self.reference_model, torch_dtype=dtype)
                .to(self.device)
                .eval()
            )
            # The analytic discrepancy mixes scorer log-probs with reference
            # probs position-by-position: it only makes sense on a shared vocab.
            if self._reference.config.vocab_size != self._scorer.config.vocab_size:
                raise ValueError(
                    "reference_model and scorer_model must share a tokenizer/vocab "
                    "for the analytic curvature estimator."
                )
        self._loaded = True

    def _curvature_one(self, text: str) -> float:
        import torch

        enc = self._tokenizer(
            text, truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        input_ids = enc["input_ids"].to(self.device)
        if input_ids.size(1) < 2:  # need at least one conditional prediction
            return float("nan")

        with torch.no_grad():
            score_logits = self._scorer(input_ids).logits
            ref_logits = (
                score_logits
                if self._reference is self._scorer
                else self._reference(input_ids).logits
            )

        # Align: logits[:, :-1] predict tokens input_ids[:, 1:]. Upcast to fp32.
        labels = input_ids[:, 1:]
        score_logits = score_logits[:, :-1, :].float()
        ref_logits = ref_logits[:, :-1, :].float()

        lprobs_score = torch.log_softmax(score_logits, dim=-1)
        probs_ref = torch.softmax(ref_logits, dim=-1)

        log_likelihood = lprobs_score.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        mean_ref = (probs_ref * lprobs_score).sum(dim=-1)              # E[lp]
        var_ref = (probs_ref * lprobs_score.square()).sum(dim=-1) - mean_ref.square()

        var_sum = var_ref.sum(dim=-1).clamp_min(1e-8)                 # guard /0
        discrepancy = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / var_sum.sqrt()
        return float(discrepancy.item())

    def score(self, texts: list[str]) -> "np.ndarray":
        """Return per-text curvature scores (higher = more machine-like)."""
        if not self._loaded:
            self.load()
        return np.asarray([self._curvature_one(t) for t in texts], dtype=float)


if __name__ == "__main__":
    # Day-3 smoke test: 10 English samples (5 human, 5 LLM-style). Loads the
    # FALLBACK scorer Qwen/Qwen2.5-0.5B (fast download), fp16 on CUDA.
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    human = [  # Wikipedia-style, human-written encyclopedic prose
        "The Amazon River in South America is the largest river by discharge "
        "volume of water in the world, and the disputed longest river system in "
        "the world in comparison to the Nile.",
        "Marie Curie was a Polish and naturalised-French physicist and chemist "
        "who conducted pioneering research on radioactivity. She was the first "
        "woman to win a Nobel Prize and remains the only person to win it in two "
        "scientific fields.",
        "Mount Kilimanjaro is a dormant volcano in Tanzania with three volcanic "
        "cones: Kibo, Mawenzi, and Shira. It is the highest mountain in Africa "
        "and the highest single free-standing mountain above sea level in the "
        "world.",
        "The movable-type printing press was introduced to Europe by Johannes "
        "Gutenberg around 1440. His method for making type is traditionally "
        "considered to have included a type-metal alloy and a hand mould for "
        "casting the type.",
        "Photosynthesis is a process used by plants, algae, and some bacteria to "
        "convert light energy into chemical energy stored in glucose. The "
        "byproduct oxygen is released into the atmosphere through tiny pores "
        "called stomata.",
    ]
    machine = [  # stereotypical LLM-assistant prose
        "Certainly! Here are some key considerations to keep in mind. First and "
        "foremost, it is important to understand that effective communication "
        "plays a crucial role in fostering meaningful relationships and driving "
        "successful outcomes across a wide variety of contexts.",
        "In today's fast-paced world, leveraging technology has become "
        "increasingly essential. By embracing innovative solutions and staying "
        "adaptable, individuals and organizations alike can unlock new "
        "opportunities and achieve their goals more efficiently and effectively.",
        "It's worth noting that maintaining a healthy work-life balance is "
        "essential for overall well-being. By prioritizing self-care, setting "
        "clear boundaries, and managing your time wisely, you can enhance both "
        "your productivity and your happiness.",
        "There are several important factors to consider when making this "
        "decision. Ultimately, the best approach will depend on your specific "
        "needs and circumstances, so it is always a good idea to carefully weigh "
        "the pros and cons before proceeding.",
        "Absolutely! Learning a new language can be a rewarding and enriching "
        "experience. With consistent practice, dedication, and the right "
        "resources, you will be well on your way to becoming proficient and "
        "confident in no time.",
    ]

    detector = FastDetectGPTDetector(scorer_model="Qwen/Qwen2.5-0.5B", device=device)
    scores = detector.score(human + machine)
    h_scores, m_scores = scores[:5], scores[5:]

    print(f"scorer=Qwen/Qwen2.5-0.5B  device={device}  "
          f"dtype={'fp16' if device == 'cuda' else 'fp32'}\n")
    print("-- human (Wikipedia-style) --")
    for text, s in zip(human, h_scores):
        print(f"  {s:8.4f}   {text[:58]}...")
    print("-- machine (LLM-style) --")
    for text, s in zip(machine, m_scores):
        print(f"  {s:8.4f}   {text[:58]}...")

    h_mean, m_mean = float(np.nanmean(h_scores)), float(np.nanmean(m_scores))
    print(f"\nmean curvature — human: {h_mean:.4f} | machine: {m_mean:.4f}")
    print(f"machine higher on average: {m_mean > h_mean}  (Δ = {m_mean - h_mean:+.4f})")
