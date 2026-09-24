"""Head B - Fast-DetectGPT conditional-probability curvature, token- and word-level.

NON-NEGOTIABLE #2: uses curvature (a second-order property), NOT raw perplexity.
Scorer model = mGPT-1.3B (fallback Qwen2.5-0.5B), a causal LM (locked §5).
NON-NEGOTIABLE #9: MuRIL is encoder-only and must NEVER be used here.

Implements the shared detector contract ``score(texts) -> np.ndarray``.

Serves docs/master-execution-plan.md Phase 2 §2.2.3-§2.2.4 and review2-sprint Day 2.

Two statistics, one forward pass
--------------------------------
At each scored position ``i`` the scorer gives ``p_i = p(. | x_<i)``; the observed
token has log-prob ``l_i``, and under ``p_i`` the log-prob has mean ``m_i`` and
variance ``v_i`` (the analytic estimator of Bao et al., reference = scorer).

**headB** (token level, = Fast-DetectGPT)::

    d_tok = Σ_i (l_i - m_i) / sqrt(Σ_i v_i)

**headB_word** (word level, Patent 2). Tokens are grouped by the whitespace-
delimited source word they came from; per word ``w`` the log-prob and both
conditional moments are summed, ``L_w = Σ_{i∈w} l_i`` (so ``M_w``, ``V_w``), and
each *word* is standardised on its own::

    z_w    = (L_w - M_w) / sqrt(max(V_w, floor))
    d_word = Σ_w z_w / sqrt(W)

Summing per word and then pooling (``Σ_w (L_w - M_w) / sqrt(Σ_w V_w)``) would
reproduce ``d_tok`` exactly, because every term is additive; the word statistic
is only different — and only fragmentation-invariant — because the
standardisation happens per word. In ``d_tok`` a Telugu word split into six
tokens carries six tokens' worth of variance; in ``d_word`` it carries one vote.
Both are ~N(0, 1) under the null that the text was sampled from the scorer.

Tokens with no word character (newlines, stray spaces) join the preceding word,
so both statistics are computed over exactly the same token set. The first token
of a text has no prediction and is unscored, as in Fast-DetectGPT.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import torch

_WORD = re.compile(r"\S+")


def token_word_ids(text: str, offsets: list[tuple[int, int]]) -> list[int]:
    """Source-word index for each token, from the tokenizer's character offsets.

    A token belongs to the word of the first non-space character in its span
    (byte-level BPE folds the leading space into the token). Tokens with no such
    character join the previous token's word (the next one's, at the start).
    """
    char_word = np.full(len(text) + 1, -1, dtype=int)
    for w, m in enumerate(_WORD.finditer(text)):
        char_word[m.start():m.end()] = w
    ids = []
    for start, end in offsets:
        span = char_word[start:max(end, start + 1)]
        hit = span[span >= 0]
        ids.append(int(hit[0]) if hit.size else -1)
    # Whitespace-only tokens: attach backwards, then forwards for a leading run.
    for i in range(1, len(ids)):
        if ids[i] < 0:
            ids[i] = ids[i - 1]
    first = next((w for w in ids if w >= 0), 0)
    return [first if w < 0 else w for w in ids]


def curvature_stats(lp: np.ndarray, mean: np.ndarray, var: np.ndarray, word_ids: np.ndarray,
                    word_var_floor: float = 1e-4) -> tuple[float, float]:
    """``(d_tok, d_word)`` from per-position observed log-prob, mean and variance.

    ``word_ids[i]`` is the source word of scored position ``i``. NaN when nothing
    is scored.
    """
    if lp.size == 0:
        return float("nan"), float("nan")
    d_tok = float((lp - mean).sum() / np.sqrt(max(var.sum(), 1e-8)))
    _, group = np.unique(word_ids, return_inverse=True)
    dev_w = np.bincount(group, weights=lp - mean)
    var_w = np.bincount(group, weights=var)
    z = dev_w / np.sqrt(np.maximum(var_w, word_var_floor))
    d_word = float(z.sum() / np.sqrt(z.size))
    return d_tok, d_word


class CurvatureScorer:
    """Fast-DetectGPT curvature scorer over a multilingual causal LM."""

    def __init__(
        self,
        scorer_model: str = "ai-forever/mGPT",
        device: str = "cpu",
        load_in_8bit: bool = False,
        config: dict | None = None,
    ) -> None:
        """Args:
        scorer_model: HF id of the causal scorer (fallback ``Qwen/Qwen2.5-0.5B``).
        device: ``"cpu"`` or ``"cuda"``.
        load_in_8bit: 8-bit loading to respect the 4GB VRAM floor (non-neg #5).
        config: the ``head_b`` section of ``configs/models.yaml``.
        """
        self.config = config or {}
        self.scorer_model = scorer_model
        self.device = device
        self.load_in_8bit = load_in_8bit
        self.max_length = int(self.config.get("max_length", 2048))
        self.vocab_chunk = int(self.config.get("vocab_chunk_positions", 512))
        self.word_var_floor = float(self.config.get("word_var_floor", 1e-4))
        # Transformer-body memory is ~ rows x longest_row^2 (attention), not sum of
        # real lengths (position_stats masks the *lm-head* work but the body still
        # runs dense over padding). A fixed row count OOMs on long buckets: 8 x 2048
        # peaked at 7.88 GiB of an 7.9956 GiB card. Capping rows x max_len instead
        # keeps short-text batches at the configured batch_size and only shrinks
        # long-text ones (te, hi).
        self.max_batch_tokens = int(self.config.get("max_batch_tokens", 8192))
        self.max_rows_per_batch = int(self.config.get("batch_size", 8))
        self.tokenizer: Any = None
        self.model: Any = None

    def load(self) -> None:
        """Load the tokenizer and causal LM (lazily, once)."""
        if self.model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.scorer_model)
        self.tokenizer.padding_side = "right"     # absolute positions (GPT-2 family): never left-pad
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        kwargs: dict[str, Any] = {}
        if self.load_in_8bit:
            from transformers import BitsAndBytesConfig
            kwargs.update(quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map=self.device)
        else:
            kwargs["torch_dtype"] = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(self.scorer_model, **kwargs).eval()
        if not self.load_in_8bit:
            self.model.to(self.device)
        self.max_length = min(self.max_length, int(getattr(self.model.config, "max_position_embeddings",
                                                           self.max_length)))

    def position_stats(self, input_ids: "torch.Tensor", attention_mask: "torch.Tensor"
                       ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Per row, ``(l, m, v)`` over its scored positions 1..n-1 (right-padded batch).

        Only the final hidden states are batched; the vocab projection and softmax
        run per row in position chunks, in fp32, so a 100k vocab never
        materialises as a (batch x length x vocab) tensor.
        """
        import torch

        with torch.no_grad():
            hidden = self.model.base_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            head = self.model.get_output_embeddings()
            out = []
            for row in range(input_ids.size(0)):
                n = int(attention_mask[row].sum())
                labels = input_ids[row, 1:n]
                parts = []
                for s in range(0, n - 1, self.vocab_chunk):
                    logits = head(hidden[row, s:min(s + self.vocab_chunk, n - 1)]).float()
                    logp = torch.log_softmax(logits, dim=-1)
                    p = logp.exp()
                    m = (p * logp).sum(-1)
                    v = ((p * logp.square()).sum(-1) - m.square()).clamp_min(0)
                    lab = labels[s:s + logits.size(0)]
                    parts.append(torch.stack([logp.gather(-1, lab[:, None]).squeeze(-1), m, v]))
                stats = (torch.cat(parts, dim=1).double().cpu().numpy() if parts
                         else np.zeros((3, 0)))
                out.append((stats[0], stats[1], stats[2]))
        return out

    def _token_budget_groups(self, lengths: list[int]) -> list[list[int]]:
        """Index groups s.t. each group's ``rows * max(length)`` stays under the token budget.

        Greedy over indices sorted short-to-long, so a mixed-length input batch
        (the caller may not have pre-sorted) still packs efficiently rather than
        padding every group to the input's longest text.
        """
        order = sorted(range(len(lengths)), key=lambda i: lengths[i])
        groups: list[list[int]] = []
        group: list[int] = []
        group_max = 0
        for i in order:
            n = lengths[i]
            grown = max(group_max, n)
            if group and (len(group) + 1) * grown > self.max_batch_tokens:
                groups.append(group)
                group, group_max = [], 0
                grown = n
            group.append(i)
            group_max = grown
            if len(group) >= self.max_rows_per_batch:
                groups.append(group)
                group, group_max = [], 0
        if group:
            groups.append(group)
        return groups

    def _score_group(self, texts: list[str], group: list[int], result: "np.ndarray", depth: int = 0) -> None:
        """Run one token-budget group; on a real CUDA OOM, clear the allocator cache
        and bisect the group rather than lose the whole run.

        A sustained scoring run makes hundreds of differently-shaped allocations;
        the caching allocator can fragment well before true usage nears the card's
        limit (measured: 216/6784 rows into a run that peaked at 5.3/8.0 GiB in
        isolation). ``empty_cache()`` after every group trades a little sync
        overhead for not carrying that fragmentation into the next shape.
        """
        import torch

        try:
            enc = self.tokenizer([texts[i] for i in group], truncation=True, max_length=self.max_length,
                                 padding=True, return_offsets_mapping=True, return_tensors="pt")
            rows = self.position_stats(enc["input_ids"].to(self.model.device),
                                       enc["attention_mask"].to(self.model.device))
            for j, gi in enumerate(group):
                lp, mean, var = rows[j]
                n = int(enc["attention_mask"][j].sum())
                words = token_word_ids(texts[gi], enc["offset_mapping"][j, :n].tolist())
                result[gi] = curvature_stats(lp, mean, var, np.asarray(words[1:]), self.word_var_floor)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(group) == 1 or depth >= 6:
                raise
            mid = len(group) // 2
            self._score_group(texts, group[:mid], result, depth + 1)
            self._score_group(texts, group[mid:], result, depth + 1)
        finally:
            if self.device == "cuda":
                torch.cuda.empty_cache()

    def stats(self, texts: list[str]) -> np.ndarray:
        """``(len(texts), 2)`` array of ``[d_tok, d_word]``.

        Internally re-batched to a VRAM-safe token budget (see ``max_batch_tokens``),
        so callers can pass any batch size without risking an OOM on long buckets.
        """
        self.load()
        texts = list(texts)
        lengths = [min(len(ids), self.max_length)
                  for ids in self.tokenizer(texts, add_special_tokens=False)["input_ids"]]
        result = np.full((len(texts), 2), np.nan)
        for group in self._token_budget_groups(lengths):
            self._score_group(texts, group, result)
        return result

    def curvature(self, text: str) -> float:
        """Token-level Fast-DetectGPT curvature for one text."""
        return float(self.stats([text])[0, 0])

    def score(self, texts: list[str]) -> np.ndarray:
        """Token-level curvature per text (headB). Higher = more machine-like."""
        return self.stats(texts)[:, 0]

    def score_word(self, texts: list[str]) -> np.ndarray:
        """Word-level curvature per text (headB_word). Higher = more machine-like."""
        return self.stats(texts)[:, 1]
