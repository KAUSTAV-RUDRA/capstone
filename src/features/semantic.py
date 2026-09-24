"""Head C - MuRIL embeddings + shallow head (OPTIONAL, VRAM >= 8GB only).

NON-NEGOTIABLE #9: MuRIL is encoder-only and CANNOT compute perplexity or
curvature. It is used here ONLY to produce embeddings for a shallow classifier.
Skipped entirely when VRAM < 8GB (locked §5 / risk register).

Two classes, mirroring Head A's split in :mod:`src.features.stylometric`
(``StylometricExtractor`` / ``HeadA``): :class:`MurilEmbedder` turns text into
pooled embeddings, :class:`HeadC` is the per-bucket classifier fit on them.
Per bucket, not pooled, for the same reason as Head A — a language's embedding
baseline differs, so a pooled model would spend capacity telling buckets apart
rather than telling human from machine (non-negotiable #3).

Serves docs/master-execution-plan.md Phase 2 §2.2.5 and review2-sprint Day 3.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from src.features.curvature import run_group_with_oom_retry, token_budget_groups

if TYPE_CHECKING:
    import torch


class MurilEmbedder:
    """MuRIL mean-pooled embeddings. Encoder-only — embeddings ONLY (non-neg #9)."""

    def __init__(
        self,
        model_name: str = "google/muril-base-cased",
        device: str = "cpu",
        config: dict | None = None,
    ) -> None:
        """Args:
        model_name: Encoder-only HF id (MuRIL).
        device: ``"cpu"`` or ``"cuda"``.
        config: the ``head_c`` section of ``configs/models.yaml``.
        """
        self.config = config or {}
        self.model_name = model_name
        self.device = device
        self.max_length = int(self.config.get("max_length", 512))
        self.max_batch_tokens = int(self.config.get("max_batch_tokens", 16384))
        self.max_rows_per_batch = int(self.config.get("batch_size", 16))
        self.tokenizer: Any = None
        self.model: Any = None

    def load(self) -> None:
        """Load the MuRIL encoder and tokenizer (lazily, once)."""
        if self.model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(self.model_name, torch_dtype=dtype).to(self.device).eval()
        self.max_length = min(self.max_length, int(getattr(self.model.config, "max_position_embeddings",
                                                           self.max_length)))

    def _pool(self, texts: list[str], group: list[int], result: "np.ndarray") -> None:
        import torch

        enc = self.tokenizer([texts[i] for i in group], truncation=True, max_length=self.max_length,
                             padding=True, return_tensors="pt")
        mask = enc["attention_mask"].to(self.device)
        with torch.no_grad():
            hidden = self.model(input_ids=enc["input_ids"].to(self.device), attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).float()
        pooled = (hidden * m).sum(1) / m.sum(1).clamp_min(1e-6)
        result[group] = pooled.double().cpu().numpy()

    def embed(self, texts: list[str]) -> "np.ndarray":
        """Return mean-pooled MuRIL embeddings, shape ``(len(texts), hidden)``.

        Batched by token budget with the same OOM-safe retry as Head B
        (:mod:`src.features.curvature`) — an encoder has no autoregressive
        lm-head to chunk, but the transformer body is still dense over padding.
        """
        self.load()
        texts = list(texts)
        lengths = [min(len(ids), self.max_length)
                  for ids in self.tokenizer(texts, add_special_tokens=False)["input_ids"]]
        result = np.full((len(texts), self.model.config.hidden_size), np.nan)
        for group in token_budget_groups(lengths, self.max_batch_tokens, self.max_rows_per_batch):
            run_group_with_oom_retry(lambda g: self._pool(texts, g, result), group)
        return result


class HeadC:
    """The trained Head C: one standardise -> logistic regression per bucket,
    over MuRIL embeddings. Same design as ``HeadA``
    (:mod:`src.features.stylometric`) — see its docstring for why per-bucket.

    ``predict`` returns P(machine), higher = more machine-like (score.py's
    column convention).
    """

    def __init__(self, hidden_size: int, C: float = 1.0, class_weight: str | None = "balanced",
                max_iter: int = 5000) -> None:
        self.hidden_size = hidden_size
        self.C = C
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.models: dict[str, Any] = {}

    def _new_model(self) -> Any:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(StandardScaler(), LogisticRegression(
            C=self.C, class_weight=self.class_weight, max_iter=self.max_iter))

    def fit(self, X: np.ndarray, y: np.ndarray, buckets: Sequence[str]) -> "HeadC":
        buckets = np.asarray(buckets)
        for bucket in sorted(set(buckets.tolist())):
            mask = buckets == bucket
            if len(set(y[mask].tolist())) < 2:
                raise ValueError(f"bucket {bucket}: train split needs both classes")
            self.models[bucket] = self._new_model().fit(X[mask], y[mask])
        return self

    def predict(self, X: np.ndarray, buckets: Sequence[str]) -> np.ndarray:
        buckets = np.asarray(buckets)
        out = np.full(len(buckets), np.nan)
        for bucket in sorted(set(buckets.tolist())):
            mask = buckets == bucket
            if bucket not in self.models:
                raise KeyError(f"no Head C model for bucket {bucket!r}")
            out[mask] = self.models[bucket].predict_proba(X[mask])[:, 1]
        return out

    def save(self, path: str | Path) -> None:
        import joblib

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "HeadC":
        import joblib

        return joblib.load(path)
