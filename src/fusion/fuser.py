"""Fuser - logistic regression (or shallow GBM) over head scores + bucket one-hot.

Interpretability is a feature (locked §5): logistic regression first; GBM only
if it wins on calibration AUROC. Consumes per-head scores from Heads A/B(/C)
plus a bucket one-hot (review2-sprint Day 4 spec: ``[headA, headB, headC?,
bucket_onehot]``) and emits a single machine-probability that the calibration
layer consumes. Head C is excluded by default per
``docs/results/headc_diagnosis.md`` (2026-09-24): its 1.000 AUROC on en/hi/te
is a lexical fingerprint of the single seen generator scored so far, not a
generalisable signal (non-negotiable #4).

Contract: ``fit(head_scores, labels, buckets)`` / ``predict_proba(head_scores,
buckets)``, where ``head_scores`` has shape ``(n_samples, n_heads)``.

Serves docs/master-execution-plan.md Phase 2 §2.2.6.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np

from src.data.schema import LANGUAGE_BUCKETS

if TYPE_CHECKING:
    pass


class Fuser:
    """Interpretable fusion model over head outputs + bucket one-hot."""

    def __init__(self, method: str = "logistic", config: dict | None = None,
                buckets: Sequence[str] | None = None) -> None:
        """Args:
        method: ``"logistic"`` (default) or ``"gbm"``.
        config: Loaded ``configs/default.yaml`` (``fusion`` block).
        buckets: Bucket vocabulary for the one-hot block; defaults to the
            project's four (en/hi/te/cm).
        """
        if method not in ("logistic", "gbm"):
            raise ValueError(f"unknown fusion method {method!r}")
        self.method = method
        self.config = config or {}
        self.buckets = list(buckets) if buckets is not None else list(LANGUAGE_BUCKETS)
        self.model: object | None = None
        self.head_names_: list[str] | None = None
        self.feature_names_: list[str] | None = None

    def _features(self, head_scores: "np.ndarray", buckets: Sequence[str]) -> "np.ndarray":
        head_scores = np.atleast_2d(np.asarray(head_scores, dtype=float))
        onehot = np.array([[1.0 if b == bk else 0.0 for bk in self.buckets] for b in buckets])
        return np.hstack([head_scores, onehot])

    def fit(self, head_scores: "np.ndarray", labels: "np.ndarray", buckets: Sequence[str],
           head_names: Sequence[str] | None = None) -> "Fuser":
        """Fit the fusion model.

        Args:
            head_scores: Shape ``(n_samples, n_heads)`` of per-head scores.
            labels: Shape ``(n_samples,)`` with 0 = human, 1 = machine.
            buckets: Shape ``(n_samples,)`` language bucket per row.
            head_names: Names for the head columns (for :meth:`coefficients`).

        Returns:
            self.
        """
        head_scores = np.atleast_2d(np.asarray(head_scores, dtype=float))
        n_heads = head_scores.shape[1]
        self.head_names_ = list(head_names) if head_names is not None else [f"head_{i}" for i in range(n_heads)]
        if len(self.head_names_) != n_heads:
            raise ValueError("head_names length must match head_scores' column count")
        self.feature_names_ = self.head_names_ + [f"bucket_{b}" for b in self.buckets]

        X = self._features(head_scores, buckets)
        y = np.asarray(labels)
        if self.method == "logistic":
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler

            # Standardised: head scores and the raw curvature statistic (headB)
            # live on very different scales (headA/headC are probabilities in
            # [0, 1], headB ranges roughly [-17, 4]), so unstandardised
            # coefficients would not be comparable -- and "interpretable by
            # design" (locked §5) requires that they are.
            self.model = make_pipeline(StandardScaler(), LogisticRegression(
                max_iter=5000, class_weight="balanced"))
        else:
            from sklearn.ensemble import GradientBoostingClassifier

            self.model = GradientBoostingClassifier(random_state=42)
        self.model.fit(X, y)
        return self

    def predict_proba(self, head_scores: "np.ndarray", buckets: Sequence[str]) -> "np.ndarray":
        """Return machine-probability per sample, shape ``(n_samples,)``."""
        if self.model is None:
            raise RuntimeError("Fuser is not fitted")
        X = self._features(head_scores, buckets)
        return self.model.predict_proba(X)[:, 1]

    def decision_function(self, head_scores: "np.ndarray", buckets: Sequence[str]) -> "np.ndarray":
        """Return the raw logit (pre-sigmoid) per sample.

        Only defined for ``method="logistic"``; this is what
        :class:`src.calibration.temperature.TemperatureScaler` scales.
        """
        if self.method != "logistic":
            raise ValueError("decision_function is only defined for method='logistic'")
        if self.model is None:
            raise RuntimeError("Fuser is not fitted")
        X = self._features(head_scores, buckets)
        return self.model.decision_function(X)

    def predict(self, head_scores: "np.ndarray", buckets: Sequence[str], threshold: float = 0.5) -> "np.ndarray":
        """Return hard 0/1 predictions (pre-abstention, for diagnostics only)."""
        return (self.predict_proba(head_scores, buckets) >= threshold).astype(int)

    def coefficients(self) -> dict[str, float]:
        """Return per-feature weights in STANDARDISED space (logistic method).

        Comparable across features on different scales (headA/headC are
        probabilities in [0, 1], headB is a raw curvature statistic) because
        :meth:`fit` standardises features before the logistic regression.
        """
        if self.method != "logistic":
            raise ValueError("coefficients is only defined for method='logistic'")
        if self.model is None or self.feature_names_ is None:
            raise RuntimeError("Fuser is not fitted")
        logreg = self.model.named_steps["logisticregression"]
        return dict(zip(self.feature_names_, logreg.coef_[0].tolist()))

    def head_contributions(self, head_scores: "np.ndarray", buckets: Sequence[str]
                          ) -> list[dict[str, float]]:
        """Per-sample, per-HEAD (not bucket-one-hot) standardised contribution.

        ``contribution = standardised_value * coefficient`` for each head
        feature only -- the basis for "driving head" in the explanation
        payload (whichever head's |contribution| is larger). Only defined for
        ``method="logistic"``.
        """
        if self.method != "logistic":
            raise ValueError("head_contributions is only defined for method='logistic'")
        if self.model is None or self.head_names_ is None:
            raise RuntimeError("Fuser is not fitted")
        X = self._features(head_scores, buckets)
        scaler, logreg = self.model.named_steps["standardscaler"], self.model.named_steps["logisticregression"]
        z = scaler.transform(X)
        contribution = z * logreg.coef_[0]
        n_heads = len(self.head_names_)
        return [
            {name: float(contribution[row, i]) for i, name in enumerate(self.head_names_)}
            for row in range(contribution.shape[0])
        ]

    def save(self, path: str | Path) -> None:
        import joblib

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "Fuser":
        import joblib

        return joblib.load(path)
