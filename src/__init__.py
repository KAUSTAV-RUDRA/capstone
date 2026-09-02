"""mgt-detect research package (PURE PYTHON — non-negotiable #7).

Detecting machine-generated text in multilingual student submissions
(en / hi / te / code-mixed) via stylometric + perplexity-curvature fusion
with per-language conformal abstention.

This package contains ZERO Django imports and must run with Django
uninstalled. The only bridge to the web layer is
``webapp/detector/services.py``.

Shared contracts enforced across the package:
- every head and every baseline exposes ``score(texts: list[str]) -> np.ndarray``
  returning a per-text machine-likelihood score (higher = more machine-like);
- ``StylometricExtractor`` exposes ``fit`` / ``transform`` / ``feature_names``;
- ``Fuser`` exposes ``fit(head_scores, labels)`` / ``predict_proba(head_scores)``;
- ``ConformalCalibrator`` exposes ``fit(scores_human_only, bucket)`` / ``threshold(bucket)``;
- ``AbstentionGate.decide(prob, bucket)`` returns ``("HUMAN"|"ABSTAIN"|"MACHINE", float)``.
"""

__version__ = "0.0.0"
