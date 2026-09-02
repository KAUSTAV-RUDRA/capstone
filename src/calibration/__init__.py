"""Per-language calibration and abstention - THE research contribution.

NON-NEGOTIABLE #3: abstention + per-language conformal thresholds are the
contribution and must never be removed, simplified, or collapsed to a single
global threshold.

- temperature scaling (:mod:`src.calibration.temperature`)
- split-conformal thresholds per bucket (:mod:`src.calibration.conformal`)
- three-way abstention gate (:mod:`src.calibration.abstention`)

Serves docs/master-execution-plan.md Phase 3 §3.1-§3.3.
"""
