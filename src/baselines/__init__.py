"""Baseline detectors for comparison (docs/project-context-master.md §7, T1).

Five baselines, all exposing the shared contract
``score(texts: list[str]) -> np.ndarray`` (higher = more machine-like):
perplexity threshold, DetectGPT, Fast-DetectGPT, Binoculars, XLM-R supervised.

Serves docs/master-execution-plan.md Phase 2 §2.2.1.
"""
