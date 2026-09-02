"""Detection heads and preprocessing features.

- Head A: stylometric (:mod:`src.features.stylometric`)
- Head B: perplexity-curvature / Fast-DetectGPT (:mod:`src.features.curvature`)
- Head C: MuRIL semantic, optional (:mod:`src.features.semantic`)
- language ID + code-mix ratio (:mod:`src.features.language_id`)

Serves docs/master-execution-plan.md Phase 2 §2.2 (Heads and fusion).
"""
