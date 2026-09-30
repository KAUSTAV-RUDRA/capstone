> **Superseded (v1 corpus, raw text).** See [README.md](README.md): T3 for held-out results, the Head C section for the v2 diagnosis.

# T3 preliminary — llama on hi

**Preliminary, not a frozen-corpus result.** `data/processed/splits.json` was NOT touched. llama's 500 hi rows (the only bucket that cleared its 24-row gate — cm and te both failed theirs, decisions.md 2026-09-28) were cleaned with `src.data.clean_artifacts.clean_rows` (Part 13's cleaning rules, unchanged) and length-matched against the EXISTING frozen hi human test split (392 rows) with `src.data.freeze_splits.length_match` — the same mechanism Part 13 used, applied read-only against already-frozen reference data. headA and the fusion/conformal artifacts were LOADED from `results/models/` and `results/conformal_thresholds.csv`, not refit. headB is mGPT inference, which has no fitted state to reuse.

Cleaning: 479 of 500 llama hi rows kept (4.2% dropped). Length-matching against the 392-row frozen hi human test set kept **367** of those.

## AUROC (human hi test vs. machine), hi bucket only

| generator | n (machine) | headA | headB (mGPT) | fused (headA+headB, calibrated) |
|---|---|---|---|---|
| llama (held-out) | 367 | 0.937 | 0.805 | 0.894 |
| qwen7b (seen) | 233 | 0.984 | 0.187 | 0.991 |

llama's row count above (367) is the length-matched count, not the full 500 — length-matching trims the pool to the reference bin distribution, same as any other generator's rows would be trimmed against the human side.

## Finding: headB's hi inversion does not reproduce on llama

qwen7b's headB AUROC on hi is **0.187** — the known curvature inversion (headB inverts on hi/te for qwen with the mGPT scorer, decisions.md 2026-09-25 "Day 3: the hi/te inversion is mGPT-specific"). **llama's headB AUROC on the same bucket, same scorer, same fitted pipeline is 0.805** — no inversion. Fused AUROC (which leans on headA, the stronger column for both generators here) absorbs most of the difference (0.991 qwen vs 0.894 llama), but headB alone flips direction entirely between the two generators.

This is a genuinely new data point, not previously observable: every measurement of the hi/te inversion to date has been qwen7b-only (the only generator with hi rows in the frozen corpus). Whatever drives the inversion — a property of qwen's specific output distribution relative to mGPT's, rather than something universal about mGPT-scored curvature on Hindi — does not transfer to llama's output. This bears directly on non-negotiable #4 (held-out generators only for evaluation): if the inversion were a property of the *bucket*, it should reproduce here; that it doesn't suggests it is a property of the *generator*, which is exactly the kind of thing T3 exists to catch, and exactly the shape of finding this session has repeatedly run into with generator-bucket fit (phi, mistral). Worth a deliberate follow-up once cm/te held-out coverage exists to check whether this is hi-specific or generator-specific more broadly.

## FPR at tau_0.01 (hi)

`tau_0.01[hi]` = **0.9943** (from `results/conformal_thresholds.csv`, fit on the 1000-row hi calibration split, unchanged). Empirical false-positive rate on the frozen hi human **test** split (392 rows, calibrated fused score) = **0.0128**. This is a property of the human side only (same reference set for both llama and qwen comparisons above), not generator-specific.

## Caveats

- Preliminary: one held-out generator, one bucket (hi only — cm and te both failed llama's gate). Not a substitute for the frozen-corpus T3 once the corpus is formally extended.
- llama's hi rows are length-matched against frozen human test only; they were never length-matched against each other the way Part 13's cal/train/test split balances prompt groups, so there is no llama train/cal split here — this is test-only, read-only scoring.
- qwen's numbers are the frozen corpus's existing hi test AUROC, included for a same-bucket seen-vs-held-out comparison, not recomputed.
- headB's non-inversion on llama (above) is one bucket, one generator, unreplicated — reported as a finding worth following up, not a settled result.
