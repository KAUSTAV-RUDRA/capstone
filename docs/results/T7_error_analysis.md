# T7: error analysis (clean test split, frozen gate)

Reproduce: `python -m experiments.exp14_error_analysis --config configs/models_norm.yaml`. Rates are share of the stated denominator, with 95% Wilson intervals in percent. Examples: `T7_examples.md`. Gate: MACHINE if p > tau_alpha, HUMAN if p < human calibration median, else ABSTAIN.

## Confusion counts

### alpha = 0.05

| bucket | human / machine | correct MACHINE | correct HUMAN | false flag (human→MACHINE) | false clear (machine→HUMAN) | abstain (all) | abstain human / machine rows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| en | 402 / 1091 | 1032/1091 = 94.6% [93.1, 95.8] | 213/402 = 53.0% [48.1, 57.8] | 21/402 = 5.2% [3.4, 7.9] | 2/1091 = 0.2% [0.1, 0.7] | 225/1493 = 15.1% [13.3, 17.0] | 41.8% [37.1, 46.7] / 5.2% [4.1, 6.7] |
| hi | 449 / 923 | 752/923 = 81.5% [78.8, 83.8] | 243/449 = 54.1% [49.5, 58.7] | 20/449 = 4.5% [2.9, 6.8] | 7/923 = 0.8% [0.4, 1.6] | 350/1372 = 25.5% [23.3, 27.9] | 41.4% [37.0, 46.0] / 17.8% [15.4, 20.4] |
| te | 397 / 225 | 203/225 = 90.2% [85.6, 93.5] | 189/397 = 47.6% [42.7, 52.5] | 20/397 = 5.0% [3.3, 7.7] | 3/225 = 1.3% [0.5, 3.8] | 207/622 = 33.3% [29.7, 37.1] | 47.4% [42.5, 52.3] / 8.4% [5.5, 12.8] |
| cm | 597 / 232 | 110/232 = 47.4% [41.1, 53.8] | 471/597 = 78.9% [75.4, 82.0] | 20/597 = 3.4% [2.2, 5.1] | 22/232 = 9.5% [6.3, 13.9] | 206/829 = 24.8% [22.0, 27.9] | 17.8% [14.9, 21.0] / 43.1% [36.9, 49.5] |

### alpha = 0.01

| bucket | human / machine | correct MACHINE | correct HUMAN | false flag (human→MACHINE) | false clear (machine→HUMAN) | abstain (all) | abstain human / machine rows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| en | 402 / 1091 | 871/1091 = 79.8% [77.4, 82.1] | 213/402 = 53.0% [48.1, 57.8] | 4/402 = 1.0% [0.4, 2.5] | 2/1091 = 0.2% [0.1, 0.7] | 403/1493 = 27.0% [24.8, 29.3] | 46.0% [41.2, 50.9] / 20.0% [17.7, 22.5] |
| hi | 449 / 923 | 359/923 = 38.9% [35.8, 42.1] | 243/449 = 54.1% [49.5, 58.7] | 4/449 = 0.9% [0.3, 2.3] | 7/923 = 0.8% [0.4, 1.6] | 759/1372 = 55.3% [52.7, 57.9] | 45.0% [40.4, 49.6] / 60.3% [57.2, 63.5] |
| te | 397 / 225 | 36/225 = 16.0% [11.8, 21.4] | 189/397 = 47.6% [42.7, 52.5] | 3/397 = 0.8% [0.3, 2.2] | 3/225 = 1.3% [0.5, 3.8] | 391/622 = 62.9% [59.0, 66.6] | 51.6% [46.7, 56.5] / 82.7% [77.2, 87.1] |
| cm | 597 / 232 | 73/232 = 31.5% [25.8, 37.7] | 471/597 = 78.9% [75.4, 82.0] | 5/597 = 0.8% [0.4, 1.9] | 22/232 = 9.5% [6.3, 13.9] | 258/829 = 31.1% [28.1, 34.4] | 20.3% [17.2, 23.7] / 59.1% [52.6, 65.2] |

## Errors by length bin and by generator / source corpus

`errors/n`: false flags are over human rows of that level, false clears over machine rows. `share` = this level's share of all errors of that type in the bucket. Levels with no errors are omitted from the markdown (all levels are in `T7_breakdown.csv`).

### False flags (human→MACHINE), alpha 0.05

| bucket | dimension | level | errors/n | rate [95% CI] | share |
| --- | --- | --- | --- | --- | --- |
| en | length_bin | 100-149 | 2/97 | 2.1% [0.6, 7.2] | 10% |
| en | length_bin | 150-199 | 8/103 | 7.8% [4.0, 14.6] | 38% |
| en | length_bin | 200+ | 11/202 | 5.4% [3.1, 9.5] | 52% |
| en | source corpus | hc3 | 12/123 | 9.8% [5.7, 16.3] | 57% |
| en | source corpus | samanantar_en | 3/190 | 1.6% [0.5, 4.5] | 14% |
| en | source corpus | wikipedia | 6/89 | 6.7% [3.1, 13.9] | 29% |
| hi | length_bin | 100-149 | 7/151 | 4.6% [2.3, 9.3] | 35% |
| hi | length_bin | 150-199 | 8/150 | 5.3% [2.7, 10.2] | 40% |
| hi | length_bin | 200+ | 5/148 | 3.4% [1.5, 7.7] | 25% |
| hi | source corpus | indiccorp_v2 | 14/333 | 4.2% [2.5, 6.9] | 70% |
| hi | source corpus | wikipedia | 6/116 | 5.2% [2.4, 10.8] | 30% |
| te | length_bin | 100-149 | 12/194 | 6.2% [3.6, 10.5] | 60% |
| te | length_bin | 150-199 | 6/168 | 3.6% [1.6, 7.6] | 30% |
| te | length_bin | 200+ | 2/35 | 5.7% [1.6, 18.6] | 10% |
| te | source corpus | indiccorp_v2 | 17/319 | 5.3% [3.4, 8.4] | 85% |
| te | source corpus | wikipedia | 3/78 | 3.8% [1.3, 10.7] | 15% |
| cm | length_bin | <25 | 15/471 | 3.2% [1.9, 5.2] | 75% |
| cm | length_bin | 25-49 | 5/118 | 4.2% [1.8, 9.5] | 25% |
| cm | source corpus | comi_lingua | 20/587 | 3.4% [2.2, 5.2] | 100% |

### False flags (human→MACHINE), alpha 0.01

| bucket | dimension | level | errors/n | rate [95% CI] | share |
| --- | --- | --- | --- | --- | --- |
| en | length_bin | 200+ | 4/202 | 2.0% [0.8, 5.0] | 100% |
| en | source corpus | hc3 | 2/123 | 1.6% [0.4, 5.7] | 50% |
| en | source corpus | samanantar_en | 1/190 | 0.5% [0.1, 2.9] | 25% |
| en | source corpus | wikipedia | 1/89 | 1.1% [0.2, 6.1] | 25% |
| hi | length_bin | 100-149 | 2/151 | 1.3% [0.4, 4.7] | 50% |
| hi | length_bin | 150-199 | 2/150 | 1.3% [0.4, 4.7] | 50% |
| hi | source corpus | indiccorp_v2 | 3/333 | 0.9% [0.3, 2.6] | 75% |
| hi | source corpus | wikipedia | 1/116 | 0.9% [0.2, 4.7] | 25% |
| te | length_bin | 100-149 | 3/194 | 1.5% [0.5, 4.4] | 100% |
| te | source corpus | indiccorp_v2 | 2/319 | 0.6% [0.2, 2.3] | 67% |
| te | source corpus | wikipedia | 1/78 | 1.3% [0.2, 6.9] | 33% |
| cm | length_bin | <25 | 4/471 | 0.8% [0.3, 2.2] | 80% |
| cm | length_bin | 25-49 | 1/118 | 0.8% [0.1, 4.6] | 20% |
| cm | source corpus | comi_lingua | 5/587 | 0.9% [0.4, 2.0] | 100% |

### False clears (machine→HUMAN) — independent of alpha

| bucket | dimension | level | errors/n | rate [95% CI] | share |
| --- | --- | --- | --- | --- | --- |
| en | length_bin | 200+ | 2/568 | 0.4% [0.1, 1.3] | 100% |
| en | generator | mistral | 1/378 | 0.3% [0.0, 1.5] | 50% |
| en | generator | qwen7b | 1/361 | 0.3% [0.0, 1.6] | 50% |
| hi | length_bin | 100-149 | 4/251 | 1.6% [0.6, 4.0] | 57% |
| hi | length_bin | 150-199 | 1/304 | 0.3% [0.1, 1.8] | 14% |
| hi | length_bin | 200+ | 2/301 | 0.7% [0.2, 2.4] | 29% |
| hi | generator | gemma | 1/125 | 0.8% [0.1, 4.4] | 14% |
| hi | generator | llama | 3/479 | 0.6% [0.2, 1.8] | 43% |
| hi | generator | qwen7b | 3/233 | 1.3% [0.4, 3.7] | 43% |
| te | length_bin | 100-149 | 1/96 | 1.0% [0.2, 5.7] | 33% |
| te | length_bin | 150-199 | 1/97 | 1.0% [0.2, 5.6] | 33% |
| te | length_bin | 200+ | 1/18 | 5.6% [1.0, 25.8] | 33% |
| te | generator | gemma | 2/61 | 3.3% [0.9, 11.2] | 67% |
| te | generator | qwen7b | 1/164 | 0.6% [0.1, 3.4] | 33% |
| cm | length_bin | <25 | 19/184 | 10.3% [6.7, 15.6] | 86% |
| cm | length_bin | 25-49 | 3/24 | 12.5% [4.3, 31.0] | 14% |
| cm | generator | gemma | 9/134 | 6.7% [3.6, 12.3] | 41% |
| cm | generator | qwen7b | 13/98 | 13.3% [7.9, 21.4] | 59% |

## What the errors have in common (per bucket)

Means below are over the clean test rows of the bucket at alpha 0.05. They describe correlations in the errors, not causes, and the groups are small (20-22 false flags, 2-22 false clears per bucket).

**en.** The 21 false flags are formal, encyclopedic text: 12 come from HC3 (9.8% of its human rows), 6 from Wikipedia, 3 from the Indian-news source (1.6%); all are 100+ words and 11 of 21 are 200+. They have longer average words (5.6 vs 5.1 characters) and fewer digits than correctly handled human rows (0.4% vs 1.0% of characters), and Head A is high (0.76 vs 0.03) while Head B is unremarkable. Only 2 machine texts are called HUMAN, both long (270-276 words) technical explanations of maths topics (HC3 prompts) with Head A near zero.

**hi.** False flags (20) are ordinary indiccorp/Wikipedia prose with no length pattern; again Head A is high (0.84 vs 0.07) and digit share low (0.2% vs 1.0%). The two sampled examples (T7_examples.md) read as garbled or clumsy web copy. The 7 false clears come from all three generators (llama 3, qwen7b 3, gemma 1) and are slightly more digit-heavy (2.1% vs 0.3% in caught machine text) with longer words; Head A is near zero. No length bin or generator stands out given n.

**te.** Same signature: false flags (20, 85% from indiccorp_v2) have Head A 0.73 vs 0.05 and fewer digits; 12 of 20 fall in the 100-149-word bin. Only 3 machine texts are cleared (2 gemma, 1 qwen7b), both samples shown are news-style with dateline and names, with Head A ~0. Head B carries almost no information here (it is inverted on te, AUROC 0.235), so the error pattern is Head A alone; one bin (200+) has 1 error in 18 rows and is not interpretable.

**cm.** Dominated by short text: the median cm passage is about 20 words. False clears (22, 9.5% of machine rows) are shorter than caught machine text (20 vs 30 words), all under 50 words, with a clear rate of 10-13% in those bins, and Head A is low (0.22 vs 0.96) because very short romanised text carries little stylometric signal; both generators are affected (qwen7b 13%, gemma 7%). False flags (20, all comi_lingua YouTube-style comments) are slightly shorter than other human texts (20 vs 23 words) with Head A high (0.94). In cm both error types look like short-text noise rather than a content pattern; Head B adds nothing (0.65 AUROC).

**Across buckets.** In every bucket the errors track Head A (high on false flags, near zero on false clears) and almost never Head B, and false flags have fewer digits than correct human rows. That is consistent with the provenance confound in the results README: the stylometric head partly reads scraped-vs-prompted surface form, so its mistakes are text that looks unusually "clean" for scraped human data, or unusually messy for generated data.
