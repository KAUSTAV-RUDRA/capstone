# T4: adversarial robustness (frozen detector)

Frozen fuser + per-bucket temperature + per-bucket conformal thresholds, fit on clean data only; attacked rows are scored, never refit. Reproduce: `python -m experiments.exp06_adversarial --config configs/models_norm.yaml`. Source: `T4_adversarial.csv`.

Columns: AUROC (95% CI); det = machine rows flagged MACHINE at alpha 0.05 / 0.01; ff = human rows flagged MACHINE (false-flag) at alpha 0.05 / 0.01; abstain and false-clear (machine called HUMAN) are for the shipped three-way gate (alpha 0.01; the CSV also has 0.05); [lo, hi] = 95% Wilson interval in percent; len = mean attacked/source word-count ratio. AUROC needs >= 30 rows per class, else n/a.

| bucket | attack | slice | n h/m | AUROC [95% CI] | clean-matched AUROC | det 5% | det 1% | ff 5% | ff 1% | abstain 1% | false-clear 1% | len |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| en | clean | all test | 402/1091 | 0.989 [0.984, 0.993] | n/a | 94.6% [93, 96] | 79.8% [77, 82] | 5.2% [3, 8] | 1.0% [0, 3] | 27.0% [25, 29] | 0.2% [0, 1] | 1.00 |
| en | hybrid | all | 150/400 | 0.985 [0.976, 0.994] | 0.990 | 93.2% [90, 95] | 72.5% [68, 77] | 4.0% [2, 8] | 1.3% [0, 5] | 30.2% [26, 34] | 0.2% [0, 1] | 1.00 |
| en | hybrid | attack_changed=false | 0/1 | n/a | n/a | 100.0% [21, 100] | 0.0% [0, 79] | n/a | n/a | 100.0% [21, 100] | 0.0% [0, 79] | 1.00 |
| en | paraphrase | all | 100/250 | 0.920 [0.890, 0.949] | 0.985 | 96.8% [94, 98] | 69.6% [64, 75] | 49.0% [39, 59] | 7.0% [3, 14] | 43.7% [39, 49] | 0.4% [0, 2] | 0.85 |
| en | paraphrase | machine sources = qwen7b (self-paraphrase) | 100/88 | 0.905 [0.862, 0.944] | n/a | 95.5% [89, 98] | 69.3% [59, 78] | 49.0% [39, 59] | 7.0% [3, 14] | 55.9% [49, 63] | 0.0% [0, 4] | 0.82 |
| en | paraphrase | machine sources != qwen7b | 100/162 | 0.928 [0.894, 0.955] | n/a | 97.5% [94, 99] | 69.8% [62, 76] | 49.0% [39, 59] | 7.0% [3, 14] | 48.1% [42, 54] | 0.6% [0, 3] | 0.85 |
| en | paraphrase | attack_changed=false | | n/a (no rows) | | | | | | | | |
| en | back_translation | all | 100/250 | 0.937 [0.911, 0.960] | 0.985 | 68.4% [62, 74] | 34.0% [28, 40] | 3.0% [1, 8] | 0.0% [0, 4] | 58.6% [53, 64] | 4.0% [2, 7] | 1.05 |
| en | back_translation | attack_changed=false | | n/a (no rows) | | | | | | | | |
| en | back_translation | bt_suspect=true | | n/a (no rows) | | | | | | | | |
| en | back_translation | bt_failed=true | | n/a (no rows) | | | | | | | | |
| hi | clean | all test | 449/923 | 0.961 [0.949, 0.971] | n/a | 81.5% [79, 84] | 38.9% [36, 42] | 4.5% [3, 7] | 0.9% [0, 2] | 55.3% [53, 58] | 0.8% [0, 2] | 1.00 |
| hi | hybrid | all | 150/400 | 0.964 [0.946, 0.979] | 0.970 | 80.5% [76, 84] | 25.2% [21, 30] | 2.7% [1, 7] | 0.7% [0, 4] | 61.1% [57, 65] | 2.2% [1, 4] | 1.00 |
| hi | hybrid | attack_changed=false | 6/13 | n/a | n/a | 69.2% [42, 87] | 23.1% [8, 50] | 0.0% [0, 39] | 0.0% [0, 39] | 52.6% [32, 73] | 0.0% [0, 23] | 1.00 |
| hi | paraphrase | all | 100/250 | 0.873 [0.831, 0.912] | 0.971 | 91.6% [88, 94] | 36.4% [31, 43] | 38.0% [29, 48] | 6.0% [3, 12] | 66.3% [61, 71] | 1.2% [0, 3] | 0.80 |
| hi | paraphrase | machine sources = qwen7b (self-paraphrase) | 100/67 | 0.881 [0.821, 0.933] | n/a | 92.5% [84, 97] | 38.8% [28, 51] | 38.0% [29, 48] | 6.0% [3, 12] | 68.9% [61, 75] | 3.0% [1, 10] | 0.78 |
| hi | paraphrase | machine sources != qwen7b | 100/183 | 0.870 [0.824, 0.910] | n/a | 91.3% [86, 95] | 35.5% [29, 43] | 38.0% [29, 48] | 6.0% [3, 12] | 68.2% [63, 73] | 0.5% [0, 3] | 0.80 |
| hi | paraphrase | attack_changed=false | | n/a (no rows) | | | | | | | | |
| hi | back_translation | all | 100/250 | 0.895 [0.860, 0.929] | 0.971 | 58.4% [52, 64] | 22.4% [18, 28] | 4.0% [2, 10] | 0.0% [0, 4] | 63.4% [58, 68] | 6.0% [4, 10] | 1.00 |
| hi | back_translation | attack_changed=false | 1/1 | n/a | n/a | 0.0% [0, 79] | 0.0% [0, 79] | 0.0% [0, 79] | 0.0% [0, 79] | 50.0% [9, 91] | 0.0% [0, 79] | 1.00 |
| hi | back_translation | bt_suspect=true | 2/0 | n/a | n/a | n/a | n/a | 50.0% [9, 91] | 0.0% [0, 66] | 100.0% [34, 100] | n/a | 0.40 |
| hi | back_translation | bt_failed=true | 1/1 | n/a | n/a | 0.0% [0, 79] | 0.0% [0, 79] | 0.0% [0, 79] | 0.0% [0, 79] | 50.0% [9, 91] | 0.0% [0, 79] | 1.00 |
| te | clean | all test | 397/225 | 0.963 [0.944, 0.980] | n/a | 90.2% [86, 93] | 16.0% [12, 21] | 5.0% [3, 8] | 0.8% [0, 2] | 62.9% [59, 67] | 1.3% [0, 4] | 1.00 |
| te | hybrid | all | 150/225 | 0.965 [0.942, 0.985] | 0.961 | 86.7% [82, 90] | 13.3% [10, 18] | 4.0% [2, 8] | 0.0% [0, 2] | 70.7% [66, 75] | 1.3% [0, 4] | 1.01 |
| te | hybrid | attack_changed=false | 1/6 | n/a | n/a | 83.3% [44, 97] | 16.7% [3, 56] | 100.0% [21, 100] | 0.0% [0, 79] | 71.4% [36, 92] | 16.7% [3, 56] | 1.00 |
| te | paraphrase | all | 100/225 | 0.872 [0.821, 0.917] | 0.951 | 90.2% [86, 93] | 31.6% [26, 38] | 30.0% [22, 40] | 5.0% [2, 11] | 67.4% [62, 72] | 1.3% [0, 4] | 0.67 |
| te | paraphrase | machine sources = qwen7b (self-paraphrase) | 100/164 | 0.884 [0.833, 0.929] | n/a | 90.9% [85, 94] | 32.3% [26, 40] | 30.0% [22, 40] | 5.0% [2, 11] | 67.4% [62, 73] | 0.6% [0, 3] | 0.66 |
| te | paraphrase | machine sources != qwen7b | 100/61 | 0.839 [0.773, 0.895] | n/a | 88.5% [78, 94] | 29.5% [20, 42] | 30.0% [22, 40] | 5.0% [2, 11] | 67.7% [60, 74] | 3.3% [1, 11] | 0.65 |
| te | paraphrase | attack_changed=false | 0/1 | n/a | n/a | 100.0% [21, 100] | 0.0% [0, 79] | n/a | n/a | 100.0% [21, 100] | 0.0% [0, 79] | 1.00 |
| te | back_translation | all | 100/225 | 0.840 [0.794, 0.886] | 0.951 | 53.8% [47, 60] | 12.0% [8, 17] | 7.0% [3, 14] | 0.0% [0, 4] | 73.2% [68, 78] | 7.1% [4, 11] | 1.00 |
| te | back_translation | attack_changed=false | | n/a (no rows) | | | | | | | | |
| te | back_translation | bt_suspect=true | 1/1 | n/a | n/a | 0.0% [0, 79] | 0.0% [0, 79] | 0.0% [0, 79] | 0.0% [0, 79] | 50.0% [9, 91] | 100.0% [21, 100] | 0.91 |
| te | back_translation | bt_failed=true | | n/a (no rows) | | | | | | | | |
| cm | clean | all test | 597/232 | 0.928 [0.911, 0.945] | n/a | 47.4% [41, 54] | 31.5% [26, 38] | 3.4% [2, 5] | 0.8% [0, 2] | 31.1% [28, 34] | 9.5% [6, 14] | 1.00 |
| cm | hybrid | all | 150/232 | 0.844 [0.804, 0.879] | 0.925 | 40.1% [34, 47] | 25.0% [20, 31] | 4.0% [2, 8] | 0.7% [0, 4] | 52.4% [47, 57] | 15.1% [11, 20] | 1.00 |
| cm | hybrid | attack_changed=false | 25/37 | n/a | n/a | 48.6% [33, 64] | 29.7% [17, 46] | 0.0% [0, 13] | 0.0% [0, 13] | 35.5% [25, 48] | 16.2% [8, 31] | 1.00 |
| cm | paraphrase | all | 100/232 | 0.883 [0.841, 0.919] | 0.924 | 46.1% [40, 53] | 26.7% [21, 33] | 4.0% [2, 10] | 1.0% [0, 5] | 54.5% [49, 60] | 9.5% [6, 14] | 0.97 |
| cm | paraphrase | machine sources = qwen7b (self-paraphrase) | 100/98 | 0.860 [0.802, 0.909] | n/a | 31.6% [23, 41] | 17.3% [11, 26] | 4.0% [2, 10] | 1.0% [0, 5] | 52.5% [46, 59] | 10.2% [6, 18] | 0.96 |
| cm | paraphrase | machine sources != qwen7b | 100/134 | 0.900 [0.862, 0.936] | n/a | 56.7% [48, 65] | 33.6% [26, 42] | 4.0% [2, 10] | 1.0% [0, 5] | 47.0% [41, 53] | 9.0% [5, 15] | 0.98 |
| cm | paraphrase | attack_changed=false | 1/1 | n/a | n/a | 0.0% [0, 79] | 0.0% [0, 79] | 0.0% [0, 79] | 0.0% [0, 79] | 100.0% [34, 100] | 0.0% [0, 79] | 1.00 |
| cm | back_translation | all | | n/a (cm not back-translated) | | | | | | | | |

## Heads alone vs fused (AUROC, all rows of each attack)

| bucket | attack | headA | headB | fused |
| --- | --- | --- | --- | --- |
| en | clean | 0.986 | 0.900 | 0.989 |
| en | hybrid | 0.981 | 0.878 | 0.985 |
| en | paraphrase | 0.906 | 0.769 | 0.920 |
| en | back_translation | 0.934 | 0.721 | 0.937 |
| hi | clean | 0.960 | 0.569 | 0.961 |
| hi | hybrid | 0.961 | 0.486 | 0.964 |
| hi | paraphrase | 0.882 | 0.552 | 0.873 |
| hi | back_translation | 0.884 | 0.618 | 0.895 |
| te | clean | 0.973 | 0.235 | 0.963 |
| te | hybrid | 0.969 | 0.274 | 0.965 |
| te | paraphrase | 0.885 | 0.420 | 0.872 |
| te | back_translation | 0.853 | 0.494 | 0.840 |
| cm | clean | 0.928 | 0.647 | 0.928 |
| cm | hybrid | 0.838 | 0.678 | 0.844 |
| cm | paraphrase | 0.881 | 0.578 | 0.883 |

## Caveats

- AUROC intervals are stratified bootstrap (1000 resamples); rates use Wilson intervals, which treat rows as independent (several attacked rows can share a prompt).
- Under `paraphrase` and `back_translation` the label-0 rows are human text that an LLM / MT model then rewrote. Their `ff` is a false flag against the original author, but the text is no longer purely human-written.
- Slices of `attack_changed=false`, `bt_suspect` and `bt_failed` have 0-30 rows (the bt flags 4 and 2 rows in total); their rates are anecdotes, not estimates.
- The attacked rows are a capped, stratified subset of the test split (see src/data/attack.py); `clean_matched_auroc` is the clean score of exactly those source rows.
