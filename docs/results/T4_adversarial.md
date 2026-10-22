# T4: adversarial robustness (frozen detector)

Frozen fuser + per-bucket temperature + per-bucket conformal thresholds, fit on clean data only; attacked rows are scored, never refit. Reproduce: `python -m experiments.exp06_adversarial --config configs/models_norm.yaml`. Source: `T4_adversarial.csv`.

Columns: AUROC (95% CI); det = machine rows flagged MACHINE at alpha 0.05 / 0.01; ff = human rows flagged MACHINE (false-flag) at alpha 0.05 / 0.01; abstain and false-clear (machine called HUMAN) are for the three-way gate at alpha 0.01 (shipped) and 0.05; len = mean attacked/source word-count ratio. AUROC needs >= 30 rows per class, else n/a.

| bucket | attack | slice | n h/m | AUROC [95% CI] | clean-matched AUROC | det 5% | det 1% | ff 5% | ff 1% | abstain 1% | abstain 5% | false-clear 1% | false-clear 5% | len |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| en | clean | all test | 402.0/1091.0 | 0.989 [0.984, 0.993] | n/a | 94.6% | 79.8% | 5.2% | 1.0% | 27.0% | 15.1% | 0.2% | 0.2% | 1.00 |
| en | hybrid | all | 150.0/400.0 | 0.985 [0.976, 0.994] | 0.990 | 93.2% | 72.5% | 4.0% | 1.3% | 30.2% | 14.4% | 0.2% | 0.2% | 1.00 |
| en | hybrid | attack_changed=false | 0.0/1.0 | n/a | n/a | 100.0% | 0.0% | n/a | n/a | 100.0% | 0.0% | 0.0% | 0.0% | 1.00 |
| en | paraphrase | all | 100.0/250.0 | 0.920 [0.890, 0.949] | 0.985 | 96.8% | 69.6% | 49.0% | 7.0% | 43.7% | 12.3% | 0.4% | 0.4% | 0.85 |
| en | paraphrase | machine sources = qwen7b (self-paraphrase) | 100.0/88.0 | 0.905 [0.862, 0.944] | n/a | 95.5% | 69.3% | 49.0% | 7.0% | 55.9% | 21.3% | 0.0% | 0.0% | 0.82 |
| en | paraphrase | machine sources != qwen7b | 100.0/162.0 | 0.928 [0.894, 0.955] | n/a | 97.5% | 69.8% | 49.0% | 7.0% | 48.1% | 14.9% | 0.6% | 0.6% | 0.85 |
| en | paraphrase | attack_changed=false | | n/a (no rows) | | | | | | | | | | |
| en | back_translation | all | 100.0/250.0 | 0.937 [0.911, 0.960] | 0.985 | 68.4% | 34.0% | 3.0% | 0.0% | 58.6% | 33.1% | 4.0% | 4.0% | 1.05 |
| en | back_translation | attack_changed=false | | n/a (no rows) | | | | | | | | | | |
| en | back_translation | bt_suspect=true | | n/a (no rows) | | | | | | | | | | |
| en | back_translation | bt_failed=true | | n/a (no rows) | | | | | | | | | | |
| hi | clean | all test | 449.0/923.0 | 0.961 [0.949, 0.971] | n/a | 81.5% | 38.9% | 4.5% | 0.9% | 55.3% | 25.5% | 0.8% | 0.8% | 1.00 |
| hi | hybrid | all | 150.0/400.0 | 0.964 [0.946, 0.979] | 0.970 | 80.5% | 25.2% | 2.7% | 0.7% | 61.1% | 20.4% | 2.2% | 2.2% | 1.00 |
| hi | hybrid | attack_changed=false | 6.0/13.0 | n/a | n/a | 69.2% | 23.1% | 0.0% | 0.0% | 52.6% | 21.1% | 0.0% | 0.0% | 1.00 |
| hi | paraphrase | all | 100.0/250.0 | 0.873 [0.831, 0.912] | 0.971 | 91.6% | 36.4% | 38.0% | 6.0% | 66.3% | 17.7% | 1.2% | 1.2% | 0.80 |
| hi | paraphrase | machine sources = qwen7b (self-paraphrase) | 100.0/67.0 | 0.881 [0.821, 0.933] | n/a | 92.5% | 38.8% | 38.0% | 6.0% | 68.9% | 28.1% | 3.0% | 3.0% | 0.78 |
| hi | paraphrase | machine sources != qwen7b | 100.0/183.0 | 0.870 [0.824, 0.910] | n/a | 91.3% | 35.5% | 38.0% | 6.0% | 68.2% | 20.8% | 0.5% | 0.5% | 0.80 |
| hi | paraphrase | attack_changed=false | | n/a (no rows) | | | | | | | | | | |
| hi | back_translation | all | 100.0/250.0 | 0.895 [0.860, 0.929] | 0.971 | 58.4% | 22.4% | 4.0% | 0.0% | 63.4% | 36.6% | 6.0% | 6.0% | 1.00 |
| hi | back_translation | attack_changed=false | 1.0/1.0 | n/a | n/a | 0.0% | 0.0% | 0.0% | 0.0% | 50.0% | 50.0% | 0.0% | 0.0% | 1.00 |
| hi | back_translation | bt_suspect=true | 2.0/0.0 | n/a | n/a | n/a | n/a | 50.0% | 0.0% | 100.0% | 50.0% | n/a | n/a | 0.40 |
| hi | back_translation | bt_failed=true | 1.0/1.0 | n/a | n/a | 0.0% | 0.0% | 0.0% | 0.0% | 50.0% | 50.0% | 0.0% | 0.0% | 1.00 |
| te | clean | all test | 397.0/225.0 | 0.963 [0.944, 0.980] | n/a | 90.2% | 16.0% | 5.0% | 0.8% | 62.9% | 33.3% | 1.3% | 1.3% | 1.00 |
| te | hybrid | all | 150.0/225.0 | 0.965 [0.942, 0.985] | 0.961 | 86.7% | 13.3% | 4.0% | 0.0% | 70.7% | 25.1% | 1.3% | 1.3% | 1.01 |
| te | hybrid | attack_changed=false | 1.0/6.0 | n/a | n/a | 83.3% | 16.7% | 100.0% | 0.0% | 71.4% | 0.0% | 16.7% | 16.7% | 1.00 |
| te | paraphrase | all | 100.0/225.0 | 0.872 [0.821, 0.917] | 0.951 | 90.2% | 31.6% | 30.0% | 5.0% | 67.4% | 19.1% | 1.3% | 1.3% | 0.67 |
| te | paraphrase | machine sources = qwen7b (self-paraphrase) | 100.0/164.0 | 0.884 [0.833, 0.929] | n/a | 90.9% | 32.3% | 30.0% | 5.0% | 67.4% | 21.6% | 0.6% | 0.6% | 0.66 |
| te | paraphrase | machine sources != qwen7b | 100.0/61.0 | 0.839 [0.773, 0.895] | n/a | 88.5% | 29.5% | 30.0% | 5.0% | 67.7% | 29.8% | 3.3% | 3.3% | 0.65 |
| te | paraphrase | attack_changed=false | 0.0/1.0 | n/a | n/a | 100.0% | 0.0% | n/a | n/a | 100.0% | 0.0% | 0.0% | 0.0% | 1.00 |
| te | back_translation | all | 100.0/225.0 | 0.840 [0.794, 0.886] | 0.951 | 53.8% | 12.0% | 7.0% | 0.0% | 73.2% | 42.2% | 7.1% | 7.1% | 1.00 |
| te | back_translation | attack_changed=false | | n/a (no rows) | | | | | | | | | | |
| te | back_translation | bt_suspect=true | 1.0/1.0 | n/a | n/a | 0.0% | 0.0% | 0.0% | 0.0% | 50.0% | 50.0% | 100.0% | 100.0% | 0.91 |
| te | back_translation | bt_failed=true | | n/a (no rows) | | | | | | | | | | |
| cm | clean | all test | 597.0/232.0 | 0.928 [0.911, 0.945] | n/a | 47.4% | 31.5% | 3.4% | 0.8% | 31.1% | 24.8% | 9.5% | 9.5% | 1.00 |
| cm | hybrid | all | 150.0/232.0 | 0.844 [0.804, 0.879] | 0.925 | 40.1% | 25.0% | 4.0% | 0.7% | 52.4% | 41.9% | 15.1% | 15.1% | 1.00 |
| cm | hybrid | attack_changed=false | 25.0/37.0 | n/a | n/a | 48.6% | 29.7% | 0.0% | 0.0% | 35.5% | 24.2% | 16.2% | 16.2% | 1.00 |
| cm | paraphrase | all | 100.0/232.0 | 0.883 [0.841, 0.919] | 0.924 | 46.1% | 26.7% | 4.0% | 1.0% | 54.5% | 40.1% | 9.5% | 9.5% | 0.97 |
| cm | paraphrase | machine sources = qwen7b (self-paraphrase) | 100.0/98.0 | 0.860 [0.802, 0.909] | n/a | 31.6% | 17.3% | 4.0% | 1.0% | 52.5% | 43.9% | 10.2% | 10.2% | 0.96 |
| cm | paraphrase | machine sources != qwen7b | 100.0/134.0 | 0.900 [0.862, 0.936] | n/a | 56.7% | 33.6% | 4.0% | 1.0% | 47.0% | 32.5% | 9.0% | 9.0% | 0.98 |
| cm | paraphrase | attack_changed=false | 1.0/1.0 | n/a | n/a | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 100.0% | 0.0% | 0.0% | 1.00 |
| cm | back_translation | all | | n/a (cm not back-translated) | | | | | | | | | | |

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

- Rates (det, ff, abstain, false-clear) carry no CI; with 100-150 human rows per cell a false-flag rate has a 95% half-width of roughly 4-10 points. Only AUROC is bootstrapped.
- Under `paraphrase` and `back_translation` the label-0 rows are human text that an LLM / MT model then rewrote. Their `ff` is a false flag against the original author, but the text is no longer purely human-written.
- Slices of `attack_changed=false`, `bt_suspect` and `bt_failed` have 0-30 rows (the bt flags 4 and 2 rows in total); their rates are anecdotes, not estimates.
- The attacked rows are a capped, stratified subset of the test split (see src/data/attack.py); `clean_matched_auroc` is the clean score of exactly those source rows.
