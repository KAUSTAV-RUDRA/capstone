# Follow-up experiments: per-bucket Head B weight (exp11) and coverage targets (exp12)

Both run on the v2 normalised scores; the adopted fuser, temperature, tau and `calibration.json` are unchanged.
Reproduce: `python -m experiments.exp11_perbucket_fusion --config configs/models_norm.yaml`, `python -m experiments.exp12_coverage_targets --config configs/models_norm.yaml`.

## 1. Per-bucket Head B weight (RQ2)

Variants, all fit on the train split over standardised [headA, headB]: **global** (adopted; one weight per head + bucket one-hot), **interaction** (headA, headB, bucket one-hot, headA x bucket, headB x bucket), **separate** (one logistic regression per bucket). Interaction and separate give the same test AUROC to three decimals (identical function class up to regularisation). The adopted fuser sees in-sample Head A scores on train, so "OOF" rows refit with 5-fold out-of-fold Head A on train.

Test AUROC:

| bucket | scope | Head A alone | Head B alone | global (adopted) | interaction | separate per bucket | separate, OOF Head A |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cm | all | 0.928 | 0.647 | 0.928 | 0.930 | 0.930 | 0.929 |
| cm | seen | 0.928 | 0.647 | 0.928 | 0.930 | 0.930 | 0.929 |
| en | all | 0.986 | 0.900 | 0.989 | 0.989 | 0.989 | 0.989 |
| en | seen | 0.986 | 0.900 | 0.989 | 0.989 | 0.989 | 0.989 |
| hi | all | 0.960 | 0.569 | 0.961 | 0.960 | 0.960 | 0.957 |
| hi | heldout | 0.947 | 0.785 | 0.953 | 0.950 | 0.950 | 0.940 |
| hi | seen | 0.973 | 0.336 | 0.969 | 0.972 | 0.972 | 0.975 |
| te | all | 0.973 | 0.235 | 0.963 | 0.976 | 0.976 | 0.975 |
| te | seen | 0.973 | 0.235 | 0.963 | 0.976 | 0.976 | 0.975 |

Fused minus Head A alone, with 95% paired-bootstrap CI (global vs separate):

| bucket | scope | global (adopted) | separate per bucket |
| --- | --- | --- | --- |
| cm | all | +0.001 [-0.002, +0.003] | +0.002 [-0.003, +0.008] |
| cm | seen | +0.001 [-0.002, +0.003] | +0.002 [-0.003, +0.007] |
| en | all | +0.003 [-0.000, +0.006] | +0.003 [-0.002, +0.009] |
| en | seen | +0.003 [-0.000, +0.006] | +0.003 [-0.002, +0.008] |
| hi | all | +0.001 [-0.002, +0.004] | +0.001 [-0.001, +0.002] |
| hi | heldout | +0.007 [+0.003, +0.011] | +0.003 [+0.001, +0.005] |
| hi | seen | -0.005 [-0.009, +0.000] | -0.001 [-0.004, +0.001] |
| te | all | -0.009 [-0.018, -0.002] | +0.003 [-0.004, +0.011] |
| te | seen | -0.009 [-0.018, -0.003] | +0.003 [-0.005, +0.010] |

Fitted per-bucket weights (separate fusers, standardised): 

| bucket | headA_weight | headB_weight |
| --- | --- | --- |
| en | 3.595 | 1.962 |
| hi | 3.268 | 0.046 |
| te | 3.688 | -0.480 |
| cm | 2.503 | 0.442 |
| en (OOF Head A) | 3.021 | 2.046 |
| hi (OOF Head A) | 2.714 | -0.084 |
| te (OOF Head A) | 2.898 | -0.621 |
| cm (OOF Head A) | 2.076 | 0.457 |

**Reading: a null for RQ2 as posed.** Per-bucket weights fix the one place the global weight hurt (te: -0.009 -> +0.003) and let Head B take its own sign (en +1.96, te -0.48, hi ~0, cm +0.44), but the gain over Head A alone is at most +0.003 in any bucket and its CI includes zero in every bucket and scope for the separate fuser, except held-out llama hi, where the global weight is +0.007 over Head A (CI [+0.003, +0.011]) and the per-bucket fuser +0.003 (CI [+0.001, +0.005]), a real but tiny gain; the OOF per-bucket fusers are *below* Head A there (-0.007). Out-of-fold Head A does not change any conclusion. So Head B adds nothing measurable beyond Head A on this corpus, with one exception worth stating carefully: in en, where Head B alone is 0.900 and gets a large weight, the gain is +0.003 (CI [-0.002, +0.009] for separate; [0.000, 0.006] for global). The negative-weight te result is a sign-flipped Head B (mGPT inversion), which a per-bucket weight uses only to recover to Head A's level. Whether the fusion "holds under paraphrase where either head alone fails" (contribution #2) is untested here (no T4).

## 2. Abstention at target coverage (only the HUMAN cutoff moves)

tau (alpha 0.01, per bucket) is unchanged, so **FPR and machine recall do not move**: they are fixed by tau (recall en 0.80, hi 0.39, te 0.16, cm 0.31; FPR ~0.01). Raising the HUMAN cutoff t only converts ABSTAIN into HUMAN, and for machine text that is a **false clear**. `oracle` picks t on the test rows (upper bound); `train` picks t on train rows (deployable, but train scores are in-sample so the achieved coverage drifts: te 0.78 at target 0.8, hi 0.83).

| bucket | mode | target | t | coverage | fpr | machine_recall | false_clear | machine_abstained | human_cleared | accuracy_covered |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| en | shipped (cal median) | - | 0.029 | 0.730 | 0.010 | 0.798 | 0.002 | 0.200 | 0.530 | 0.994 |
| en | oracle | 0.7 | 0.027 | 0.701 | 0.010 | 0.798 | 0.002 | 0.200 | 0.420 | 0.994 |
| en | train | 0.7 | 0.027 | 0.704 | 0.010 | 0.798 | 0.002 | 0.200 | 0.433 | 0.994 |
| en | oracle | 0.8 | 0.037 | 0.800 | 0.010 | 0.798 | 0.005 | 0.196 | 0.781 | 0.992 |
| en | train | 0.8 | 0.042 | 0.812 | 0.010 | 0.798 | 0.009 | 0.192 | 0.813 | 0.988 |
| en | oracle | 0.9 | 0.908 | 0.900 | 0.010 | 0.798 | 0.075 | 0.126 | 0.963 | 0.936 |
| en | train | 0.9 | 0.961 | 0.916 | 0.010 | 0.798 | 0.092 | 0.110 | 0.975 | 0.924 |
| hi | shipped (cal median) | - | 0.023 | 0.447 | 0.009 | 0.389 | 0.008 | 0.603 | 0.541 | 0.982 |
| hi | oracle | 0.7 | 0.736 | 0.700 | 0.009 | 0.389 | 0.183 | 0.428 | 0.955 | 0.820 |
| hi | train | 0.7 | 0.753 | 0.704 | 0.009 | 0.389 | 0.186 | 0.425 | 0.960 | 0.818 |
| hi | oracle | 0.8 | 0.945 | 0.800 | 0.009 | 0.389 | 0.321 | 0.290 | 0.978 | 0.727 |
| hi | train | 0.8 | 0.957 | 0.829 | 0.009 | 0.389 | 0.362 | 0.249 | 0.980 | 0.703 |
| hi | oracle | 0.9 | 0.968 | 0.900 | 0.009 | 0.389 | 0.465 | 0.146 | 0.987 | 0.649 |
| hi | train | 0.9 | 0.969 | 0.906 | 0.009 | 0.389 | 0.473 | 0.138 | 0.987 | 0.645 |
| te | shipped (cal median) | - | 0.005 | 0.371 | 0.008 | 0.160 | 0.013 | 0.827 | 0.476 | 0.974 |
| te | oracle | 0.7 | 0.379 | 0.701 | 0.008 | 0.160 | 0.093 | 0.747 | 0.947 | 0.945 |
| te | train | 0.7 | 0.308 | 0.696 | 0.008 | 0.160 | 0.089 | 0.751 | 0.942 | 0.947 |
| te | oracle | 0.8 | 0.965 | 0.801 | 0.008 | 0.160 | 0.302 | 0.538 | 0.985 | 0.857 |
| te | train | 0.8 | 0.948 | 0.778 | 0.008 | 0.160 | 0.249 | 0.591 | 0.980 | 0.878 |
| te | oracle | 0.9 | 0.976 | 0.900 | 0.008 | 0.160 | 0.573 | 0.267 | 0.987 | 0.764 |
| te | train | 0.9 | 0.976 | 0.891 | 0.008 | 0.160 | 0.547 | 0.293 | 0.987 | 0.773 |
| cm | shipped (cal median) | - | 0.133 | 0.689 | 0.008 | 0.315 | 0.095 | 0.591 | 0.789 | 0.953 |
| cm | oracle | 0.7 | 0.144 | 0.701 | 0.008 | 0.315 | 0.103 | 0.582 | 0.802 | 0.950 |
| cm | train | 0.7 | 0.084 | 0.659 | 0.008 | 0.315 | 0.073 | 0.612 | 0.755 | 0.960 |
| cm | oracle | 0.8 | 0.453 | 0.801 | 0.008 | 0.315 | 0.216 | 0.470 | 0.898 | 0.917 |
| cm | train | 0.8 | 0.453 | 0.802 | 0.008 | 0.315 | 0.220 | 0.466 | 0.898 | 0.916 |
| cm | oracle | 0.9 | 0.780 | 0.901 | 0.008 | 0.315 | 0.440 | 0.246 | 0.950 | 0.857 |
| cm | train | 0.9 | 0.763 | 0.893 | 0.008 | 0.315 | 0.418 | 0.267 | 0.946 | 0.862 |

**Reading.** Coverage is not free: it is bought with false clears. en is fine up to 80% coverage (0.2% / 0.5% false clears at 70% / 80%); 90% costs 7.5%. hi at 70% coverage already has 18% of machine text cleared as HUMAN (accuracy on covered 0.82), at 90% 47%. te at 70%: 9% false clears, at 90%: 57%. cm: 10% / 22% / 44% at 70/80/90. The shipped median cutoff (coverage 0.73 / 0.45 / 0.37 / 0.69) keeps false clears at or below 1.3% in every bucket (en stays below 1% up to 80% coverage), and it is conservative because the alpha=0.01 tau leaves most hi/te/cm machine text below it: the limit is recall at tau (0.39 hi, 0.16 te), not the HUMAN cutoff. A usable te/hi deployment needs a higher-recall score (or alpha 0.05), not a lower HUMAN cutoff. Coverage also depends on class mix (en test is 73% machine, te 36%), so a coverage target transfers across populations only through the class-wise rates.

hi, seen vs held-out llama (oracle t): 

| scope | mode | target | coverage | machine_recall | false_clear | machine_abstained |
| --- | --- | --- | --- | --- | --- | --- |
| seen only | shipped (cal median) | - | 0.436 | 0.311 | 0.009 | 0.680 |
| llama only | shipped (cal median) | - | 0.508 | 0.461 | 0.006 | 0.532 |
| seen only | oracle | 0.7 | 0.699 | 0.311 | 0.119 | 0.570 |
| llama only | oracle | 0.7 | 0.830 | 0.461 | 0.242 | 0.296 |
| seen only | oracle | 0.8 | 0.775 | 0.311 | 0.250 | 0.439 |
| llama only | oracle | 0.8 | 0.915 | 0.461 | 0.386 | 0.152 |
| seen only | oracle | 0.9 | 0.881 | 0.311 | 0.455 | 0.234 |
| llama only | oracle | 0.9 | 0.964 | 0.461 | 0.474 | 0.065 |

llama is easier to flag (recall 0.46 vs 0.31) so at the same cutoff it has fewer abstentions but more false clears in absolute terms at high coverage.
