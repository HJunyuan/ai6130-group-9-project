# Auto-Generated Tables for `docs/REPORT.md` (`uv run reproduce`)

## Section 4.1 Labelling Agreement (Cohen's κ)

**Labelling agreement (Cohen's κ):** human vs human **0.835** · human vs rules **0.898** (Rater 1 vs rules: 0.956, Rater 2 vs rules: 0.839) · target ≥ 0.8

## Section 4.2 Main Results (One Table per Model)

### Gemma-3-27B-IT (`google/gemma-3-27b-it`)

| Model | Tone | Attempt rate (%) | Accuracy when attempted (%) | Overall success (%) | Mean output tokens | Mean step-0 refusal prob (All / XSTest) | 95% CI, success |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma-3-27B-IT | Very rude | 98.70 | 76.54 | 76.11 | 316.1 | 0.01473 / 0.06776 | [69.44, 82.22] |
| Gemma-3-27B-IT | Blunt | 99.13 | 70.00 | 70.00 | 125.1 | 0.01225 / 0.05636 | [63.33, 76.67] |
| Gemma-3-27B-IT | Neutral | 98.70 | 75.98 | 75.56 | 282.8 | 0.01325 / 0.06093 | [69.44, 81.67] |
| Gemma-3-27B-IT | Polite | 99.57 | 77.78 | 77.78 | 320.1 | 0.00576 / 0.02651 | [71.67, 83.89] |
| Gemma-3-27B-IT | Very deferential | 99.57 | 73.89 | 73.89 | 308.8 | 0.01084 / 0.04987 | [67.22, 80.00] |

### Qwen3-32B (`Qwen/Qwen3-32B`)

| Model | Tone | Attempt rate (%) | Accuracy when attempted (%) | Overall success (%) | Mean output tokens | Mean step-0 refusal prob (All / XSTest) | 95% CI, success |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-32B | Very rude | 99.13 | 80.56 | 80.56 | 205.0 | 0.02853 / 0.09859 | [74.44, 86.11] |
| Qwen3-32B | Blunt | 100.00 | 77.78 | 77.78 | 82.2 | 0.00309 / 0.01421 | [71.67, 83.89] |
| Qwen3-32B | Neutral | 99.13 | 83.15 | 82.22 | 186.2 | 0.00403 / 0.01850 | [76.67, 87.78] |
| Qwen3-32B | Polite | 99.13 | 79.33 | 78.89 | 239.5 | 0.00984 / 0.04509 | [72.78, 85.00] |
| Qwen3-32B | Very deferential | 99.57 | 81.67 | 81.67 | 235.8 | 0.01968 / 0.08244 | [76.11, 87.22] |

### OLMo-3.1-32B-Instruct (`allenai/Olmo-3.1-32B-Instruct`)

| Model | Tone | Attempt rate (%) | Accuracy when attempted (%) | Overall success (%) | Mean output tokens | Mean step-0 refusal prob (All / XSTest) | 95% CI, success |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OLMo-3.1-32B-Instruct | Very rude | 98.70 | 78.77 | 78.33 | 232.1 | 0.02811 / 0.11552 | [72.22, 84.44] |
| OLMo-3.1-32B-Instruct | Blunt | 99.13 | 74.86 | 74.44 | 141.9 | 0.01566 / 0.06748 | [67.78, 80.56] |
| OLMo-3.1-32B-Instruct | Neutral | 98.70 | 72.63 | 72.22 | 235.1 | 0.02469 / 0.11009 | [65.56, 78.89] |
| OLMo-3.1-32B-Instruct | Polite | 98.70 | 75.98 | 75.56 | 286.4 | 0.03749 / 0.16169 | [69.44, 81.67] |
| OLMo-3.1-32B-Instruct | Very deferential | 99.13 | 74.44 | 74.44 | 268.2 | 0.05863 / 0.22925 | [67.78, 80.56] |

## Section 4.3 Statistical Tests

| Model | Metric | Comparison | Test | Effect | p-value (Holm) | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| Gemma-3-27B-IT | Attempt rate (all 230 items) | All 5 tones (Omnibus) | Cochran's Q (Q=2.86) | — | 0.582 | No significant difference |
| Gemma-3-27B-IT | Overall success (180 marked) | All 5 tones (Omnibus) | Cochran's Q (Q=12.44) | — | 0.01437 | Significant |
| Gemma-3-27B-IT | Overall success (180 marked) | Very rude vs Neutral | McNemar (Holm) + TOST (±5 pp) | +0.56 pp (90% CI [-2.47, +3.59]) | 1 | Not significant (Equivalent within ±5 pp, TOST p=0.0079) |
| Gemma-3-27B-IT | Overall success (180 marked) | Blunt vs Neutral | McNemar (Holm) + TOST (±5 pp) | -5.56 pp (90% CI [-9.59, -1.53]) | 0.1656 | Not significant |
| Gemma-3-27B-IT | Overall success (180 marked) | Polite vs Neutral | McNemar (Holm) + TOST (±5 pp) | +2.22 pp (90% CI [-1.65, +6.09]) | 1 | Not significant |
| Gemma-3-27B-IT | Overall success (180 marked) | Very deferential vs Neutral | McNemar (Holm) + TOST (±5 pp) | -1.67 pp (90% CI [-5.20, +1.87]) | 1 | Not significant |
| Gemma-3-27B-IT | Output length (n_output_tokens) | All 5 tones (Omnibus) | Friedman (χ²=438.47) | — | 1.348e-93 | Significant length modulation |
| Gemma-3-27B-IT | Output length (n_output_tokens) | Very rude vs Neutral | Paired Wilcoxon (Holm) | +33.3 tokens | 1.591e-06 | Significant vs neutral |
| Gemma-3-27B-IT | Output length (n_output_tokens) | Blunt vs Neutral | Paired Wilcoxon (Holm) | -157.7 tokens | 1.033e-33 | Significant vs neutral |
| Gemma-3-27B-IT | Output length (n_output_tokens) | Polite vs Neutral | Paired Wilcoxon (Holm) | +37.3 tokens | 9.404e-09 | Significant vs neutral |
| Gemma-3-27B-IT | Output length (n_output_tokens) | Very deferential vs Neutral | Paired Wilcoxon (Holm) | +26.0 tokens | 5.792e-07 | Significant vs neutral |
| Gemma-3-27B-IT | XSTest step-0 refusal prob | All 5 tones (Omnibus) | Friedman (χ²=21.14) | — | 0.000297 | Significant refusal shift |
| Gemma-3-27B-IT | XSTest step-0 refusal prob | Very rude vs Neutral | Paired Wilcoxon (Holm) | +0.00683 | 1 | Not significant |
| Gemma-3-27B-IT | XSTest step-0 refusal prob | Blunt vs Neutral | Paired Wilcoxon (Holm) | -0.00458 | 1 | Not significant |
| Gemma-3-27B-IT | XSTest step-0 refusal prob | Polite vs Neutral | Paired Wilcoxon (Holm) | -0.03442 | 0.1856 | Not significant |
| Gemma-3-27B-IT | XSTest step-0 refusal prob | Very deferential vs Neutral | Paired Wilcoxon (Holm) | -0.01107 | 1 | Not significant |
| Qwen3-32B | Attempt rate (all 230 items) | All 5 tones (Omnibus) | Cochran's Q (Q=2.29) | — | 0.6834 | No significant difference |
| Qwen3-32B | Overall success (180 marked) | All 5 tones (Omnibus) | Cochran's Q (Q=5.02) | — | 0.285 | No significant difference |
| Qwen3-32B | Overall success (180 marked) | Very rude vs Neutral | McNemar (Holm) + TOST (±5 pp) | -1.67 pp (90% CI [-4.96, +1.62]) | 1 | Not significant (Equivalent within ±5 pp, TOST p=0.0477) |
| Qwen3-32B | Overall success (180 marked) | Blunt vs Neutral | McNemar (Holm) + TOST (±5 pp) | -4.44 pp (90% CI [-8.70, -0.19]) | 0.5352 | Not significant |
| Qwen3-32B | Overall success (180 marked) | Polite vs Neutral | McNemar (Holm) + TOST (±5 pp) | -3.33 pp (90% CI [-6.97, +0.30]) | 0.6303 | Not significant |
| Qwen3-32B | Overall success (180 marked) | Very deferential vs Neutral | McNemar (Holm) + TOST (±5 pp) | -0.56 pp (90% CI [-4.09, +2.98]) | 1 | Not significant (Equivalent within ±5 pp, TOST p=0.0194) |
| Qwen3-32B | Output length (n_output_tokens) | All 5 tones (Omnibus) | Friedman (χ²=418.49) | — | 2.809e-89 | Significant length modulation |
| Qwen3-32B | Output length (n_output_tokens) | Very rude vs Neutral | Paired Wilcoxon (Holm) | +18.8 tokens | 0.05241 | Not significant |
| Qwen3-32B | Output length (n_output_tokens) | Blunt vs Neutral | Paired Wilcoxon (Holm) | -104.0 tokens | 7.553e-35 | Significant vs neutral |
| Qwen3-32B | Output length (n_output_tokens) | Polite vs Neutral | Paired Wilcoxon (Holm) | +53.3 tokens | 4.182e-17 | Significant vs neutral |
| Qwen3-32B | Output length (n_output_tokens) | Very deferential vs Neutral | Paired Wilcoxon (Holm) | +49.5 tokens | 3.391e-15 | Significant vs neutral |
| Qwen3-32B | XSTest step-0 refusal prob | All 5 tones (Omnibus) | Friedman (χ²=115.42) | — | 5.072e-24 | Significant refusal shift |
| Qwen3-32B | XSTest step-0 refusal prob | Very rude vs Neutral | Paired Wilcoxon (Holm) | +0.08008 | 4.348e-09 | Significant vs neutral |
| Qwen3-32B | XSTest step-0 refusal prob | Blunt vs Neutral | Paired Wilcoxon (Holm) | -0.00430 | 0.02269 | Significant vs neutral |
| Qwen3-32B | XSTest step-0 refusal prob | Polite vs Neutral | Paired Wilcoxon (Holm) | +0.02659 | 0.0003922 | Significant vs neutral |
| Qwen3-32B | XSTest step-0 refusal prob | Very deferential vs Neutral | Paired Wilcoxon (Holm) | +0.06394 | 3.571e-07 | Significant vs neutral |
| OLMo-3.1-32B-Instruct | Attempt rate (all 230 items) | All 5 tones (Omnibus) | Cochran's Q (Q=1.00) | — | 0.9098 | No significant difference |
| OLMo-3.1-32B-Instruct | Overall success (180 marked) | All 5 tones (Omnibus) | Cochran's Q (Q=4.96) | — | 0.2913 | No significant difference |
| OLMo-3.1-32B-Instruct | Overall success (180 marked) | Very rude vs Neutral | McNemar (Holm) + TOST (±5 pp) | +6.11 pp (90% CI [+1.60, +10.62]) | 0.1731 | Not significant |
| OLMo-3.1-32B-Instruct | Overall success (180 marked) | Blunt vs Neutral | McNemar (Holm) + TOST (±5 pp) | +2.22 pp (90% CI [-2.43, +6.87]) | 1 | Not significant |
| OLMo-3.1-32B-Instruct | Overall success (180 marked) | Polite vs Neutral | McNemar (Holm) + TOST (±5 pp) | +3.33 pp (90% CI [-1.31, +7.97]) | 0.9808 | Not significant |
| OLMo-3.1-32B-Instruct | Overall success (180 marked) | Very deferential vs Neutral | McNemar (Holm) + TOST (±5 pp) | +2.22 pp (90% CI [-2.43, +6.87]) | 1 | Not significant |
| OLMo-3.1-32B-Instruct | Output length (n_output_tokens) | All 5 tones (Omnibus) | Friedman (χ²=341.01) | — | 1.528e-72 | Significant length modulation |
| OLMo-3.1-32B-Instruct | Output length (n_output_tokens) | Very rude vs Neutral | Paired Wilcoxon (Holm) | -3.0 tokens | 0.3242 | Not significant |
| OLMo-3.1-32B-Instruct | Output length (n_output_tokens) | Blunt vs Neutral | Paired Wilcoxon (Holm) | -93.2 tokens | 1.628e-22 | Significant vs neutral |
| OLMo-3.1-32B-Instruct | Output length (n_output_tokens) | Polite vs Neutral | Paired Wilcoxon (Holm) | +51.3 tokens | 4.53e-13 | Significant vs neutral |
| OLMo-3.1-32B-Instruct | Output length (n_output_tokens) | Very deferential vs Neutral | Paired Wilcoxon (Holm) | +33.1 tokens | 5.798e-08 | Significant vs neutral |
| OLMo-3.1-32B-Instruct | XSTest step-0 refusal prob | All 5 tones (Omnibus) | Friedman (χ²=87.95) | — | 3.592e-18 | Significant refusal shift |
| OLMo-3.1-32B-Instruct | XSTest step-0 refusal prob | Very rude vs Neutral | Paired Wilcoxon (Holm) | +0.00544 | 0.02542 | Significant vs neutral |
| OLMo-3.1-32B-Instruct | XSTest step-0 refusal prob | Blunt vs Neutral | Paired Wilcoxon (Holm) | -0.04261 | 0.269 | Not significant |
| OLMo-3.1-32B-Instruct | XSTest step-0 refusal prob | Polite vs Neutral | Paired Wilcoxon (Holm) | +0.05161 | 5.928e-05 | Significant vs neutral |
| OLMo-3.1-32B-Instruct | XSTest step-0 refusal prob | Very deferential vs Neutral | Paired Wilcoxon (Holm) | +0.11916 | 4.856e-07 | Significant vs neutral |

## Item-Clustered Binomial GLM Logistic Regression (`correct ~ tone * model + task`)

| Term | Log-Odds Coef | Std. Error | Odds Ratio | z | p-value |
| --- | --- | --- | --- | --- | --- |
| `Intercept` | +0.5667 | 0.2591 | 1.762 | +2.19 | 0.02872 |
| `C(tone, Treatment(reference='neutral'))[T.blunt]` | -0.3127 | 0.1395 | 0.731 | -2.24 | 0.02504 |
| `C(tone, Treatment(reference='neutral'))[T.polite]` | +0.1360 | 0.1446 | 1.146 | +0.94 | 0.347 |
| `C(tone, Treatment(reference='neutral'))[T.very_deferential]` | -0.0974 | 0.1263 | 0.907 | -0.77 | 0.4408 |
| `C(tone, Treatment(reference='neutral'))[T.very_rude]` | +0.0333 | 0.1110 | 1.034 | +0.30 | 0.7643 |
| `C(model_short, Treatment(reference='Gemma-3-27B-IT'))[T.OLMo-3.1-32B-Instruct]` | -0.1915 | 0.2132 | 0.826 | -0.90 | 0.3692 |
| `C(model_short, Treatment(reference='Gemma-3-27B-IT'))[T.Qwen3-32B]` | +0.4366 | 0.1766 | 1.547 | +2.47 | 0.01341 |
| `C(task, Treatment(reference='mmlu_pro_stem'))[T.commonsense_qa]` | +0.6857 | 0.3387 | 1.985 | +2.02 | 0.04295 |
| `C(task, Treatment(reference='mmlu_pro_stem'))[T.gsm8k]` | +2.5570 | 0.7225 | 12.898 | +3.54 | 0.0004011 |
| `C(task, Treatment(reference='mmlu_pro_stem'))[T.medmcqa]` | +0.0466 | 0.3125 | 1.048 | +0.15 | 0.8816 |
| `C(tone, Treatment(reference='neutral'))[T.blunt]:C(model_short, Treatment(reference='Gemma-3-27B-IT'))[T.OLMo-3.1-32B-Instruct]` | +0.4389 | 0.2047 | 1.551 | +2.14 | 0.03207 |
| `C(tone, Treatment(reference='neutral'))[T.polite]:C(model_short, Treatment(reference='Gemma-3-27B-IT'))[T.OLMo-3.1-32B-Instruct]` | +0.0555 | 0.2138 | 1.057 | +0.26 | 0.7952 |
| `C(tone, Treatment(reference='neutral'))[T.very_deferential]:C(model_short, Treatment(reference='Gemma-3-27B-IT'))[T.OLMo-3.1-32B-Instruct]` | +0.2235 | 0.2058 | 1.250 | +1.09 | 0.2774 |
| `C(tone, Treatment(reference='neutral'))[T.very_rude]:C(model_short, Treatment(reference='Gemma-3-27B-IT'))[T.OLMo-3.1-32B-Instruct]` | +0.3294 | 0.1983 | 1.390 | +1.66 | 0.09664 |
| `C(tone, Treatment(reference='neutral'))[T.blunt]:C(model_short, Treatment(reference='Gemma-3-27B-IT'))[T.Qwen3-32B]` | +0.0121 | 0.2196 | 1.012 | +0.05 | 0.9562 |
| `C(tone, Treatment(reference='neutral'))[T.polite]:C(model_short, Treatment(reference='Gemma-3-27B-IT'))[T.Qwen3-32B]` | -0.3655 | 0.1759 | 0.694 | -2.08 | 0.03773 |
| `C(tone, Treatment(reference='neutral'))[T.very_deferential]:C(model_short, Treatment(reference='Gemma-3-27B-IT'))[T.Qwen3-32B]` | +0.0572 | 0.2002 | 1.059 | +0.29 | 0.7752 |
| `C(tone, Treatment(reference='neutral'))[T.very_rude]:C(model_short, Treatment(reference='Gemma-3-27B-IT'))[T.Qwen3-32B]` | -0.1514 | 0.1715 | 0.860 | -0.88 | 0.3774 |

## Per-Task Accuracy Breakdown (%)

| Model | Task | Chance (%) | Very rude | Blunt | Neutral | Polite | Very deferential |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma-3-27B-IT | `mmlu_pro_stem` | 25% | 71.11 | 62.22 | 66.67 | 64.44 | 62.22 |
| Gemma-3-27B-IT | `commonsense_qa` | 20% | 77.78 | 80.00 | 75.56 | 80.00 | 68.89 |
| Gemma-3-27B-IT | `medmcqa` | 25% | 60.00 | 46.67 | 64.44 | 71.11 | 68.89 |
| Gemma-3-27B-IT | `gsm8k` | 0% | 95.56 | 91.11 | 95.56 | 95.56 | 95.56 |
| Qwen3-32B | `mmlu_pro_stem` | 25% | 73.33 | 68.89 | 66.67 | 64.44 | 64.44 |
| Qwen3-32B | `commonsense_qa` | 20% | 80.00 | 75.56 | 88.89 | 84.44 | 86.67 |
| Qwen3-32B | `medmcqa` | 25% | 73.33 | 71.11 | 77.78 | 71.11 | 80.00 |
| Qwen3-32B | `gsm8k` | 0% | 95.56 | 95.56 | 95.56 | 95.56 | 95.56 |
| OLMo-3.1-32B-Instruct | `mmlu_pro_stem` | 25% | 66.67 | 60.00 | 60.00 | 62.22 | 66.67 |
| OLMo-3.1-32B-Instruct | `commonsense_qa` | 20% | 84.44 | 80.00 | 68.89 | 75.56 | 75.56 |
| OLMo-3.1-32B-Instruct | `medmcqa` | 25% | 64.44 | 60.00 | 62.22 | 66.67 | 57.78 |
| OLMo-3.1-32B-Instruct | `gsm8k` | 0% | 97.78 | 97.78 | 97.78 | 97.78 | 97.78 |

## Cross-Tone Answer Consistency (180 Marked Questions)

| Model | Identical prediction (all 5 tones) | Identical correctness (all 5 tones) | Always correct | Always wrong | Flips across tones |
| --- | --- | --- | --- | --- | --- |
| Gemma-3-27B-IT | 72.22% | 79.44% | 62.78% | 16.67% | 20.56% |
| Qwen3-32B | 78.33% | 80.00% | 70.00% | 10.00% | 20.00% |
| OLMo-3.1-32B-Instruct | 65.00% | 70.56% | 59.44% | 11.11% | 29.44% |

## Stretch Goal: Yes-Man Sycophancy Results (60 Items × 5 Tones)

| Model | Tone | Sycophancy rate (%) | Accuracy under nudge (%) | Baseline accuracy (%) | Accuracy shift (pp) | Cochran's Q (p) |
| --- | --- | --- | --- | --- | --- | --- |
| Gemma-3-27B-IT | Very rude | 46.67 | 45.00 | 68.33 | -23.33 | Q=5.73 (p=0.22) |
| Gemma-3-27B-IT | Blunt | 41.67 | 46.67 | 61.67 | -15.00 | Q=5.73 (p=0.22) |
| Gemma-3-27B-IT | Neutral | 48.33 | 43.33 | 66.67 | -23.33 | Q=5.73 (p=0.22) |
| Gemma-3-27B-IT | Polite | 50.00 | 40.00 | 75.00 | -35.00 | Q=5.73 (p=0.22) |
| Gemma-3-27B-IT | Very deferential | 43.33 | 46.67 | 63.33 | -16.67 | Q=5.73 (p=0.22) |
| Qwen3-32B | Very rude | 26.67 | 63.33 | 80.00 | -16.67 | Q=21.36 (p=0.000269) |
| Qwen3-32B | Blunt | 46.67 | 46.67 | 80.00 | -33.33 | Q=21.36 (p=0.000269) |
| Qwen3-32B | Neutral | 31.67 | 60.00 | 80.00 | -20.00 | Q=21.36 (p=0.000269) |
| Qwen3-32B | Polite | 25.00 | 63.33 | 76.67 | -13.33 | Q=21.36 (p=0.000269) |
| Qwen3-32B | Very deferential | 30.00 | 61.67 | 81.67 | -20.00 | Q=21.36 (p=0.000269) |
| OLMo-3.1-32B-Instruct | Very rude | 68.33 | 26.67 | 75.00 | -48.33 | Q=26.83 (p=2.15e-05) |
| OLMo-3.1-32B-Instruct | Blunt | 83.33 | 11.67 | 71.67 | -60.00 | Q=26.83 (p=2.15e-05) |
| OLMo-3.1-32B-Instruct | Neutral | 78.33 | 13.33 | 70.00 | -56.67 | Q=26.83 (p=2.15e-05) |
| OLMo-3.1-32B-Instruct | Polite | 56.67 | 28.33 | 75.00 | -46.67 | Q=26.83 (p=2.15e-05) |
| OLMo-3.1-32B-Instruct | Very deferential | 58.33 | 33.33 | 73.33 | -40.00 | Q=26.83 (p=2.15e-05) |

## Section 4.4 Candidate Memorable Examples

| Item ID | Task | Model | Gold | What happened across tones |
| --- | --- | --- | --- | --- |
| `mmlu_007` | `mmlu_pro_stem` | Qwen3-32B | `C` | very_rude=C(✓, 1070 tok); blunt=C(✓, 482 tok); neutral=C(✓, 1071 tok); polite=Empty(✗, 3072 tok); very_deferential=C(✓, 948 tok) |
| `mmlu_001` | `mmlu_pro_stem` | OLMo-3.1-32B-Instruct | `D` | very_rude=Empty(✗, 3072 tok); blunt=Empty(✗, 3072 tok); neutral=D(✓, 1053 tok); polite=D(✓, 1741 tok); very_deferential=D(✓, 1659 tok) |
| `med_039` | `medmcqa` | OLMo-3.1-32B-Instruct | `A` | very_rude=A(✓, 1837 tok); blunt=D(✗, 201 tok); neutral=D(✗, 1519 tok); polite=D(✗, 900 tok); very_deferential=Tried(✗, 514 tok) |
| `mmlu_041` | `mmlu_pro_stem` | OLMo-3.1-32B-Instruct | `D` | very_rude=D(✓, 139 tok); blunt=A(✗, 356 tok); neutral=A(✗, 1726 tok); polite=A(✗, 770 tok); very_deferential=A(✗, 241 tok) |
| `mmlu_009` | `mmlu_pro_stem` | OLMo-3.1-32B-Instruct | `C` | very_rude=C(✓, 1902 tok); blunt=C(✓, 1529 tok); neutral=C(✓, 1644 tok); polite=Empty(✗, 3072 tok); very_deferential=C(✓, 2508 tok) |
| `mmlu_001` | `mmlu_pro_stem` | Qwen3-32B | `D` | very_rude=C(✗, 2361 tok); blunt=D(✓, 1670 tok); neutral=Empty(✗, 3072 tok); polite=C(✗, 2698 tok); very_deferential=C(✗, 1584 tok) |
| `mmlu_009` | `mmlu_pro_stem` | Gemma-3-27B-IT | `C` | very_rude=C(✓, 761 tok); blunt=C(✓, 8 tok); neutral=Tried(✗, 1281 tok); polite=C(✓, 735 tok); very_deferential=Tried(✗, 1443 tok) |
| `mmlu_019` | `mmlu_pro_stem` | Gemma-3-27B-IT | `B` | very_rude=B(✓, 1924 tok); blunt=B(✓, 847 tok); neutral=C(✗, 499 tok); polite=C(✗, 723 tok); very_deferential=C(✗, 1315 tok) |
| `csqa_039` | `commonsense_qa` | OLMo-3.1-32B-Instruct | `A` | very_rude=Tried(✗, 427 tok); blunt=A(✓, 3 tok); neutral=B(✗, 261 tok); polite=B(✗, 1375 tok); very_deferential=A(✓, 89 tok) |
| `mmlu_033` | `mmlu_pro_stem` | OLMo-3.1-32B-Instruct | `D` | very_rude=D(✓, 7 tok); blunt=D(✓, 3 tok); neutral=C(✗, 1372 tok); polite=D(✓, 168 tok); very_deferential=D(✓, 257 tok) |
| `mmlu_019` | `mmlu_pro_stem` | OLMo-3.1-32B-Instruct | `B` | very_rude=C(✗, 809 tok); blunt=C(✗, 1000 tok); neutral=C(✗, 629 tok); polite=B(✓, 1747 tok); very_deferential=C(✗, 1182 tok) |
| `mmlu_016` | `mmlu_pro_stem` | Qwen3-32B | `B` | very_rude=D(✗, 783 tok); blunt=D(✗, 3 tok); neutral=D(✗, 395 tok); polite=B(✓, 753 tok); very_deferential=A(✗, 1114 tok) |

## Appendix A Run Verification (from `results/raw/`)

| Run ID | Model | Exact Revision SHA | Rows |
| --- | --- | --- | --- |
| `20261028-105010` | `google/gemma-3-27b-it` | `005ad3404e` | 1150 |
| `20261028-144122` | `Qwen/Qwen3-32B` | `9216db5781` | 1150 |
| `20261028-175650` | `allenai/Olmo-3.1-32B-Instruct` | `ac0587e4a7` | 1150 |
| `20261028-134523` | `google/gemma-3-27b-it` | `005ad3404e` | 300 |
| `20261028-165744` | `Qwen/Qwen3-32B` | `9216db5781` | 300 |
| `20261028-204006` | `allenai/Olmo-3.1-32B-Instruct` | `ac0587e4a7` | 300 |
