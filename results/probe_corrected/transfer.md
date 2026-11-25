### Prompt transfer: one frozen probe scored on both prompts

| Scope | Trained on | AUROC on same prompt | AUROC on other prompt, strict | recentred | Spearman (strict) |
|---|---|---|---|---|--:|
| all | baseline | 0.910 [0.891, 0.927] | 0.896 [0.876, 0.914] | 0.902 [0.882, 0.920] | 0.963 |
| all | intervention | 0.914 [0.897, 0.930] | 0.903 [0.885, 0.920] | 0.905 [0.886, 0.922] | 0.972 |
| arithmetic | baseline | 0.963 [0.948, 0.976] | 0.939 [0.919, 0.957] | 0.959 [0.943, 0.974] | 0.885 |
| arithmetic | intervention | 0.960 [0.944, 0.974] | 0.932 [0.910, 0.951] | 0.957 [0.941, 0.971] | 0.935 |
| factual | baseline | 0.839 [0.804, 0.873] | 0.824 [0.785, 0.858] | 0.831 [0.794, 0.865] | 0.931 |
| factual | intervention | 0.833 [0.798, 0.868] | 0.836 [0.800, 0.871] | 0.841 [0.806, 0.874] | 0.960 |

Baseline-trained probe, questions that called a tool under the baseline prompt; scores are log-odds of need. 'Flipped' = stopped calling under the intervention prompt, 'stayed' = still called.

| Scope | n flipped / stayed | Baseline score, flipped minus stayed | Score change across prompts (recentred), flipped minus stayed | AUROC of baseline score for staying |
|---|---|---|---|--:|
| all | 339 / 599 | -3.11 [-3.42, -2.83] | +0.25 [+0.17, +0.33] | 0.816 |
| arithmetic | 84 / 414 | -5.28 [-5.92, -4.58] | +0.19 [+0.01, +0.37] | 0.897 |
| factual | 255 / 185 | -1.68 [-1.93, -1.42] | +0.01 [-0.07, +0.10] | 0.804 |
