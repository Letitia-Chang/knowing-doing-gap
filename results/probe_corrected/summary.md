### Question set and local necessity labels

982 questions ({'arithmetic': 500, 'factual': 482}); tool_needed = any of N=10 no-tool samples wrong.

| Label rule | overall | arithmetic | factual |
|---|--:|--:|--:|
| any wrong (primary) | 57.8% | 62.6% | 52.9% |
| accuracy < 80% | 54.4% | 60.4% | 48.1% |
| accuracy < 50% | 50.9% | 58.6% | 42.9% |

On the original 30 questions, local-bf16 labels agree with the TogetherAI-Turbo labels on 30/30 (needed: 13 local vs 13 Together).

### Probe: nested-CV out-of-fold performance (95% bootstrap CI over questions)

| Scope | Representation | n (needed) | Nested AUROC | Nested balanced acc. | Shuffled-label AUROC | Surface LR AUROC | Surface GBM AUROC | Layers chosen (count) |
|---|---|--:|--:|--:|--:|--:|--:|---|
| all | last_token | 982 (568) | 0.910 [0.891, 0.927] | 0.832 [0.807, 0.855] | 0.471 | 0.745 | 0.788 | L19 (4), L20 (4), L21 (3) |
| arithmetic | last_token | 500 (313) | 0.963 [0.948, 0.976] | 0.900 [0.872, 0.926] | 0.469 | 0.866 | 0.917 | L14 (9), L28 (2), L18 (1) |
| factual | last_token | 482 (255) | 0.839 [0.804, 0.872] | 0.749 [0.709, 0.788] | 0.464 | 0.605 | 0.556 | L21 (3), L22 (3), L24 (3) |
| all | mean_pool | 982 (568) | 0.899 [0.880, 0.916] | 0.822 [0.797, 0.846] | 0.497 | 0.745 | 0.788 | L19 (5), L13 (3), L18 (2) |
| arithmetic | mean_pool | 500 (313) | 0.957 [0.940, 0.972] | 0.900 [0.875, 0.926] | 0.465 | 0.866 | 0.917 | L14 (11), L16 (2), L13 (1) |
| factual | mean_pool | 482 (255) | 0.810 [0.770, 0.846] | 0.751 [0.710, 0.788] | 0.508 | 0.605 | 0.556 | L19 (6), L18 (4), L20 (4) |

Per-layer out-of-fold AUROC, last-token probe (layer 0 = embeddings; the peak is optimistic because it is picked on the same curve):

| Scope | Layer 0 | Layer 1 | Peak (layer) | Final layer |
|---|--:|--:|--:|--:|
| all | 0.500 | 0.814 | 0.913 (L20) | 0.909 |
| arithmetic | 0.500 | 0.946 | 0.963 (L14) | 0.962 |
| factual | 0.500 | 0.606 | 0.843 (L21) | 0.832 |

Label-threshold sensitivity (pooled scope, last-token probe): tool_needed = accuracy < 80% -> nested AUROC 0.914 [0.896, 0.930]; tool_needed = accuracy < 50% -> nested AUROC 0.919 [0.901, 0.935]

### Probe-predicted need vs actual tool calls (out-of-fold probe, threshold 0.5)

| Prompt | n | Tool-call rate | Needed + called | Needed + NOT called | Not needed + called (over-use) | Not needed + not called |
|---|--:|--:|--:|--:|--:|--:|
| baseline | 982 | 95.5% | 555 | 13 | 383 | 31 |
| intervention | 982 | 61.0% | 434 | 134 | 165 | 249 |

| Prompt | Probe cell: knows & does | knows, doesn't | not-needed & does | neither | Probe says NOT needed: over-use items | ...not-needed & not-called items | ...ALL not-needed items | Probe says needed: needed & not-called items |
|---|--:|--:|--:|--:|---|---|---|---|
| baseline | 519 | 8 | 419 | 36 | 84% [80%, 88%] (n=383) | 94% [84%, 100%] (n=31) | 85% [81%, 88%] (n=414) | 46% [23%, 69%] (n=13) |
| intervention | 439 | 91 | 160 | 292 | 74% [67%, 81%] (n=165) | 91% [88%, 94%] (n=249) | 84% [81%, 88%] (n=414) | 51% [43%, 60%] (n=134) |

Probe score vs behaviour: baseline: AUROC(probe, tool_needed)=0.910, AUROC(probe, tool_called)=0.738 (within not-needed only 0.634, within needed only 0.773), ECE=0.037; intervention: AUROC(probe, tool_needed)=0.914, AUROC(probe, tool_called)=0.822 (within not-needed only 0.682, within needed only 0.845), ECE=0.036

### Same questions, two prompts (secondary)

Tool-call rate 95.5% (baseline) -> 61.0% (intervention); 339 questions flipped from calling to not calling. Probe P(needed) across the two prompts: Spearman rho = 0.985, mean |change| = 0.040, mean change on the flipped questions = -0.001.

Run info: model Qwen/Qwen2.5-7B-Instruct, precision bf16, GPU NVIDIA L40S, torch 2.14.1+cu126, transformers 5.24.0.
