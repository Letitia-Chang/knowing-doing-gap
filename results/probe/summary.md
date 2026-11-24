### Question set and local necessity labels

1000 questions ({'arithmetic': 500, 'factual': 500}); tool_needed = any of N=10 no-tool samples wrong.

| Label rule | overall | arithmetic | factual |
|---|--:|--:|--:|
| any wrong (primary) | 60.8% | 62.6% | 59.0% |
| accuracy < 80% | 56.9% | 60.4% | 53.4% |
| accuracy < 50% | 53.5% | 58.6% | 48.4% |

On the original 30 questions, local-bf16 labels agree with the TogetherAI-Turbo labels on 30/30 (needed: 13 local vs 13 Together).

### Probe: nested-CV out-of-fold performance (95% bootstrap CI over questions)

| Scope | Representation | n (needed) | Nested AUROC | Nested balanced acc. | Shuffled-label AUROC | Surface LR AUROC | Surface GBM AUROC | Layers chosen (count) |
|---|---|--:|--:|--:|--:|--:|--:|---|
| all | last_token | 1000 (608) | 0.900 [0.882, 0.916] | 0.814 [0.789, 0.838] | 0.489 | 0.746 | 0.779 | L24 (4), L25 (4), L21 (3) |
| arithmetic | last_token | 500 (313) | 0.963 [0.948, 0.976] | 0.900 [0.872, 0.926] | 0.469 | 0.866 | 0.917 | L14 (9), L28 (2), L18 (1) |
| factual | last_token | 500 (295) | 0.814 [0.774, 0.852] | 0.752 [0.714, 0.790] | 0.496 | 0.555 | 0.509 | L25 (7), L28 (5), L24 (2) |
| all | mean_pool | 1000 (608) | 0.883 [0.862, 0.901] | 0.800 [0.773, 0.825] | 0.486 | 0.746 | 0.779 | L19 (9), L13 (2), L14 (1) |

Label-threshold sensitivity (pooled scope, last-token probe): tool_needed = accuracy < 80% -> nested AUROC 0.909 [0.891, 0.924]; tool_needed = accuracy < 50% -> nested AUROC 0.906 [0.888, 0.923]

### Probe-predicted need vs actual tool calls (out-of-fold probe, threshold 0.5)

| Prompt | n | Tool-call rate | Needed + called | Needed + NOT called | Not needed + called (over-use) | Not needed + not called |
|---|--:|--:|--:|--:|--:|--:|
| baseline | 1000 | 95.5% | 585 | 23 | 370 | 22 |
| intervention | 1000 | 60.7% | 442 | 166 | 165 | 227 |

| Prompt | Probe cell: knows & does | knows, doesn't | not-needed & does | neither | Probe says NOT needed: over-use items | ...not-needed & not-called items | ...ALL not-needed items | Probe says needed: needed & not-called items |
|---|--:|--:|--:|--:|---|---|---|---|
| baseline | 552 | 18 | 403 | 27 | 81% [77%, 85%] (n=370) | 86% [73%, 100%] (n=22) | 81% [77%, 85%] (n=392) | 65% [43%, 83%] (n=23) |
| intervention | 439 | 122 | 168 | 271 | 75% [68%, 81%] (n=165) | 87% [83%, 92%] (n=227) | 82% [78%, 86%] (n=392) | 56% [48%, 63%] (n=166) |

Probe score vs behaviour: baseline: AUROC(probe, tool_needed)=0.900, AUROC(probe, tool_called)=0.623 (within not-needed only 0.508, within needed only 0.709), ECE=0.043; intervention: AUROC(probe, tool_needed)=0.901, AUROC(probe, tool_called)=0.786 (within not-needed only 0.628, within needed only 0.827), ECE=0.051

### Same questions, two prompts (secondary)

Tool-call rate 95.5% (baseline) -> 60.7% (intervention); 348 questions flipped from calling to not calling. Probe P(needed) across the two prompts: Spearman rho = 0.980, mean |change| = 0.046, mean change on the flipped questions = -0.008.

Run info: model Qwen/Qwen2.5-7B-Instruct, precision bf16, GPU NVIDIA L40S, torch 2.14.1+cu126, transformers 5.24.0.
