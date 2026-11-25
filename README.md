# The Knowing-Doing Gap: Does an LLM Know It Needs a Tool?

An experiment measuring the gap between what an LLM *believes* about its own capability and what it *does* when tools are available. It has two parts, both adaptations of the paper's tool-necessity setup on Qwen2.5-7B-Instruct:

1. **Behavioral experiment** (30 questions, black-box through TogetherAI): does the model call a tool when it needs one, and only then?
2. **Hidden-state probe extension** (1,000 questions, local bf16 run with access to activations): does the model's internal representation encode "this question needs a tool", and does its behavior follow it? See [Hidden-state probe (extension)](#hidden-state-probe-extension).

Based on: Cheng, Fan, JafariRaviz, Rezaei, Feizi — ["Model-Adaptive Tool Necessity Reveals the Knowing-Doing Gap in LLM Tool Use"](https://arxiv.org/abs/2605.14038), 2026.

The experiment tests whether an LLM with access to tools can decide when to answer from memory and when to call a tool. It separates three signals for each question:

1. **Model-adaptive tool necessity**: whether Qwen2.5 can answer the question reliably without tools across repeated no-tool runs.
2. **Cognition**: whether the model reports low confidence answering without a tool.
3. **Action**: whether the model actually calls a calculator or search tool in an agentic setting.

Each question is mapped into one of four cells:

| Necessity      | Action            | Interpretation                    |
| -------------- | ----------------- | --------------------------------- |
| Needed tool    | Called tool       | Aligned behavior                  |
| No tool needed | Did not call tool | Aligned behavior                  |
| Needed tool    | Did not call tool | Tool underuse / knowing-doing gap |
| No tool needed | Called tool       | Tool over-reliance                |

## Results at a glance

**Behavioral experiment** (30 questions, TogetherAI):

| Metric                         | Baseline | Intervention |
| ------------------------------ | -------: | ------------: |
| Tool call rate                 |    93.3% |         56.7% |
| Necessity-action mismatch rate |    50.0% |         26.7% |
| Action accuracy                |    93.3% |         83.3% |

Qwen2.5-7B-Instruct calls a tool on 28 of 30 questions — including most of the ones it can already answer correctly on its own. That's the opposite failure mode from the paper's under-trained models (which *know* they need a tool but fail to call it): here the model **over-relies** on tools it doesn't need. A one-line system-prompt intervention ("use a tool only when not fully certain") nearly halves the mismatch rate, at the cost of two questions flipping into true tool-underuse. See [`results/summary_metrics.csv`](results/summary_metrics.csv) and the figures below for the full breakdown.

<p>
  <img src="results/figures/baseline_four_cell_outcomes.png" width="45%" alt="Baseline four-cell outcomes">
  <img src="results/figures/baseline_vs_intervention_mismatch.png" width="45%" alt="Baseline vs intervention mismatch rate">
</p>

**Hidden-state probe** (1,000 questions, local bf16; every number is from [`results/probe/summary.md`](results/probe/summary.md)):

| | Result |
|---|---|
| Linear probe on the last prompt token, predicting tool necessity (nested CV AUROC, 95% CI) | 0.900 [0.882, 0.916]; shuffled labels 0.489; best surface-feature baseline 0.779 |
| Arithmetic / factual QA separately | 0.963 vs best surface baseline 0.917 / 0.814 vs 0.555 |
| Tool-call rate, baseline prompt → intervention prompt | 95.5% → 60.7% |
| Probe's P(needed) across the two prompts on the same questions | Spearman 0.980, while 348 questions flipped from calling to not calling |

In short: tool necessity is linearly decodable from the model's hidden state, and the decoded signal barely changes when a prompt change shifts behavior. For arithmetic, much of the signal is already readable from the question's surface form; for factual QA the probe is well above surface baselines. A probe shows what is decodable, not what the model uses. See the [full results and caveats](#results).

<p>
  <img src="results/probe/figures/probe_layer_curve.png" width="100%" alt="Probe AUROC by layer with shuffled-label and surface-feature baselines">
</p>

## How this differs from the paper

- **Behavioral experiment: black-box only.** It measures behavior through the API (repeated no-tool runs, self-reported confidence, actual tool calls) and does no hidden-state probing.
- **Hidden-state extension: a partial adaptation.** It trains linear probes for tool necessity on a locally run model's hidden states, but it does **not** reproduce the paper's action probe, its necessity/action direction-similarity analysis, or its per-sample stage attribution, so it does not test the paper's orthogonality claim. See [what is replicated, adapted, and new](#what-is-replicated-adapted-and-new).
- **One model, small question sets.** 30 questions (behavioral) and 1,000 questions (probe extension) on a single model, versus the paper's larger multi-model study.
- **Own intervention.** The one-line system-prompt intervention is my addition, not part of the original paper.

## Repository structure

```text
knowing-doing-gap/
  data/
    questions.json          # 30 curated questions: 15 arithmetic + 15 factual QA
    questions_probe.json    # Probe extension: 1,000 questions (the 30 above + generated arithmetic + TriviaQA)
    mock_search.json        # Mock search results for factual questions
  deliverables/
    question_set.xlsx       # Human-readable question-set table
  src/
    model_client.py         # TogetherAI OpenAI-compatible client
    label_necessity.py      # Repeated no-tool runs for model-adaptive necessity
    run_cognition.py        # Confidence-based cognition probe
    run_action.py           # Agentic tool-use action probe + intervention
    analyze_results.py      # Merge results, compute metrics, generate figures
    graders.py              # Lightweight answer grading helpers
    tools.py                # Calculator + mock search tools
    io_utils.py             # Shared JSON loading/writing helpers
    probe/                  # Hidden-state probe extension (see "Hidden-state probe (extension)")
      build_questions.py    # Builds data/questions_probe.json (needs `datasets`; output is committed)
      question_gen.py       # Arithmetic generators + factual selection (pure Python)
      label_necessity_local.py  # Stage 2: necessity labels with the local model (GPU)
      run_action_local.py   # Stage 3: tool-call decision with the local model (GPU)
      extract_activations.py    # Stage 4: hidden states for every layer (GPU)
      run_probe.py          # Stage 5: nested-CV probes + controls (CPU)
      knowing_doing.py      # Stage 6: probe-predicted need vs actual tool calls (CPU)
      make_figures.py       # Stage 7a: figures from saved results
      summarize.py          # Stage 7b: writes results/probe/summary.{json,md}
      common.py, cv.py, hf_runtime.py   # Shared logic, CV/statistics, HF model helpers
  tests/
    test_graders.py         # Regression tests for answer grading
    test_tools.py           # Regression tests for the calculator/search tools
    test_run_action.py      # Regression tests for the tool-call JSON parser
    test_probe_*.py         # Probe extension tests (question generation, labels, CV/leakage, CPU pipeline); no GPU needed
  run_probe.sh              # Probe extension runner: ./run_probe.sh [gpu|cpu|all]
  requirements-probe.txt    # Extra dependencies for the probe extension
  results/
    metadata.json
    questions_labeled.json
    summary_metrics.csv
    merged_results.csv
    four_cell_baseline.csv
    four_cell_intervention.csv
    figures/
      baseline_four_cell_outcomes.png
      baseline_mismatch_by_category.png
      intervention_four_cell_outcomes.png
      baseline_vs_intervention_mismatch.png
    probe/                  # Probe extension outputs: summary.md, figures/, CSVs (activation .npz files are gitignored)
  run_experiment.sh         # End-to-end runner; writes to results/ by default
  requirements.txt
  .env.example
```

The included `results/` folder contains the review-ready outputs from the final Qwen2.5-7B-Instruct run. Large raw/intermediate model traces are intentionally excluded from the submitted package.

To preserve the included final results, run sanity checks into a separate folder such as `results_sanity/`.

## Setup

```bash
cd knowing-doing-gap
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Then edit `.env`:

```bash
TOGETHER_API_KEY=your_real_key
TOGETHER_MODEL=Qwen/Qwen2.5-7B-Instruct-Turbo
```

Do **not** commit `.env`.

## Running tests

The grading logic, tool executors, and the tool-call JSON parser have unit/regression tests (no API key required):

```bash
python3 -m pytest tests/
```

## Run the full experiment

The default runner writes directly to `results/`:

```bash
./run_experiment.sh
```

This runs:

```bash
python3 src/label_necessity.py --questions data/questions.json --outdir results --n 10 --temp 0.7
python3 src/run_cognition.py --questions results/questions_labeled.json --outdir results
python3 src/run_action.py --questions results/questions_labeled.json --outdir results
python3 src/run_action.py --questions results/questions_labeled.json --outdir results --intervention
python3 src/analyze_results.py --run-dir results
```

## Optional sanity checks

Run the first six questions into `results/`:

```bash
./run_experiment.sh results --limit 6
```

Or run a mixed subset by question ID:

```bash
./run_experiment.sh results --ids A01,A04,A08,A11,A14,F01,F04,F08,F12,F15
```

For a true sanity check, it is safer to change the output folder so the included final results are not overwritten:

```bash
./run_experiment.sh results_sanity --limit 6
```

## Final clean-run results

The included clean run used:

```text
Model: Qwen/Qwen2.5-7B-Instruct-Turbo
Provider: TogetherAI
Necessity labeling: N = 10 no-tool runs, temperature = 0.7
Question set: 30 total, 15 arithmetic + 15 factual QA
```

See [Results at a glance](#results-at-a-glance) above for the headline numbers. The full per-question breakdown is in [`results/merged_results.csv`](results/merged_results.csv), and the four-cell counts are in [`results/four_cell_baseline.csv`](results/four_cell_baseline.csv) / [`results/four_cell_intervention.csv`](results/four_cell_intervention.csv).

## Limitations

- **Grading is string-based**, not semantic: `graders.py` checks whether a normalized reference string appears in the model's answer. This is reliable for short numeric/factual answers but would need a more tolerant matcher (or an LLM judge) for open-ended questions.
- **Necessity is empirically estimated, not certain**: a question is labeled `tool_needed=True` if the model got even 1 of 10 no-tool runs wrong. A single unlucky sample at temperature 0.7 can flip a borderline question's label; increasing `--n` would tighten this at the cost of more API calls.
- **The search tool is mocked** (`data/mock_search.json`), so the action stage measures the *decision* to search, not whether a live search would have produced a useful result.
- **Single-model run**: the committed `results/` folder reflects one model (Qwen2.5-7B-Instruct-Turbo). Findings about over- vs under-reliance are specific to this model's instruction tuning and shouldn't be generalized without re-running the pipeline on other models.

## Notes

- The action stage uses a manual JSON tool protocol instead of native function calling. This makes the experiment more reliable for hosted open models that may not follow native tool-call formats consistently.
- The factual QA tool is a mock search tool, not a live web search API. This keeps the experiment reproducible.
- Re-running the pipeline will recreate intermediate files such as `action_baseline.json`, `action_intervention.json`, and `cognition.json` inside the selected output folder. These files are needed by the analysis script, but they are not included in the submitted `results/` folder because the final tables and figures are enough for review.

## Hidden-state probe (extension)

> Every number in this section is generated by `./run_probe.sh` into [`results/probe/summary.md`](results/probe/summary.md) and `summary.json`, and copied from there. The run used Qwen2.5-7B-Instruct in bf16 on one NVIDIA L40S.

The behavioral experiment above treats the model as a black box (TogetherAI gives no access to activations). This extension asks the question the paper answers with hidden states: **does Qwen2.5-7B-Instruct's internal representation encode "this question needs a tool" even when its behavior does not follow?** That is the knowing-doing gap at the representation level.

This is an **adaptation of the probing part of Cheng et al. ([arXiv:2605.14038](https://arxiv.org/abs/2605.14038)), not a new method.** The paper is the source of the necessity definition, the cognition/action decomposition and the idea of linear probes on hidden states. The table below states what is replicated, what is adapted, and what I added.

### What is replicated, adapted, and new

| | What | How it relates to the paper |
|---|---|---|
| **Replicated** | Model-adaptive necessity: N = 10 no-tool samples at temperature 0.7, `tool_needed` = any sample wrong | Same definition (paper §3, §4.1), recomputed with the local model |
| **Replicated** | Linear probes for tool necessity on hidden states | Same idea (paper §5.1) |
| **Replicated** | Greedy decoding for the tool-call decision; necessity × action cells | Same design (paper §4.2, Table 1) |
| **Adapted** | Model | Qwen2.5-7B-Instruct run locally; the paper uses Qwen3-8B, Qwen3-4B, Llama-3.1-8B-Instruct and Llama-3.2-3B-Instruct |
| **Adapted** | Arithmetic data | The paper's 13 problem families and sampling shares (Appendix A, Table 2), re-implemented from that description in `src/probe/question_gen.py`, at a much smaller scale than the paper's 4,000 problems |
| **Adapted** | Factual data | TriviaQA (`rc.nocontext`) sample plus the original 15 questions; the paper uses TruthfulQA |
| **Adapted** | Probe training and metrics | scikit-learn L2 logistic regression with the regularization strength tuned by inner CV, reported as AUROC and balanced accuracy; the paper trains an Adam-optimized linear classifier on a 70/30 split and reports MCC |
| **Adapted** | Token positions | Last prompt token only (the decision point, "t = −1"), plus a mean-pooled comparison; the paper sweeps the last 20 token positions × every layer |
| **Adapted** | Tool protocol | The repo's manual JSON protocol (same as `src/run_action.py`); the paper uses model-specific native tool-call handlers. Only the first-turn decision is analysed; tools are not executed |
| **New (my additions)** | Controls and statistics | Shuffled-label probes, surface-feature baselines (question length, digit count, category), per-category probes, nested-CV layer selection, bootstrap confidence intervals |
| **New (my additions)** | Probe vs behavior analysis | Whether the probe says "not needed" on the questions where the model over-uses tools, a calibration plot, and the same analysis under the intervention prompt as a secondary condition |
| **New (my additions)** | Sensitivity checks | Alternative necessity thresholds (accuracy < 80% and < 50%) and the mean-pooled representation |
| **Not replicated** | The paper's central geometric claim | I do **not** train an action probe or compare necessity and action directions (paper §5.2–5.3), and I do not reproduce the per-sample stage attribution (§5.4). Nothing here tests whether the two directions are orthogonal |
| **Not implemented** | Steering test | Planned as optional and exploratory; not implemented |

### Question set

`data/questions_probe.json` has 1,000 questions, 500 per category. The original 30 are kept verbatim with their IDs, so the behavioral results stay comparable. The other 970 have neutral IDs (`Q0001`, …).

- **Arithmetic:** generated with a fixed seed from the paper's problem families (easy, larger-short and multi-step groups); gold answers are computed with the repo's AST-whitelisted calculator.
- **Factual:** sampled from TriviaQA. Gold answers come from the dataset and are **not independently verified by me**. TriviaQA's alias lists are noisy, so an alias is accepted only if it shares a content word with the canonical answer, and answers are matched as whole tokens. The run writes `results/probe/gold_audit.csv`, listing the factual questions where all 10 no-tool samples were graded wrong, so that a person can spot-check them for bad gold answers.
- Rebuild with `python src/probe/build_questions.py` (needs `pip install datasets`). The output is committed, so you only need this to change the question set.

### Pipeline

| Stage | Script | Needs | Output (in `results/probe/`) |
|---|---|---|---|
| 1 | `build_questions.py` | internet, `datasets` | `data/questions_probe.json` (committed) |
| 2 | `label_necessity_local.py` | GPU | `questions_probe_labeled.json` (adds accuracy fraction and `tool_needed` at three thresholds), `necessity_runs_local.json`, `gold_audit.csv` |
| 3 | `run_action_local.py` | GPU | `action_local_baseline.json`, `action_local_intervention.json` |
| 4 | `extract_activations.py` | GPU | `activations.npz`, `activations_intervention.npz` (layers × questions × hidden; last-token and mean-pooled) |
| 5 | `run_probe.py` | CPU | `probe_layer_curve*.csv`, `probe_nested*.json`, `oof_*.csv` |
| 6 | `knowing_doing.py` | CPU | `knowing_doing*.json`, `knowing_doing_cells*.csv`, `calibration*.csv` |
| 7 | `make_figures.py`, `summarize.py` | CPU | `figures/`, `summary.json`, `summary.md` |

Both the baseline prompt and the intervention prompt are imported from `src/run_action.py`, and the necessity prompt from `src/label_necessity.py`, so the probe study sees the same decision context as the original behavioral experiment (a test checks this). Activations are taken at the last token of the templated prompt, the position that produces the first generated token. Each transformer layer plus the embedding layer is saved.

### How the probes are evaluated

- **Probe:** L2 logistic regression with standardized inputs and class-balanced weights, predicting `tool_needed` (the label is imbalanced, so metrics are AUROC and balanced accuracy).
- **Nested cross-validation:** the outer loop is repeated stratified 5-fold. In each outer training set, an inner CV chooses the regularization strength for each layer and ranks the layers. The layer with the best inner score is used to predict the held-out fold. So the layer is never chosen using the questions it is scored on, and the "best layer" numbers are not selected on test data. The per-layer curve is shown alongside, but its peak is optimistic and is not used as the headline number.
- **Confidence intervals:** 95% percentile bootstrap over questions on the out-of-fold predictions averaged across repeats. CV folds share training data, so fold-to-fold spread would give intervals that are too narrow.
- **Controls:** (1) the same pipeline on shuffled labels (chance), (2) logistic-regression and gradient-boosting models on surface features only (question length, word count, digit count, number count, longest number, operator count, category), and (3) a category-only baseline, so the probe is compared with what question type alone predicts.
- **Per-category probes:** arithmetic and factual separately. A scope is skipped, and reported as skipped, if its minority class is too small.
- **Probe vs behavior (Step 6):** each question's probe prediction is its own out-of-fold prediction, so nothing is scored on data it was selected on. Cells are probe belief × actual call: *knows & does*, *knows, doesn't*, *probe says not needed & does* (over-use), and *neither*. The key question is whether, on questions where the model calls a tool it does not need, the probe says "not needed". If so, the over-use looks like a behavior/policy issue rather than missing knowledge.

### Running it

```bash
pip install -r requirements.txt -r requirements-probe.txt
./run_probe.sh gpu    # stages 2-4 (needs a GPU)
./run_probe.sh cpu    # stages 5-7 (CPU only)
./run_probe.sh        # everything
```

Environment overrides: `PRECISION` (`bf16`, default, or `8bit`), `MODEL`, `OUT`, `LIMIT` (use only the first N questions, for smoke tests), `N_JOBS`, `SEED`.

**Hardware:** the 7B model in bf16 does not fit on a 16 GB GPU (the weights alone are about 15 GB, and a 16 GB V100 ran out of memory); a GPU with 24 GB or more is recommended. `PRECISION=8bit` is a fallback; it is recorded in `results/probe/run_info.json` and must be reported as 8-bit, not bf16. The activation files are large (hundreds of MB), so they are gitignored; regenerate them with `./run_probe.sh gpu`. All other outputs are committed.

**Tests** (no GPU, no model download):

```bash
python -m pytest tests/
```

They cover question generation, labeling thresholds, the tool-call decision parser, prompt equality with `run_action.py`, split and leakage logic (changing the labels of a held-out fold must not change layer selection or predictions), and a synthetic end-to-end run of every CPU stage.

### Results

The tables are [`results/probe/summary.md`](results/probe/summary.md), copied verbatim.

#### Question set and local necessity labels

1000 questions ({'arithmetic': 500, 'factual': 500}); tool_needed = any of N=10 no-tool samples wrong.

| Label rule | overall | arithmetic | factual |
|---|--:|--:|--:|
| any wrong (primary) | 60.8% | 62.6% | 59.0% |
| accuracy < 80% | 56.9% | 60.4% | 53.4% |
| accuracy < 50% | 53.5% | 58.6% | 48.4% |

On the original 30 questions, local-bf16 labels agree with the TogetherAI-Turbo labels on 30/30 (needed: 13 local vs 13 Together).

#### Probe: nested-CV out-of-fold performance (95% bootstrap CI over questions)

| Scope | Representation | n (needed) | Nested AUROC | Nested balanced acc. | Shuffled-label AUROC | Surface LR AUROC | Surface GBM AUROC | Layers chosen (count) |
|---|---|--:|--:|--:|--:|--:|--:|---|
| all | last_token | 1000 (608) | 0.900 [0.882, 0.916] | 0.814 [0.789, 0.838] | 0.489 | 0.746 | 0.779 | L24 (4), L25 (4), L21 (3) |
| arithmetic | last_token | 500 (313) | 0.963 [0.948, 0.976] | 0.900 [0.872, 0.926] | 0.469 | 0.866 | 0.917 | L14 (9), L28 (2), L18 (1) |
| factual | last_token | 500 (295) | 0.814 [0.774, 0.852] | 0.752 [0.714, 0.790] | 0.496 | 0.555 | 0.509 | L25 (7), L28 (5), L24 (2) |
| all | mean_pool | 1000 (608) | 0.883 [0.862, 0.901] | 0.800 [0.773, 0.825] | 0.486 | 0.746 | 0.779 | L19 (9), L13 (2), L14 (1) |

Per-layer out-of-fold AUROC, last-token probe (layer 0 = embeddings; the peak is optimistic because it is picked on the same curve):

| Scope | Layer 0 | Layer 1 | Peak (layer) | Final layer |
|---|--:|--:|--:|--:|
| all | 0.500 | 0.812 | 0.904 (L24) | 0.900 |
| arithmetic | 0.500 | 0.946 | 0.963 (L14) | 0.962 |
| factual | 0.500 | 0.590 | 0.818 (L24) | 0.812 |

Label-threshold sensitivity (pooled scope, last-token probe): tool_needed = accuracy < 80% -> nested AUROC 0.909 [0.891, 0.924]; tool_needed = accuracy < 50% -> nested AUROC 0.906 [0.888, 0.923]

#### Probe-predicted need vs actual tool calls (out-of-fold probe, threshold 0.5)

| Prompt | n | Tool-call rate | Needed + called | Needed + NOT called | Not needed + called (over-use) | Not needed + not called |
|---|--:|--:|--:|--:|--:|--:|
| baseline | 1000 | 95.5% | 585 | 23 | 370 | 22 |
| intervention | 1000 | 60.7% | 442 | 166 | 165 | 227 |

| Prompt | Probe cell: knows & does | knows, doesn't | not-needed & does | neither | Probe says NOT needed: over-use items | ...not-needed & not-called items | ...ALL not-needed items | Probe says needed: needed & not-called items |
|---|--:|--:|--:|--:|---|---|---|---|
| baseline | 552 | 18 | 403 | 27 | 81% [77%, 85%] (n=370) | 86% [73%, 100%] (n=22) | 81% [77%, 85%] (n=392) | 65% [43%, 83%] (n=23) |
| intervention | 439 | 122 | 168 | 271 | 75% [68%, 81%] (n=165) | 87% [83%, 92%] (n=227) | 82% [78%, 86%] (n=392) | 56% [48%, 63%] (n=166) |

Probe score vs behaviour: baseline: AUROC(probe, tool_needed)=0.900, AUROC(probe, tool_called)=0.623 (within not-needed only 0.508, within needed only 0.709), ECE=0.043; intervention: AUROC(probe, tool_needed)=0.901, AUROC(probe, tool_called)=0.786 (within not-needed only 0.628, within needed only 0.827), ECE=0.051

#### Same questions, two prompts (secondary)

Tool-call rate 95.5% (baseline) -> 60.7% (intervention); 348 questions flipped from calling to not calling. Probe P(needed) across the two prompts: Spearman rho = 0.980, mean |change| = 0.046, mean change on the flipped questions = -0.008.

Run info: model Qwen/Qwen2.5-7B-Instruct, precision bf16, GPU NVIDIA L40S, torch 2.14.1+cu126, transformers 5.24.0.

<p>
  <img src="results/probe/figures/probe_layer_curve.png" width="100%" alt="Probe AUROC by layer">
</p>
<p>
  <img src="results/probe/figures/four_cell.png" width="100%" alt="Probe-predicted need vs actual tool call, baseline prompt">
</p>
<p>
  <img src="results/probe/figures/four_cell_intervention.png" width="100%" alt="Probe-predicted need vs actual tool call, intervention prompt">
</p>
<p>
  <img src="results/probe/figures/calibration.png" width="40%" alt="Calibration of the probe, baseline prompt">
</p>

#### Reading the results

1. **Tool necessity is linearly decodable from the last prompt token.** The pooled probe reaches nested AUROC 0.900 [0.882, 0.916], against 0.489 with shuffled labels and 0.779 for the best surface-feature model. The label-threshold variants give about the same (0.909 and 0.906), so the result does not hinge on the "any of 10 samples wrong" rule.
2. **How much of that is surface form depends on the domain.** For arithmetic the probe (0.963) is only modestly above the best surface baseline (0.917), and its AUROC is already 0.946 at layer 1 (peak 0.963 at layer 14), so most of the signal is readable almost immediately from the expression itself. For factual QA the probe (0.814) is far above the surface baselines (0.555 and 0.509) and builds with depth (0.590 at layer 1, 0.818 at its peak at layer 24), which is the stronger evidence of a representation that goes beyond the question's surface form. What that representation encodes (for example topic familiarity versus answerability) is not determined here.
3. **Under the baseline prompt, behavior does not follow the probe's belief.** The model calls a tool on 95.5% of questions, including 370 of the 392 questions it did not need a tool for. The probe says "not needed" on 81% of those over-use items, but also on 81% of *all* not-needed items (and on 86% of the 22 not-needed questions where the model did not call a tool), so it does not treat over-used questions differently from the rest. Among not-needed questions the probe score does not separate called from not-called (AUROC 0.508). So the over-use looks like a prompt-and-policy effect, not a case of the representation failing to flag these questions as unneeded.
4. **A prompt change moves behavior but not the representation.** The intervention prompt lowers the call rate to 60.7%, and 348 questions flip from calling to not calling, while the probe's P(needed) for the same questions is nearly unchanged (Spearman 0.980; mean change on the flipped questions -0.008). Behavior also follows the probe more under this prompt (AUROC of the probe score for predicting tool calls 0.786 versus 0.623 under the baseline). The cost is under-use: needed-but-not-called questions grow from 23 to 166, and the probe says "needed" on 56% of those 166, the knowing-but-not-doing pattern.
5. **Caveats that bear on how to read this:** the probes for the two prompts are trained separately; a probe shows what is decodable, not that the model uses it (no causal test was run); and the layer chosen by nested CV varies across folds (for the pooled probe it was spread over layers such as 24, 25 and 21), so there is no single "best layer".

### Reproducibility check

The committed results were reproduced twice from a fresh clone of `main` (commit `f7c9a8a`) after the original run:

- **CPU stages only**, from the original activations and labels: every probe, knowing-doing and calibration CSV and `summary.md` came out byte-identical.
- **The full pipeline, GPU stages included**, on a different NVIDIA L40S node: necessity labels, all sampled answers, tool-call decisions and both activation files were byte-identical to the original run, and so were the CSVs and `summary.md`.

The only differences were recorded file paths, stage run times, and the last floating-point digit of three values in `summary.json`.

What this does not show: both runs used the same GPU model, software environment, seed and batch sizes. Another GPU model, library version or batch size has not been tested and may not match bit for bit. Reproducing a result also says nothing about whether its labels are right (see the limitations below). The script that compared the two runs is not in this repo. To check for yourself, run `OUT=results_repro ./run_probe.sh` and diff the folder against `results/probe/`. The original activation files have these SHA-256 hashes:

```text
715bf2d6ed634716d46870a3369b5477318b727e99c96236e804335d09117e3e  activations.npz
e2ba66138ff86606ed5e638f26b0194b7b4a09d3eb4fc15a57771a6850c1eb4d  activations_intervention.npz
```

### Limitations

- **Single model, single seed.** Everything is about Qwen2.5-7B-Instruct with one seed set; nothing here generalizes to other models, and the paper's own results differ by model and by domain.
- **A probe is not a causal claim.** A linear probe shows that a necessity signal is *linearly decodable* from the representation. It does not show that the model uses that signal, and "the probe says not needed" does not prove the model "knew" in any deeper sense.
- **The probe measures capability, not intent.** It is trained on the model-adaptive necessity label (could the model answer unaided), not on whether the model should have called a tool.
- **Partly synthetic question set.** The arithmetic is generated from the paper's families; the factual gold answers come from TriviaQA and may contain errors or incomplete alias lists. Spot-check `gold_audit.csv`.
- **Necessity is a sampled estimate.** `tool_needed` = any of 10 samples wrong at temperature 0.7, so a single unlucky sample can flip a borderline label. The sensitivity thresholds show how much this matters; they do not remove it.
- **Few probe training samples relative to dimensionality** (3,584-dimensional hidden states), so strong regularization is used and per-category results are noisier than pooled ones.
- **Behavior is one decision.** The action stage records only the first-turn decision under greedy decoding; the mock search returns the gold answer, so post-tool accuracy is not analysed. The prompt includes a question ID; the original 30 questions carry `A`/`F` prefixes that reveal the category, while the 970 new ones use neutral IDs.
- **Numerical noise.** Hidden states in bf16 can differ slightly with batch composition. The batch size and library versions are recorded in `run_info.json`. If the 8-bit fallback is used, the weights are quantized and the activations are not bf16 activations.
- **A control was added after I first saw the results.** My first version of the over-use analysis reported only that the probe says "not needed" on 81% of over-use items. That number is uninformative alone, because it equals the probe's accuracy on all not-needed items. I added the comparison against all not-needed items and the within-class AUROCs in a separate commit (`knowing_doing.py`, `summarize.py`), with the original values unchanged. It was a post-hoc addition, not a pre-planned test.
- **The "probe says not needed" cells use a 0.5 threshold on a class-balanced probe**, and the calibration error (ECE) is measured against the necessity label, not against behavior.
- **No steering test** has been run.

## License

[MIT](LICENSE)
