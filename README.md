# The Knowing-Doing Gap: Does an LLM Know It Needs a Tool?

A behavioral probing experiment measuring the gap between what an LLM *believes* about its own capability and what it *does* when tools are available. This is a black-box, behavioral adaptation of the paper's tool-necessity test, run on **Qwen/Qwen2.5-7B-Instruct-Turbo** via TogetherAI.

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

## How this differs from the paper

- **Behavioral only.** The paper also probes hidden states to show the model internally "knows" it needs a tool. This project measures behavior through the API (repeated no-tool runs, self-reported confidence, actual tool calls) and does no hidden-state probing.
- **One model, small question set.** 30 questions on a single model, versus the paper's larger multi-model study.
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

> **Status: draft. The run that produces the results has not finished, so this section contains no result numbers yet.** Every number that will appear here is generated by `./run_probe.sh` into `results/probe/summary.md`, and will be copied from there, never typed by hand.

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

*Pending. This subsection will be generated from `results/probe/summary.md` after the run finishes: question-set and necessity statistics, probe performance against the controls (overall and per category), the four-cell analysis and over-use result, calibration, label-threshold sensitivity, and the secondary intervention-prompt comparison. The top-of-README summary will be updated only once these results exist.*

### Limitations

- **Single model, single run.** Everything is about Qwen2.5-7B-Instruct with one seed set; nothing here generalizes to other models, and the paper's own results differ by model and by domain.
- **A probe is not a causal claim.** A linear probe shows that a necessity signal is *linearly decodable* from the representation. It does not show that the model uses that signal, and "the probe says not needed" does not prove the model "knew" in any deeper sense.
- **The probe measures capability, not intent.** It is trained on the model-adaptive necessity label (could the model answer unaided), not on whether the model should have called a tool.
- **Partly synthetic question set.** The arithmetic is generated from the paper's families; the factual gold answers come from TriviaQA and may contain errors or incomplete alias lists. Spot-check `gold_audit.csv`.
- **Necessity is a sampled estimate.** `tool_needed` = any of 10 samples wrong at temperature 0.7, so a single unlucky sample can flip a borderline label. The sensitivity thresholds show how much this matters; they do not remove it.
- **Few probe training samples relative to dimensionality** (3,584-dimensional hidden states), so strong regularization is used and per-category results are noisier than pooled ones.
- **Behavior is one decision.** The action stage records only the first-turn decision under greedy decoding; the mock search returns the gold answer, so post-tool accuracy is not analysed. The prompt includes a question ID; the original 30 questions carry `A`/`F` prefixes that reveal the category, while the 970 new ones use neutral IDs.
- **Numerical noise.** Hidden states in bf16 can differ slightly with batch composition. The batch size and library versions are recorded in `run_info.json`. If the 8-bit fallback is used, the weights are quantized and the activations are not bf16 activations.
- **No steering test** has been run.

## License

[MIT](LICENSE)
