#!/bin/bash
# Hidden-state probe extension: end-to-end runner.
#
#   ./run_probe.sh            # everything (GPU stages, then CPU stages)
#   ./run_probe.sh gpu        # stages 2-4 only: local necessity labels, behaviour, activations (needs a GPU)
#   ./run_probe.sh cpu        # stages 5-7 only: probes, knowing-doing analysis, figures, summary (CPU)
#
# Environment overrides:
#   OUT=results/probe  QUESTIONS=data/questions_probe.json  MODEL=Qwen/Qwen2.5-7B-Instruct
#   PRECISION=bf16|8bit  LIMIT=<n questions, for smoke tests>  N_JOBS=-1  SEED=0
set -euo pipefail
cd "$(dirname "$0")"

STAGE=${1:-all}
OUT=${OUT:-results/probe}
QUESTIONS=${QUESTIONS:-data/questions_probe.json}
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
PRECISION=${PRECISION:-bf16}
N_JOBS=${N_JOBS:--1}
SEED=${SEED:-0}
LIMIT_ARG=${LIMIT:+--limit $LIMIT}
PY=${PYTHON:-python3}

gpu_stages() {
  $PY src/probe/label_necessity_local.py --questions "$QUESTIONS" --outdir "$OUT" --model "$MODEL" --precision "$PRECISION" --n 10 --temp 0.7 --seed "$SEED" $LIMIT_ARG
  for PROMPT in baseline intervention; do
    $PY src/probe/run_action_local.py --questions "$OUT/questions_probe_labeled.json" --outdir "$OUT" --prompt "$PROMPT" --model "$MODEL" --precision "$PRECISION" $LIMIT_ARG
    $PY src/probe/extract_activations.py --questions "$OUT/questions_probe_labeled.json" --outdir "$OUT" --prompt "$PROMPT" --model "$MODEL" --precision "$PRECISION" $LIMIT_ARG
  done
}

cpu_stages() {
  # Headline probes (baseline prompt): all scopes, last-token + mean-pooled comparison.
  $PY src/probe/run_probe.py --outdir "$OUT" --seed "$SEED" --n-jobs "$N_JOBS" $LIMIT_ARG
  # Label-threshold sensitivity (pooled scope only).
  for PAIR in 0.8:_acc80 0.5:_acc50; do
    $PY src/probe/run_probe.py --outdir "$OUT" --seed "$SEED" --n-jobs "$N_JOBS" --label-threshold "${PAIR%%:*}" --tag "${PAIR##*:}" --scopes all --no-mean-pool $LIMIT_ARG
  done
  # Secondary condition: probe trained on activations under the intervention prompt.
  $PY src/probe/run_probe.py --outdir "$OUT" --seed "$SEED" --n-jobs "$N_JOBS" --activations "$OUT/activations_intervention.npz" --tag _intervention --scopes all --no-mean-pool $LIMIT_ARG
  $PY src/probe/knowing_doing.py --outdir "$OUT" --prompt baseline
  $PY src/probe/knowing_doing.py --outdir "$OUT" --prompt intervention --tag _intervention
  $PY src/probe/make_figures.py --outdir "$OUT"
  $PY src/probe/make_figures.py --outdir "$OUT" --tag _intervention
  $PY src/probe/summarize.py --outdir "$OUT"
}

case "$STAGE" in
  all) gpu_stages; cpu_stages ;;
  gpu) gpu_stages ;;
  cpu) cpu_stages ;;
  *) echo "Usage: ./run_probe.sh [all|gpu|cpu]"; exit 1 ;;
esac
