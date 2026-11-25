#!/bin/bash
# Correction pass (CPU only): regrade the saved answers with the v2 grader and data/label_decisions.json,
# then rerun the probes and the two added analyses. results/probe/ (the original run) is only read.
#
# Needs the two activation files from the original run in $ACT (results/probe/ by default).
# Environment overrides: ACT, OUT, N_JOBS, SEED, PYTHON
set -euo pipefail
cd "$(dirname "$0")"
ACT=${ACT:-results/probe}
OUT=${OUT:-results/probe_corrected}
N_JOBS=${N_JOBS:--1}
SEED=${SEED:-0}
PY=${PYTHON:-python3}

$PY src/probe/regrade.py --outdir "$OUT"
$PY src/probe/run_probe.py --outdir "$OUT" --activations "$ACT/activations.npz" --seed "$SEED" --n-jobs "$N_JOBS" --mean-pool-scopes all,arithmetic,factual
for PAIR in 0.8:_acc80 0.5:_acc50; do
  $PY src/probe/run_probe.py --outdir "$OUT" --activations "$ACT/activations.npz" --seed "$SEED" --n-jobs "$N_JOBS" --label-threshold "${PAIR%%:*}" --tag "${PAIR##*:}" --scopes all --no-mean-pool
done
$PY src/probe/run_probe.py --outdir "$OUT" --activations "$ACT/activations_intervention.npz" --seed "$SEED" --n-jobs "$N_JOBS" --tag _intervention --scopes all --no-mean-pool
$PY src/probe/knowing_doing.py --outdir "$OUT" --prompt baseline
$PY src/probe/knowing_doing.py --outdir "$OUT" --prompt intervention --tag _intervention
$PY src/probe/make_figures.py --outdir "$OUT"
$PY src/probe/make_figures.py --outdir "$OUT" --tag _intervention
$PY src/probe/summarize.py --outdir "$OUT"
$PY src/probe/compare_corrected.py --old results/probe --new "$OUT"
$PY src/probe/transfer.py --outdir "$OUT" --actdir "$ACT" --seed "$SEED" --n-jobs "$N_JOBS"
$PY src/probe/directions.py --outdir "$OUT" --actdir "$ACT" --seed "$SEED" --n-jobs "$N_JOBS"
