#!/bin/bash

set -e

RUN_DIR=${1:-results}
ARG2=$2
ARG3=$3

if [ "$RUN_DIR" = "--limit" ] || [ "$RUN_DIR" = "--ids" ]; then
  echo "Usage:"
  echo "  ./run_experiment.sh"
  echo "  ./run_experiment.sh results --limit <limit>"
  echo "  ./run_experiment.sh results --ids A01,A04,A08,F01,F12"
  exit 1
fi

QUESTIONS_FILE="data/questions.json"

mkdir -p "$RUN_DIR"

# -------------------------------
# Parse optional arguments
# -------------------------------

LIMIT_ARG=""

if [ -n "$ARG2" ]; then
  if [ "$ARG2" = "--ids" ]; then
    if [ -z "$ARG3" ]; then
      echo "Error: --ids requires a comma-separated list, e.g. --ids A01,A04,F01"
      exit 1
    fi

    SELECTED_QUESTIONS_FILE="$RUN_DIR/questions_selected.json"

    python3 - <<PY
import json

ids = set("${ARG3}".split(","))

with open("data/questions.json", "r") as f:
    data = json.load(f)

questions = data["questions"] if isinstance(data, dict) and "questions" in data else data
selected = [q for q in questions if q.get("id") in ids]

found = {q.get("id") for q in selected}
missing = ids - found

if missing:
    raise SystemExit(f"Missing question IDs: {sorted(missing)}")

with open("${SELECTED_QUESTIONS_FILE}", "w") as f:
    json.dump({"questions": selected}, f, indent=2)

print(f"Selected {len(selected)} questions: {[q.get('id') for q in selected]}")
print("Wrote ${SELECTED_QUESTIONS_FILE}")
PY

    QUESTIONS_FILE="$SELECTED_QUESTIONS_FILE"

  elif [ "$ARG2" = "--limit" ]; then
    if [ -z "$ARG3" ]; then
      echo "Error: --limit requires a number, e.g. --limit 6"
      exit 1
    fi
    LIMIT_ARG="--limit $ARG3"

  else
    # Backward-compatible mode:
    # ./run_experiment.sh results 6
    LIMIT_ARG="--limit $ARG2"
  fi
fi

# -------------------------------
# Run experiment pipeline
# -------------------------------

python3 src/label_necessity.py \
  --questions "$QUESTIONS_FILE" \
  --outdir "$RUN_DIR" \
  --n 10 \
  --temp 0.7 \
  $LIMIT_ARG

python3 src/run_cognition.py \
  --questions "$RUN_DIR/questions_labeled.json" \
  --outdir "$RUN_DIR" \
  $LIMIT_ARG

python3 src/run_action.py \
  --questions "$RUN_DIR/questions_labeled.json" \
  --outdir "$RUN_DIR" \
  $LIMIT_ARG

python3 src/run_action.py \
  --questions "$RUN_DIR/questions_labeled.json" \
  --outdir "$RUN_DIR" \
  --intervention \
  $LIMIT_ARG

python3 src/analyze_results.py \
  --run-dir "$RUN_DIR"