#!/bin/bash
# Held-out test-slice evaluation entry point.
#
# Mirrors scripts/evaluate.sh: pass a prediction CSV as the first argument,
# or omit it to score the prediction CSV found under outputs/. The script writes
# scoring/scores_test.json and, when scoring/scores_train.json exists, scores
# exactly the method entries that appeared in the train-slice report.

set -euo pipefail

cd /home/user

mkdir -p scoring

if [[ $# -gt 0 ]]; then
  PREDICTIONS="$1"
else
  mapfile -t CANDIDATES < <(find outputs -maxdepth 1 -type f -name '*predictions.csv' | sort)
  if [[ "${#CANDIDATES[@]}" -ne 1 ]]; then
    printf 'Expected exactly one prediction CSV under outputs/, found %s:\n' "${#CANDIDATES[@]}" >&2
    printf '  %s\n' "${CANDIDATES[@]}" >&2
    exit 1
  fi
  PREDICTIONS="${CANDIDATES[0]}"
fi

python eval/test/evaluate.py "$PREDICTIONS" \
  --output scoring/scores_test.json \
  --train-scores scoring/scores_train.json
