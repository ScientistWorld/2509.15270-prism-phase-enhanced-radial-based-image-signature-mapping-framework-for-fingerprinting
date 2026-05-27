#!/bin/bash
# Train-slice evaluation entry point.
#
# Mirrors scripts/evaluate.sh: pass a prediction CSV as the first argument,
# or omit it to score the prediction CSV found under outputs/. The script writes
# scoring/scores_train.json.

set -euo pipefail

cd /home/user

mkdir -p scoring

if [[ $# -gt 0 ]]; then
  python eval/train/evaluate.py "$@" --output scoring/scores_train.json
else
  mapfile -t CANDIDATES < <(find outputs -maxdepth 1 -type f -name '*predictions.csv' | sort)
  if [[ "${#CANDIDATES[@]}" -ne 1 ]]; then
    printf 'Expected exactly one prediction CSV under outputs/, found %s:\n' "${#CANDIDATES[@]}" >&2
    printf '  %s\n' "${CANDIDATES[@]}" >&2
    exit 1
  fi
  PREDICTIONS="${CANDIDATES[0]}"
  python eval/train/evaluate.py "$PREDICTIONS" --output scoring/scores_train.json
fi
