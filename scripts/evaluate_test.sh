#!/bin/bash
# Held-out test-slice evaluation entry point.
#
# Mirrors scripts/evaluate.sh: pass a prediction CSV as the first argument,
# or omit it to use outputs/prism_predictions.csv. The script writes
# scoring/scores_test.json and, when scoring/scores_train.json exists, scores
# exactly the method entries that appeared in the train-slice report.

set -euo pipefail

cd /home/user

mkdir -p scoring

PREDICTIONS="${1:-outputs/prism_predictions.csv}"

python eval/test/evaluate.py "$PREDICTIONS" \
  --output scoring/scores_test.json \
  --train-scores scoring/scores_train.json
