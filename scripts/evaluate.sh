#!/bin/bash
# Standard evaluation entry point.
#
# Scores every method in the prediction CSVs under outputs/ (or the CSV paths
# given as arguments) on the visible dev slice (scoring/scores_train.json).
# When the hidden test scorer is present it also scores the held-out test
# split (scoring/scores_test.json). scoring/scores.json mirrors the test
# scores when available (the paper's published-split setting), otherwise the
# dev scores. Uses only eval/ code; never imports method/.

set -euo pipefail

cd "$(dirname "$0")/.."
bash scripts/evaluate_train.sh "$@"
if [[ -f eval/test/evaluate.py && -f eval/test/labels.csv ]]; then
  bash scripts/evaluate_test.sh "$@"
  cp scoring/scores_test.json scoring/scores.json
else
  cp scoring/scores_train.json scoring/scores.json
fi
