#!/bin/bash
# Train-slice evaluation entry point.
#
# Mirrors scripts/evaluate.sh: pass a prediction CSV as the first argument,
# or omit it to use outputs/prism_predictions.csv. The script writes
# scoring/scores_train.json.

set -euo pipefail

cd /home/user

mkdir -p scoring

PREDICTIONS="${1:-outputs/prism_predictions.csv}"

python eval/train/evaluate.py "$PREDICTIONS" --output scoring/scores_train.json
