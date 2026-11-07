#!/bin/bash
# Visible dev-slice scorer -> scoring/scores_train.json.
#
# With no arguments it scores every method found in any *.csv under outputs/
# that has the columns method,sample_id,prediction (see briefing/evaluation.md).
# Optional arguments: explicit prediction CSV paths instead of discovery.
# The dev slice is data/splits/dev.csv (20% of the published training split).

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p scoring
python3 eval/train/evaluate.py "$@" --output scoring/scores_train.json
