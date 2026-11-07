#!/bin/bash
# Hidden test-slice scorer -> scoring/scores_test.json.
#
# With no arguments it scores every method found in any *.csv under outputs/
# that has the columns method,sample_id,prediction (see briefing/evaluation.md).
# Optional arguments: explicit prediction CSV paths instead of discovery.
# Ground truth, test size and metrics are fixed inside eval/test/.

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p scoring
python3 eval/test/evaluate.py "$@" --output scoring/scores_test.json
