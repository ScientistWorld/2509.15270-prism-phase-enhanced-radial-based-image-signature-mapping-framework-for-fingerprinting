#!/bin/bash
# Run the paper's proposed method (PRISM) and its magnitude-only / phase-only
# ablations; writes outputs/prism_predictions.csv.
#
# Usage:
#   bash scripts/method.sh [extra args for method/run_prism.py]

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p outputs
python3 method/run_prism.py "$@"
