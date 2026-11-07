#!/bin/bash
# Run the majority-class baseline; writes outputs/baseline_predictions.csv.
#
# Usage:
#   bash scripts/baseline.sh

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p outputs
python3 baseline/majority.py "$@"
