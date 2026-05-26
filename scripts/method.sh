#!/bin/bash
# Run the paper's proposed method.
#
# This script trains/runs the paper's method using code in method/.
#
# Usage:
#   bash scripts/method.sh

set -euo pipefail

cd /home/user

mkdir -p outputs

python method/run_prism.py "$@"

bash scripts/evaluate.sh outputs/prism_predictions.csv
