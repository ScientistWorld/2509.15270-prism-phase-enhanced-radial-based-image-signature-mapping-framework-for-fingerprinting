#!/bin/bash
# Example job script submitted via the end-of-turn YAML block.
#
# This runs inside the compute container with:
#   - Your workspace mounted at /home/user
#   - GPU(s) available
#   - No internet access
#
# Make sure all data, code, and dependencies are already in the
# workspace or baked into the container before submitting.

set -euo pipefail

cd /home/user

bash scripts/method.sh
bash scripts/baseline.sh

bash scripts/evaluate.sh outputs/prism_predictions.csv outputs/baseline_predictions.csv
bash scripts/evaluate_train.sh outputs/prism_predictions.csv outputs/baseline_predictions.csv
bash scripts/evaluate_test.sh outputs/prism_predictions.csv outputs/baseline_predictions.csv
