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

if [ "${PRISM_SMOKE_TEST:-0}" = "1" ]; then
  bash scripts/method.sh --max-train-per-class 12 --max-test-per-class 8
else
  bash scripts/method.sh
fi
