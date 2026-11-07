#!/bin/bash
# Reproduce the paper's Tiny GenImage results end-to-end.
#
# Assumes scripts/download.sh has populated data/downloads/ (training shards
# and the label-free test shards) and the container from environment/ is used.
# Runs PRISM (+ ablations) and the majority baseline, then scores everything
# under outputs/ with scripts/evaluate.sh.

set -euo pipefail

cd "$(dirname "$0")/.."
bash scripts/method.sh
bash scripts/baseline.sh
bash scripts/evaluate.sh
