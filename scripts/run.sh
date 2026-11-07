#!/bin/bash
# Reference reproduction job (CPU, ~70 min on one core).
#
# Runs inside the container with the workspace at /home/user and no internet.
# Assumes scripts/download.sh has populated data/downloads/ (training shards and
# the label-free test shards). Runs PRISM + its magnitude-only / phase-only
# ablations and the majority baseline, then scores every method under outputs/
# on the dev slice (scores_train.json) and the hidden test (scores_test.json).

set -euo pipefail

cd /home/user
# CPU jobs get one core; keep native thread pools from oversubscribing it.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

python3 data/make_dev_split.py      # verifies data/splits/dev.csv against the data
bash scripts/reproduce.sh
