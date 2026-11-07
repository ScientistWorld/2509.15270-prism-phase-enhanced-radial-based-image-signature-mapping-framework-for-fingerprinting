#!/bin/bash
# Hardening verification job (CPU). Steps are idempotent, so a re-run resumes.
#   1. hidden side: label-free test inputs + hidden labels from the published validation split
#   2. visible dev slice (20% of the training split)
#   3. reference reproduction: PRISM + ablations + majority baseline, scored by both scorers
#   4. audit: better/weaker variants, shortcuts, seeds, bootstrap, pre-hardening replays

set -euo pipefail

cd /home/user

echo "== 1. export hidden test inputs"
if [[ ! -f eval/test/labels.csv || $(ls data/downloads/tiny-genimage/test/test-*.parquet 2>/dev/null | wc -l) -ne 4 ]]; then
  python3 eval/test/export_inputs.py
fi
echo "== 2. dev split"
python3 data/make_dev_split.py
echo "== 3. reference reproduction"
bash scripts/reproduce.sh
echo "== 4. hardening audit"
python3 redteam/hardening/after_scores.py
echo "== done"
