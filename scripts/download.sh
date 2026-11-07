#!/bin/bash
# Download all data needed for this environment (Tiny GenImage, public HF mirror).
#
# Run on a node with internet access BEFORE submitting compute jobs.
# Idempotent: every fetch is skipped when its target already exists.
#
# Estimated total download size: 8.36 GB
# Estimated disk usage after download: ~10.1 GB (raw shards + 1.75 GB label-free test copy)
#
# Layout produced:
#   data/downloads/tiny-genimage/data/train-*.parquet   published training split (labelled)
#   eval/test/downloads/validation-*.parquet            published validation split (labelled,
#                                                        hidden side of the benchmark)
#   data/downloads/tiny-genimage/test/test-*.parquet    the same 7,000 validation images as
#                                                        label-free test inputs (sample_id + bytes)
# The last step needs a Python with pyarrow (installed by environment/setup.sh);
# when the host python3 lacks it, `uv run --with pyarrow` is used instead.

set -euo pipefail

cd "$(dirname "$0")/.."

REVISION="89c4fe9efd0ebc7ce5c7641ef57d578ccd639c69"
BASE_URL="https://huggingface.co/datasets/TheKernel01/Tiny-GenImage/resolve/${REVISION}"
TRAIN_DIR="data/downloads/tiny-genimage"
HIDDEN_DIR="eval/test/downloads"

fetch() {
  local rel="$1" out="$2"
  [ -e "$out" ] && return 0
  mkdir -p "$(dirname "$out")"
  curl -fL --retry 5 -C - "$BASE_URL/$rel" -o "${out}.part"
  mv "${out}.part" "$out"
}

fetch README.md "$TRAIN_DIR/README.md"

pids=()
for i in $(seq -w 0 13); do
  fetch "data/train-000${i}-of-00014.parquet" "$TRAIN_DIR/data/train-000${i}-of-00014.parquet" &
  pids+=($!)
  if [ "${#pids[@]}" -ge 4 ]; then wait "${pids[0]}"; pids=("${pids[@]:1}"); fi
done
for pid in "${pids[@]}"; do wait "$pid"; done

shopt -s nullglob
test_shards=("$TRAIN_DIR"/test/test-*-of-00004.parquet)
if [ "${#test_shards[@]}" -ne 4 ]; then
  for i in 0 1 2 3; do
    fetch "data/validation-0000${i}-of-00004.parquet" "$HIDDEN_DIR/validation-0000${i}-of-00004.parquet"
  done
  if python3 -c "import pyarrow" >/dev/null 2>&1; then
    python3 eval/test/export_inputs.py
  else
    uv run --no-project --with pyarrow python3 eval/test/export_inputs.py
  fi
fi
