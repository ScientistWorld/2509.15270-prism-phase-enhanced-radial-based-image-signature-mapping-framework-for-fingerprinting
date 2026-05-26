#!/bin/bash
# Download all data and models needed for this environment.
#
# Run on a node with internet access BEFORE building the container
# or submitting compute jobs.
#
# This script should be idempotent — safe to run multiple times.
# It should skip files that already exist.
#
# Update the storage estimate below as you add downloads.
#
# Estimated total download size: 8.36 GB
# Estimated disk usage after download: 8.36 GB
#
# This script downloads to workspace-local paths so a fresh clone can
# reproduce the benchmark without relying on machine-specific caches.

set -euo pipefail

cd /home/user

mkdir -p data/downloads/tiny-genimage/data

BASE_URL="https://huggingface.co/datasets/TheKernel01/Tiny-GenImage/resolve/main"

download_one() {
  local rel="$1"
  local out="data/downloads/tiny-genimage/$rel"
  local tmp="${out}.part"
  mkdir -p "$(dirname "$out")"
  if [ -e "$out" ]; then
    return 0
  fi
  curl -L -C - "$BASE_URL/$rel" -o "$tmp"
  mv "$tmp" "$out"
}

[ -e data/downloads/tiny-genimage/README.md ] || \
  download_one README.md

download_many() {
  local running=0
  for rel in "$@"; do
    download_one "$rel" &
    running=$((running + 1))
    if [ "$running" -ge 4 ]; then
      wait -n
      running=$((running - 1))
    fi
  done
  wait
}

files=()

for i in $(seq -w 0 13); do
  files+=("data/train-000${i}-of-00014.parquet")
done

for i in $(seq -w 0 3); do
  files+=("data/validation-0000${i}-of-00004.parquet")
done

download_many "${files[@]}"
