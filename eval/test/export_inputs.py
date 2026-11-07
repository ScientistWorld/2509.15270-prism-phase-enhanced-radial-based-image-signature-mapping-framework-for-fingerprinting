#!/usr/bin/env python3
"""Hidden-side preparation of the held-out test inputs.

Reads the published Tiny GenImage *validation* shards (which carry labels)
from ``eval/test/downloads/`` and writes:

* ``data/downloads/tiny-genimage/test/test-0000k-of-00004.parquet`` - the
  method-visible test inputs: ``sample_id`` + raw encoded image bytes only,
  ordered by ``sample_id`` (the public shards follow a fixed label cycle, so
  their row order must not survive) and without the parquet ``path`` field.
* ``eval/test/labels.csv`` - the hidden answers (written once, verified after).
* ``eval/test/downloads/index_map.csv`` - public row index -> sample_id, used
  only by audits that replay pre-hardening artifacts.

Called by scripts/download.sh. Never imported by the scorer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
N_OUT_SHARDS = 4


def raw_bytes(value: object) -> bytes:
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return bytes(value["bytes"])
        return Path(value["path"]).read_bytes()
    return bytes(value)


def iter_raw(raw_dir: Path, columns: list[str]):
    shards = sorted(raw_dir.glob("validation-*.parquet"))
    if len(shards) != 4:
        raise SystemExit(f"expected 4 validation shards under {raw_dir}, found {len(shards)}")
    index = 0
    for shard in shards:
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=128, columns=columns):
            for row in batch.to_pylist():
                yield index, row
                index += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=HERE / "downloads")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data/downloads/tiny-genimage/test")
    parser.add_argument("--labels", type=Path, default=HERE / "labels.csv")
    args = parser.parse_args()

    # Pass 1: content ids and labels.
    ids: list[str] = []
    labels: dict[str, int] = {}
    seen: Counter[str] = Counter()
    path_examples: list[str] = []
    for index, row in iter_raw(args.raw_dir, ["image", "generator"]):
        image = row["image"]
        if isinstance(image, dict) and image.get("path") and len(path_examples) < 5:
            path_examples.append(str(image["path"]))
        base = f"te-{hashlib.sha256(raw_bytes(image)).hexdigest()[:20]}"
        k = seen[base]
        seen[base] += 1
        sid = base if k == 0 else f"{base}-dup{k}"
        ids.append(sid)
        labels[sid] = int(row["generator"])
    duplicates = sum(1 for s in ids if "-dup" in s)
    print(f"test images: {len(ids)}; byte-identical duplicates: {duplicates}")
    print(f"label counts: {sorted(Counter(labels.values()).items())}")
    print(f"raw parquet path field examples (dropped from exported inputs): {path_examples}")

    with (args.raw_dir / "index_map.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["public_index", "sample_id"])
        writer.writerows(enumerate(ids))

    rows = sorted(labels.items())
    if args.labels.exists():
        with args.labels.open(newline="") as f:
            existing = [(r["sample_id"], int(r["generator"])) for r in csv.DictReader(f)]
        if existing != rows:
            raise SystemExit(f"{args.labels} does not match the downloaded validation split")
        print(f"{args.labels}: verified {len(rows)} hidden labels")
    else:
        with args.labels.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["sample_id", "generator"])
            writer.writerows(rows)
        print(f"{args.labels}: wrote {len(rows)} hidden labels")

    # Pass 2..N: write label-free shards in sample_id order, one shard per pass
    # to bound memory.
    ordered = sorted(ids)
    chunk = -(-len(ordered) // N_OUT_SHARDS)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([("sample_id", pa.string()), ("image", pa.binary())])
    for k in range(N_OUT_SHARDS):
        target = args.out_dir / f"test-{k:05d}-of-{N_OUT_SHARDS:05d}.parquet"
        wanted = set(ordered[k * chunk:(k + 1) * chunk])
        payload: dict[str, bytes] = {}
        for index, row in iter_raw(args.raw_dir, ["image"]):
            if ids[index] in wanted:
                payload[ids[index]] = raw_bytes(row["image"])
        keys = sorted(payload)
        table = pa.table({"sample_id": keys, "image": [payload[s] for s in keys]}, schema=schema)
        tmp = target.with_suffix(".parquet.part")
        pq.write_table(table, tmp, row_group_size=256)
        tmp.replace(target)
        print(f"wrote {target} ({len(keys)} images)")


if __name__ == "__main__":
    main()
