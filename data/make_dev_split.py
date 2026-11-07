#!/usr/bin/env python3
"""Create (or verify) data/splits/dev.csv: the visible development slice.

A fixed, label-stratified 20% of the published Tiny GenImage training split.
Membership is decided by a SHA-256 ranking of each image's content id inside
its generator class, so it is deterministic and independent of shard order.
The committed CSV is the source of truth; this script only regenerates it.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.genimage import DEFAULT_ROOT, DEV_SPLIT_CSV, iter_train  # noqa: E402

DEV_SALT = "tiny-genimage-dev-v1"
DEV_FRACTION = 0.20


def build(root: Path) -> list[tuple[str, int]]:
    by_class: dict[int, list[str]] = defaultdict(list)
    for record in iter_train(root, decode=False):
        by_class[record.generator_label].append(record.sample_id)
    rows: list[tuple[str, int]] = []
    for label, ids in sorted(by_class.items()):
        ranked = sorted(ids, key=lambda s: hashlib.sha256(f"{DEV_SALT}|{s}".encode()).hexdigest())
        rows.extend((sid, label) for sid in ranked[: round(len(ranked) * DEV_FRACTION)])
    return sorted(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEV_SPLIT_CSV)
    args = parser.parse_args()

    rows = build(args.data_root)
    if args.output.exists():
        with args.output.open(newline="") as f:
            existing = [(r["sample_id"], int(r["generator"])) for r in csv.DictReader(f)]
        if existing != rows:
            raise SystemExit(f"{args.output} does not match the downloaded training data")
        print(f"{args.output}: verified {len(rows)} dev rows")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "generator"])
        writer.writerows(rows)
    print(f"{args.output}: wrote {len(rows)} dev rows")


if __name__ == "__main__":
    main()
