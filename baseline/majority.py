#!/usr/bin/env python3
"""Majority-class baseline for Tiny GenImage prediction artifacts.

Predicts the most frequent training label for every image: for dev images
the majority of the training images outside the dev slice, for test images
the majority of the full training split.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.genimage import (  # noqa: E402
    DEFAULT_ROOT,
    PREDICTION_FIELDS,
    iter_test,
    iter_train_labels,
    load_dev_split,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/baseline_predictions.csv")
    args = parser.parse_args()

    train_counts = Counter(iter_train_labels(args.data_root))
    dev = load_dev_split()
    fit_counts = train_counts - Counter(dev.values())
    test_label = train_counts.most_common(1)[0][0]
    dev_label = fit_counts.most_common(1)[0][0]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        for sid in sorted(dev):
            writer.writerow(dict(method="majority_baseline", split="dev", sample_id=sid, prediction=dev_label))
        for rec in iter_test(args.data_root, decode=False):
            writer.writerow(dict(method="majority_baseline", split="test", sample_id=rec.sample_id,
                                 prediction=test_label))


if __name__ == "__main__":
    main()
