#!/usr/bin/env python3
"""Majority-class baseline for Tiny GenImage prediction artifacts."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.genimage import BINARY_NAMES, GENERATOR_NAMES, iter_records  # noqa: E402


def majority_label(data_root: Path, task: str) -> int:
    counts: Counter[int] = Counter()
    for record in tqdm(iter_records(data_root, "train"), desc=f"baseline train {task}", unit="img"):
        counts[record.binary_label if task == "binary" else record.generator_label] += 1
    return counts.most_common(1)[0][0]


def write_task(writer: csv.DictWriter, data_root: Path, task: str, experiment: str) -> None:
    pred = majority_label(data_root, task)
    names = BINARY_NAMES if task == "binary" else GENERATOR_NAMES
    for record in tqdm(iter_records(data_root, "validation"), desc=f"baseline eval {task}", unit="img"):
        true = record.binary_label if task == "binary" else record.generator_label
        writer.writerow(
            {
                "experiment": experiment,
                "method": "majority_baseline",
                "split": "validation",
                "sample_id": f"validation/{record.index}",
                "y_true": true,
                "y_pred": pred,
                "y_true_name": names.get(true, str(true)),
                "y_pred_name": names.get(pred, str(pred)),
                "feature_count": 0,
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/downloads/tiny-genimage")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/baseline_predictions.csv")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "experiment",
        "method",
        "split",
        "sample_id",
        "y_true",
        "y_pred",
        "y_true_name",
        "y_pred_name",
        "feature_count",
    ]
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        write_task(writer, args.data_root, "attribution", "genimage_tiny_attribution")
        write_task(writer, args.data_root, "binary", "genimage_tiny_binary")


if __name__ == "__main__":
    main()
