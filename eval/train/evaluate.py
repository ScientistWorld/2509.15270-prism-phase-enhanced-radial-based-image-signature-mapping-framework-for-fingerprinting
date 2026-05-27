#!/usr/bin/env python3
"""Train-slice scorer for Tiny GenImage prediction artifacts.

This evaluator is method-agnostic: it consumes the same CSV prediction
artifact as scripts/evaluate.sh and never imports from method/.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from eval.metrics import classification_metrics


SLICE = "train"
TRAIN_SPLIT_SEED = "tiny-genimage-visible-train-split-v2"
TRAIN_FRACTION = 0.40


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"experiment", "method", "sample_id", "y_true", "y_pred"}
    if rows and not required.issubset(rows[0]):
        missing = sorted(required - set(rows[0]))
        raise ValueError(f"{path} is missing required prediction columns: {missing}")
    return rows


def train_membership(rows: list[dict[str, str]]) -> set[tuple[str, str]]:
    """Return deterministic, label-stratified visible-train membership.

    The train slice deliberately exposes only its own membership rule. The
    held-out test scorer uses an additional hidden salt and an unused buffer,
    so the complement of this set is not the test set.
    """

    by_label: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        by_label[(row["experiment"], row["y_true"])].add(row["sample_id"])

    selected: set[tuple[str, str]] = set()
    for (experiment, _label), sample_ids in by_label.items():
        ordered = sorted(
            sample_ids,
            key=lambda sample_id: hashlib.sha256(
                f"{TRAIN_SPLIT_SEED}|{experiment}|{sample_id}".encode("utf-8")
            ).hexdigest(),
        )
        count = max(1, int(len(ordered) * TRAIN_FRACTION))
        selected.update((experiment, sample_id) for sample_id in ordered[:count])
    return selected


def score_rows(rows: list[dict[str, str]]) -> dict[str, object]:
    groups: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: {"y_true": [], "y_pred": [], "feature_count": []}
    )
    for row in rows:
        key = (row["experiment"], row["method"])
        groups[key]["y_true"].append(int(row["y_true"]))
        groups[key]["y_pred"].append(int(row["y_pred"]))
        if row.get("feature_count"):
            groups[key]["feature_count"].append(float(row["feature_count"]))

    experiments: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for (experiment, method), values in sorted(groups.items()):
        feature_counts = values["feature_count"]
        feature_count = feature_counts[0] if feature_counts else 384.0
        experiments.setdefault(experiment, {"results": {}})
        experiments[experiment]["results"][method] = classification_metrics(
            [int(value) for value in values["y_true"]],
            [int(value) for value in values["y_pred"]],
            feature_count=feature_count,
        )
    return {"slice": SLICE, "experiments": experiments}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path, nargs="?", default=Path("outputs/prism_predictions.csv"))
    parser.add_argument("--output", type=Path, default=Path("scoring/scores_train.json"))
    args = parser.parse_args()

    rows = read_rows(args.predictions)
    membership = train_membership(rows)
    selected = [row for row in rows if (row["experiment"], row["sample_id"]) in membership]
    if not selected:
        raise RuntimeError(f"No prediction rows selected for {SLICE} slice from {args.predictions}")

    scores = score_rows(selected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scores, indent=2) + "\n")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
