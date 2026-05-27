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

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


SLICE = "train"
SPLIT_SEED = "tiny-genimage-eval-split-v1"


def metric_values(y_true: list[int], y_pred: list[int], feature_count: float) -> dict[str, float]:
    return {
        "accuracy": 100.0 * accuracy_score(y_true, y_pred),
        "precision_weighted": 100.0
        * precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": 100.0
        * recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": 100.0 * f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "feature_count": float(feature_count),
    }


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"experiment", "method", "sample_id", "y_true", "y_pred"}
    if rows and not required.issubset(rows[0]):
        missing = sorted(required - set(rows[0]))
        raise ValueError(f"{path} is missing required prediction columns: {missing}")
    return rows


def split_membership(rows: list[dict[str, str]]) -> dict[tuple[str, str], str]:
    """Return deterministic, label-stratified train/test membership.

    The split key intentionally omits method so all methods' predictions for
    the same experiment/sample are evaluated on the same side.
    """

    by_label: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        by_label[(row["experiment"], row["y_true"])].add(row["sample_id"])

    membership: dict[tuple[str, str], str] = {}
    for (experiment, _label), sample_ids in by_label.items():
        ordered = sorted(
            sample_ids,
            key=lambda sample_id: hashlib.sha256(
                f"{SPLIT_SEED}|{experiment}|{sample_id}".encode("utf-8")
            ).hexdigest(),
        )
        for rank, sample_id in enumerate(ordered):
            membership[(experiment, sample_id)] = "train" if rank % 2 == 0 else "test"
    return membership


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
        experiments[experiment]["results"][method] = metric_values(
            [int(value) for value in values["y_true"]],
            [int(value) for value in values["y_pred"]],
            feature_count=feature_count,
        )
    return {"experiments": experiments}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path, nargs="?", default=Path("outputs/prism_predictions.csv"))
    parser.add_argument("--output", type=Path, default=Path("scoring/scores_train.json"))
    args = parser.parse_args()

    rows = read_rows(args.predictions)
    membership = split_membership(rows)
    selected = [
        row
        for row in rows
        if membership.get((row["experiment"], row["sample_id"])) == SLICE
    ]
    if not selected:
        raise RuntimeError(f"No prediction rows selected for {SLICE} slice from {args.predictions}")

    scores = score_rows(selected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scores, indent=2) + "\n")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
