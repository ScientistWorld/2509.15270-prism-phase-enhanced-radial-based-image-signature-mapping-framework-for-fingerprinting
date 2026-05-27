#!/usr/bin/env python3
"""Held-out test-slice scorer for Tiny GenImage prediction artifacts.

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


SLICE = "test"
TRAIN_SPLIT_SEED = "tiny-genimage-visible-train-split-v2"
TEST_SPLIT_SEED = "tiny-genimage-heldout-test-split-v2"
TRAIN_FRACTION = 0.40
TEST_FRACTION = 0.40


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"experiment", "method", "sample_id", "y_true", "y_pred"}
    if rows and not required.issubset(rows[0]):
        missing = sorted(required - set(rows[0]))
        raise ValueError(f"{path} is missing required prediction columns: {missing}")
    return rows


def train_membership(rows: list[dict[str, str]]) -> set[tuple[str, str]]:
    """Reproduce visible-train membership so the held-out slice is disjoint."""

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


def test_membership(rows: list[dict[str, str]]) -> set[tuple[str, str]]:
    """Return a hidden-salted, label-stratified test slice.

    Only a subset of the non-train examples is used. The remaining buffer
    prevents visible train membership from revealing held-out test membership.
    """

    by_label: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        by_label[(row["experiment"], row["y_true"])].add(row["sample_id"])

    train = train_membership(rows)
    selected: set[tuple[str, str]] = set()
    for (experiment, _label), sample_ids in by_label.items():
        candidates = [
            sample_id
            for sample_id in sample_ids
            if (experiment, sample_id) not in train
        ]
        ordered = sorted(
            candidates,
            key=lambda sample_id: hashlib.sha256(
                f"{TEST_SPLIT_SEED}|{experiment}|{sample_id}".encode("utf-8")
            ).hexdigest(),
        )
        count = max(1, int(len(sample_ids) * TEST_FRACTION))
        selected.update((experiment, sample_id) for sample_id in ordered[:count])
    return selected


def expected_methods(train_scores: Path) -> dict[str, set[str]]:
    if not train_scores.exists():
        return {}
    payload = json.loads(train_scores.read_text())
    experiments = payload.get("experiments", {})
    return {
        experiment: set(details.get("results", {}))
        for experiment, details in experiments.items()
    }


def score_rows(
    rows: list[dict[str, str]],
    required_methods: dict[str, set[str]],
) -> dict[str, object]:
    groups: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: {"y_true": [], "y_pred": [], "feature_count": []}
    )
    for row in rows:
        experiment = row["experiment"]
        method = row["method"]
        if required_methods and method not in required_methods.get(experiment, set()):
            continue
        key = (experiment, method)
        groups[key]["y_true"].append(int(row["y_true"]))
        groups[key]["y_pred"].append(int(row["y_pred"]))
        if row.get("feature_count"):
            groups[key]["feature_count"].append(float(row["feature_count"]))

    missing = []
    for experiment, methods in sorted(required_methods.items()):
        for method in sorted(methods):
            if (experiment, method) not in groups:
                missing.append(f"{experiment}/{method}")
    if missing:
        raise RuntimeError(
            "Test predictions are missing methods present in scoring/scores_train.json: "
            + ", ".join(missing)
        )

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
    parser.add_argument("--output", type=Path, default=Path("scoring/scores_test.json"))
    parser.add_argument("--train-scores", type=Path, default=Path("scoring/scores_train.json"))
    args = parser.parse_args()

    rows = read_rows(args.predictions)
    membership = test_membership(rows)
    selected = [row for row in rows if (row["experiment"], row["sample_id"]) in membership]
    if not selected:
        raise RuntimeError(f"No prediction rows selected for {SLICE} slice from {args.predictions}")

    scores = score_rows(selected, expected_methods(args.train_scores))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scores, indent=2) + "\n")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
