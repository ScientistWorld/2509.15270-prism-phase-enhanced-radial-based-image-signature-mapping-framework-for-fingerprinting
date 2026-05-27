#!/usr/bin/env python3
"""Score prediction CSV files without importing the reproduced method."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def _metrics(y_true: list[int], y_pred: list[int], feature_count: float = 384.0) -> dict[str, float]:
    return {
        "accuracy": 100.0 * accuracy_score(y_true, y_pred),
        "precision_weighted": 100.0
        * precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": 100.0
        * recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": 100.0 * f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "feature_count": float(feature_count),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path, nargs="?", default=Path("outputs/prism_predictions.csv"))
    parser.add_argument("--output", type=Path, default=Path("scoring/scores.json"))
    args = parser.parse_args()

    groups: dict[tuple[str, str], dict[str, list[int]]] = defaultdict(
        lambda: {"y_true": [], "y_pred": [], "feature_count": []}
    )
    with args.predictions.open(newline="") as f:
        for row in csv.DictReader(f):
            key = (row["experiment"], row["method"])
            groups[key]["y_true"].append(int(row["y_true"]))
            groups[key]["y_pred"].append(int(row["y_pred"]))
            if row.get("feature_count"):
                groups[key]["feature_count"].append(float(row["feature_count"]))

    experiments: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for (experiment, method), values in sorted(groups.items()):
        experiments.setdefault(experiment, {"results": {}})
        feature_counts = values["feature_count"]
        feature_count = feature_counts[0] if feature_counts else 384.0
        experiments[experiment]["results"][method] = _metrics(
            values["y_true"], values["y_pred"], feature_count=feature_count
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"experiments": experiments}, indent=2) + "\n")
    print(json.dumps({"experiments": experiments}, indent=2))


if __name__ == "__main__":
    main()
