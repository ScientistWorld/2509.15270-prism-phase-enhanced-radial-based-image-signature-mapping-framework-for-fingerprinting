#!/usr/bin/env python3
"""Score prediction CSV files without importing the reproduced method."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from eval.metrics import classification_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path, nargs="*")
    parser.add_argument("--output", type=Path, default=Path("scoring/scores.json"))
    args = parser.parse_args()
    prediction_paths = args.predictions or [Path("outputs/prism_predictions.csv")]

    groups: dict[tuple[str, str], dict[str, list[int]]] = defaultdict(
        lambda: {"y_true": [], "y_pred": [], "feature_count": []}
    )
    for path in prediction_paths:
        with path.open(newline="") as f:
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
        experiments[experiment]["results"][method] = classification_metrics(
            values["y_true"], values["y_pred"], feature_count=feature_count
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"experiments": experiments}, indent=2) + "\n")
    print(json.dumps({"experiments": experiments}, indent=2))


if __name__ == "__main__":
    main()
