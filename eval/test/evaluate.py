#!/usr/bin/env python3
"""Hidden test scorer for Tiny GenImage source attribution.

Scores every method found in prediction CSVs against the fixed hidden answers
in ``eval/test/labels.csv`` (the 7,000 images of the published validation
split). Everything on the reference side is fixed here: ground truth, test
size, label space and metric definitions. Only ``method``, ``sample_id`` and
``prediction`` are read from a submission; any other column (labels, feature
counts, scores) is ignored.

A method is scored as failed (every metric 0) if, on the test ids, it
  * omits any id, or
  * gives a prediction that is not an integer generator label 0..8, or
  * gives conflicting predictions for the same id.
The binary real-vs-fake experiment is derived here from the source prediction
(0 = real, any generator = fake); it is not a separate submission channel.

Usage:
  python3 eval/test/evaluate.py [CSV ...] [--output scoring/scores_test.json]
With no CSV arguments every ``*.csv`` under ``outputs/`` is read.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LABELS_PATH = HERE / "labels.csv"
EXPECTED_SIZE = 7000
LABEL_SPACE = frozenset(range(9))
REAL = 0
REQUIRED_COLUMNS = ("method", "sample_id", "prediction")
INTEGER = re.compile(r"[0-9]+(?:\.0*)?")

ATTR = "genimage_tiny_attribution"
BINARY = "genimage_tiny_binary"
METRICS = {
    ATTR: ("accuracy", "balanced_accuracy", "precision_weighted", "f1_weighted",
           "f1_macro", "worst_class_recall"),
    BINARY: ("accuracy", "precision_weighted", "f1_weighted", "worst_class_recall"),
}


def load_answers() -> dict[str, int]:
    answers: dict[str, int] = {}
    with LABELS_PATH.open(newline="") as f:
        for row in csv.DictReader(f):
            sid = row["sample_id"]
            if sid in answers:
                raise SystemExit(f"hidden labels contain duplicate id {sid}")
            answers[sid] = int(row["generator"])
    if len(answers) != EXPECTED_SIZE or not set(answers.values()) <= LABEL_SPACE:
        raise SystemExit(f"hidden labels are corrupt: {len(answers)} rows")
    return answers


def parse_prediction(text: str | None) -> int | None:
    value = (text or "").strip()
    if not INTEGER.fullmatch(value):
        return None
    number = int(float(value)) if "." in value else int(value)
    return number if number in LABEL_SPACE else None


def collect(paths: list[Path], answers: dict[str, int]):
    methods: dict[str, dict] = {}
    used, skipped = [], []
    for path in paths:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            if not set(REQUIRED_COLUMNS) <= set(reader.fieldnames or []):
                skipped.append(str(path))
                continue
            used.append(str(path))
            for row in reader:
                name = (row.get("method") or "").strip()
                if not name:
                    continue
                entry = methods.setdefault(name, {"pred": {}, "invalid": 0, "conflicts": 0})
                sid = (row.get("sample_id") or "").strip()
                if sid not in answers:
                    continue
                value = parse_prediction(row.get("prediction"))
                if value is None:
                    entry["invalid"] += 1
                    continue
                previous = entry["pred"].get(sid)
                if previous is not None and previous != value:
                    entry["conflicts"] += 1
                entry["pred"][sid] = value
    return methods, used, skipped


def classification_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    total = len(y_true)
    classes = sorted(set(y_true))
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    precision, recall, f1, support = {}, {}, {}, {}
    for c in classes:
        tp = sum(t == c and p == c for t, p in zip(y_true, y_pred))
        fp = sum(t != c and p == c for t, p in zip(y_true, y_pred))
        n = sum(t == c for t in y_true)
        support[c] = n
        precision[c] = tp / (tp + fp) if tp + fp else 0.0
        recall[c] = tp / n
        s = precision[c] + recall[c]
        f1[c] = 2 * precision[c] * recall[c] / s if s else 0.0
    return {
        "accuracy": 100.0 * correct / total,
        "balanced_accuracy": 100.0 * sum(recall.values()) / len(classes),
        "precision_weighted": 100.0 * sum(precision[c] * support[c] for c in classes) / total,
        "f1_weighted": 100.0 * sum(f1[c] * support[c] for c in classes) / total,
        "f1_macro": 100.0 * sum(f1.values()) / len(classes),
        "worst_class_recall": 100.0 * min(recall.values()),
    }


def score(entry: dict, answers: dict[str, int]) -> tuple[dict[str, dict[str, float]], list[str]]:
    reasons = []
    missing = sum(1 for sid in answers if sid not in entry["pred"])
    if missing:
        reasons.append(f"missing predictions for {missing} of {len(answers)} test images")
    if entry["invalid"]:
        reasons.append(f"{entry['invalid']} predictions outside the integer label space 0..8")
    if entry["conflicts"]:
        reasons.append(f"{entry['conflicts']} conflicting duplicate predictions")
    if reasons:
        return {exp: {m: 0.0 for m in names} for exp, names in METRICS.items()}, reasons
    ids = sorted(answers)
    y_true = [answers[s] for s in ids]
    y_pred = [entry["pred"][s] for s in ids]
    attr = classification_metrics(y_true, y_pred)
    binary = classification_metrics([int(y != REAL) for y in y_true], [int(y != REAL) for y in y_pred])
    return {
        ATTR: {m: attr[m] for m in METRICS[ATTR]},
        BINARY: {m: binary[m] for m in METRICS[BINARY]},
    }, []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path, nargs="*")
    parser.add_argument("--output", type=Path, default=ROOT / "scoring/scores_test.json")
    args = parser.parse_args()

    paths = args.predictions or sorted(p for p in (ROOT / "outputs").rglob("*.csv") if p.is_file())
    answers = load_answers()
    methods, used, skipped = collect(paths, answers)
    if not methods:
        print(f"No prediction rows with columns {REQUIRED_COLUMNS} found in: "
              f"{[str(p) for p in paths]}", file=sys.stderr)
        sys.exit(1)

    experiments = {exp: {"results": {}} for exp in METRICS}
    failures = {}
    for name in sorted(methods):
        per_exp, reasons = score(methods[name], answers)
        for exp, values in per_exp.items():
            experiments[exp]["results"][name] = values
        if reasons:
            failures[name] = reasons
    payload = {
        "slice": "test",
        "n_samples": len(answers),
        "experiments": experiments,
        "_failed_methods": failures,
        "_prediction_files": used,
        "_skipped_files": skipped,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
