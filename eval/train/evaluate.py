#!/usr/bin/env python3
"""Visible development-slice scorer for Tiny GenImage source attribution.

Scores every method found in prediction CSVs on the 5,600 dev images listed in
``data/splits/dev.csv`` (a fixed, label-stratified 20% of the published
training split). Predictions for those images should come from a model that
did not train on them. The interface and metrics match the hidden test
scorer; only ``method``, ``sample_id`` and ``prediction`` are read.

A method is scored as failed (every metric 0) if, on the dev ids, it omits
any id, predicts something other than an integer label 0..8, or gives
conflicting predictions for one id. Binary real-vs-fake scores are derived
from the source prediction (0 = real, any generator = fake).

Usage:
  python3 eval/train/evaluate.py [CSV ...] [--output scoring/scores_train.json]
With no CSV arguments every ``*.csv`` under ``outputs/`` is read.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEV_LABELS = ROOT / "data/splits/dev.csv"
VALID_LABELS = frozenset(range(9))
COLUMNS = ("method", "sample_id", "prediction")
INT_TEXT = re.compile(r"[0-9]+(?:\.0*)?")

ATTRIBUTION = "genimage_tiny_attribution"
REAL_VS_FAKE = "genimage_tiny_binary"
REPORTED = {
    ATTRIBUTION: ("accuracy", "balanced_accuracy", "precision_weighted", "f1_weighted",
                  "f1_macro", "worst_class_recall"),
    REAL_VS_FAKE: ("accuracy", "precision_weighted", "f1_weighted", "worst_class_recall"),
}


def read_dev_labels() -> dict[str, int]:
    with DEV_LABELS.open(newline="") as f:
        labels = {r["sample_id"]: int(r["generator"]) for r in csv.DictReader(f)}
    if not labels or not set(labels.values()) <= VALID_LABELS:
        raise SystemExit(f"{DEV_LABELS} is empty or holds labels outside 0..8")
    return labels


def to_label(text: str | None) -> int | None:
    value = (text or "").strip()
    if not INT_TEXT.fullmatch(value):
        return None
    label = int(float(value))
    return label if label in VALID_LABELS else None


def gather(files: list[Path], labels: dict[str, int]):
    found: dict[str, dict] = {}
    read, ignored = [], []
    for file in files:
        with file.open(newline="") as handle:
            rows = csv.DictReader(handle)
            if not set(COLUMNS) <= set(rows.fieldnames or []):
                ignored.append(str(file))
                continue
            read.append(str(file))
            for row in rows:
                method = (row.get("method") or "").strip()
                if not method:
                    continue
                state = found.setdefault(method, {"pred": {}, "bad": 0, "clash": 0})
                sample = (row.get("sample_id") or "").strip()
                if sample not in labels:
                    continue
                label = to_label(row.get("prediction"))
                if label is None:
                    state["bad"] += 1
                elif state["pred"].setdefault(sample, label) != label:
                    state["clash"] += 1
    return found, read, ignored


def summarize(truth: list[int], guess: list[int]) -> dict[str, float]:
    n = len(truth)
    present = sorted(set(truth))
    stats = {}
    for c in present:
        hit = sum(1 for t, g in zip(truth, guess) if t == c and g == c)
        called = sum(1 for g in guess if g == c)
        size = sum(1 for t in truth if t == c)
        p = hit / called if called else 0.0
        r = hit / size
        stats[c] = (p, r, (2 * p * r / (p + r)) if p + r else 0.0, size)
    return {
        "accuracy": 100.0 * sum(1 for t, g in zip(truth, guess) if t == g) / n,
        "balanced_accuracy": 100.0 * sum(s[1] for s in stats.values()) / len(stats),
        "precision_weighted": 100.0 * sum(s[0] * s[3] for s in stats.values()) / n,
        "f1_weighted": 100.0 * sum(s[2] * s[3] for s in stats.values()) / n,
        "f1_macro": 100.0 * sum(s[2] for s in stats.values()) / len(stats),
        "worst_class_recall": 100.0 * min(s[1] for s in stats.values()),
    }


def evaluate(state: dict, labels: dict[str, int]):
    problems = []
    absent = len(labels) - sum(1 for s in labels if s in state["pred"])
    if absent:
        problems.append(f"missing predictions for {absent} of {len(labels)} dev images")
    if state["bad"]:
        problems.append(f"{state['bad']} predictions outside the integer label space 0..8")
    if state["clash"]:
        problems.append(f"{state['clash']} conflicting duplicate predictions")
    if problems:
        return {exp: dict.fromkeys(names, 0.0) for exp, names in REPORTED.items()}, problems
    order = sorted(labels)
    truth = [labels[s] for s in order]
    guess = [state["pred"][s] for s in order]
    source = summarize(truth, guess)
    binary = summarize([int(t > 0) for t in truth], [int(g > 0) for g in guess])
    return {
        ATTRIBUTION: {k: source[k] for k in REPORTED[ATTRIBUTION]},
        REAL_VS_FAKE: {k: binary[k] for k in REPORTED[REAL_VS_FAKE]},
    }, []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path, nargs="*")
    parser.add_argument("--output", type=Path, default=ROOT / "scoring/scores_train.json")
    args = parser.parse_args()

    files = args.predictions or sorted(p for p in (ROOT / "outputs").rglob("*.csv") if p.is_file())
    labels = read_dev_labels()
    found, read, ignored = gather(files, labels)
    if not found:
        print(f"No prediction rows with columns {COLUMNS} in {[str(f) for f in files]}", file=sys.stderr)
        sys.exit(1)

    experiments = {exp: {"results": {}} for exp in REPORTED}
    failed = {}
    for method in sorted(found):
        values, problems = evaluate(found[method], labels)
        for exp in REPORTED:
            experiments[exp]["results"][method] = values[exp]
        if problems:
            failed[method] = problems
    report = {
        "slice": "train",
        "n_samples": len(labels),
        "experiments": experiments,
        "_failed_methods": failed,
        "_prediction_files": read,
        "_skipped_files": ignored,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
