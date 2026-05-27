#!/usr/bin/env python3
"""Run PRISM on Tiny GenImage and write reusable prediction artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.genimage import BINARY_NAMES, GENERATOR_NAMES, iter_records  # noqa: E402
from method.prism_features import extract_prism_features  # noqa: E402


def _select_record(label: int, counts: dict[int, int], limit: int | None) -> bool:
    if limit is None:
        counts[label] += 1
        return True
    if counts[label] >= limit:
        return False
    counts[label] += 1
    return True


def extract_split(
    data_root: Path,
    split: str,
    selection_task: str,
    max_per_class: int | None,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    counts: dict[int, int] = defaultdict(int)
    features: list[np.ndarray] = []
    labels: list[int] = []
    indices: list[int] = []

    desc = f"extract {split} {selection_task}"
    for record in tqdm(iter_records(data_root, split), desc=desc, unit="img"):
        selection_label = record.binary_label if selection_task == "binary" else record.generator_label
        if not _select_record(selection_label, counts, max_per_class):
            continue
        features.append(extract_prism_features(record.image, n_bins=n_bins))
        labels.append(record.generator_label)
        indices.append(record.index)
    if not features:
        raise RuntimeError(f"No records selected for split={split} task={selection_task}")
    return np.vstack(features), np.asarray(labels, dtype=np.int64), indices


def run_task(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    test_indices: list[int],
    task: str,
    experiment: str,
) -> list[dict[str, object]]:
    scaler = StandardScaler()
    scaled_train_x = scaler.fit_transform(train_x)
    scaled_test_x = scaler.transform(test_x)

    lda = LinearDiscriminantAnalysis(solver="svd")
    lda.fit(scaled_train_x, train_y)
    pred_y = lda.predict(scaled_test_x)

    names = BINARY_NAMES if task == "binary" else GENERATOR_NAMES
    rows: list[dict[str, object]] = []
    for idx, true_label, pred_label in zip(test_indices, test_y.tolist(), pred_y.tolist()):
        rows.append(
            {
                "experiment": experiment,
                "method": "prism",
                "split": "validation",
                "sample_id": f"validation/{idx}",
                "y_true": int(true_label),
                "y_pred": int(pred_label),
                "y_true_name": names.get(int(true_label), str(true_label)),
                "y_pred_name": names.get(int(pred_label), str(pred_label)),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/downloads/tiny-genimage")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/prism_predictions.csv")
    parser.add_argument("--n-bins", type=int, default=64)
    parser.add_argument("--task", choices=["attribution", "binary", "both"], default="both")
    parser.add_argument("--max-train-per-class", type=int)
    parser.add_argument("--max-test-per-class", type=int)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # PRISM uses the same Fourier feature representation for both attribution
    # and binary detection, so extract it once and reuse it for each LDA task.
    selection_task = "binary" if args.task == "binary" else "attribution"
    train_x, train_generator_y, _ = extract_split(
        args.data_root, "train", selection_task, args.max_train_per_class, args.n_bins
    )
    test_x, test_generator_y, test_indices = extract_split(
        args.data_root, "validation", selection_task, args.max_test_per_class, args.n_bins
    )

    rows: list[dict[str, object]] = []
    if args.task in {"attribution", "both"}:
        rows.extend(
            run_task(
                train_x,
                train_generator_y,
                test_x,
                test_generator_y,
                test_indices,
                "attribution",
                "genimage_tiny_attribution",
            )
        )
    if args.task in {"binary", "both"}:
        rows.extend(
            run_task(
                train_x,
                (train_generator_y > 0).astype(np.int64),
                test_x,
                (test_generator_y > 0).astype(np.int64),
                test_indices,
                "binary",
                "genimage_tiny_binary",
            )
        )

    fieldnames = [
        "experiment",
        "method",
        "split",
        "sample_id",
        "y_true",
        "y_pred",
        "y_true_name",
        "y_pred_name",
    ]
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    meta = {
        "method": "prism",
        "feature_count": 3 * 2 * args.n_bins,
        "n_bins": args.n_bins,
        "predictions": str(args.output),
    }
    args.output.with_suffix(".metadata.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
