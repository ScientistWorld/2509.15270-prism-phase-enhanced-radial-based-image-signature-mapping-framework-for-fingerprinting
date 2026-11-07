#!/usr/bin/env python3
"""Run PRISM on Tiny GenImage and write the prediction artifact.

Pipeline (paper Sec. 4): radial-DFT magnitude + phase features (64 bins x 2 x
RGB = 384 values per image), standardised on the training data, classified by
Linear Discriminant Analysis.

Protocol on the published split:
* test predictions: fit on all 28,000 training images, predict the 7,000
  held-out test images (paper Table 1/2 setting);
* dev predictions: fit on the training images outside data/splits/dev.csv,
  predict the 5,600 dev images (visible development slice).

The magnitude-only and phase-only ablations (paper Fig. 5) reuse the same
features. Extracted features are cached in outputs/cache/ so ablations and
re-runs do not repeat the FFT pass.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.genimage import (  # noqa: E402
    DEFAULT_ROOT,
    PREDICTION_FIELDS,
    iter_test,
    iter_train,
    load_dev_split,
    test_shards,
    train_shards,
)
from method.prism_features import extract_prism_features  # noqa: E402

VARIANTS = [("both", "prism"), ("magnitude", "prism_magnitude_only"), ("phase", "prism_phase_only")]


def shard_fingerprint(data_root: Path) -> str:
    """Identify the exact input shards (name + size) the cache was built from."""
    parts = [f"{p.name}:{p.stat().st_size}" for p in train_shards(data_root) + test_shards(data_root)]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def extract_features(data_root: Path, n_bins: int, cache: Path) -> dict[str, np.ndarray]:
    fingerprint = shard_fingerprint(data_root)
    if cache.exists():
        loaded = dict(np.load(cache, allow_pickle=False))
        if int(loaded["n_bins"]) == n_bins and str(loaded.get("fingerprint", "")) == fingerprint:
            print(f"using cached features {cache}")
            return loaded
        print(f"cache {cache} does not match the current inputs; re-extracting")
    train_ids, train_x, train_y = [], [], []
    for rec in tqdm(iter_train(data_root), desc="extract train", unit="img", mininterval=30):
        train_ids.append(rec.sample_id)
        train_x.append(extract_prism_features(rec.image, n_bins=n_bins))
        train_y.append(rec.generator_label)
    test_ids, test_x = [], []
    for rec in tqdm(iter_test(data_root), desc="extract test", unit="img", mininterval=30):
        test_ids.append(rec.sample_id)
        test_x.append(extract_prism_features(rec.image, n_bins=n_bins))
    arrays = {
        "n_bins": np.asarray(n_bins),
        "fingerprint": np.asarray(fingerprint),
        "train_ids": np.asarray(train_ids),
        "train_x": np.vstack(train_x),
        "train_y": np.asarray(train_y, dtype=np.int64),
        "test_ids": np.asarray(test_ids),
        "test_x": np.vstack(test_x),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_name(cache.stem + ".part.npz")
    np.savez(tmp, **arrays)
    tmp.replace(cache)
    return arrays


def select_feature_mode(features: np.ndarray, mode: str, n_bins: int) -> np.ndarray:
    """Full, magnitude-only or phase-only PRISM descriptors.

    Layout per image: [R mag, R phase, G mag, G phase, B mag, B phase].
    """
    if mode == "both":
        return features
    offset = {"magnitude": 0, "phase": n_bins}[mode]
    return np.concatenate(
        [features[:, c * 2 * n_bins + offset: c * 2 * n_bins + offset + n_bins] for c in range(3)],
        axis=1,
    )


def fit_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    lda = LinearDiscriminantAnalysis(solver="svd")
    lda.fit(scaler.fit_transform(train_x), train_y)
    return lda.predict(scaler.transform(test_x))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/prism_predictions.csv")
    parser.add_argument("--cache", type=Path, default=ROOT / "outputs/cache/prism_features.npz")
    parser.add_argument("--n-bins", type=int, default=64)
    args = parser.parse_args()

    feats = extract_features(args.data_root, args.n_bins, args.cache)
    train_ids = feats["train_ids"].tolist()
    train_x, train_y = feats["train_x"], feats["train_y"]
    test_ids, test_x = feats["test_ids"].tolist(), feats["test_x"]

    dev = load_dev_split()
    in_dev = np.array([sid in dev for sid in train_ids])
    if int(in_dev.sum()) != len(dev):
        raise SystemExit(f"dev split ids not found in training data ({int(in_dev.sum())}/{len(dev)})")
    dev_ids = [sid for sid, flag in zip(train_ids, in_dev) if flag]

    rows: list[dict[str, object]] = []
    for mode, method in VARIANTS:
        x_tr = select_feature_mode(train_x, mode, args.n_bins)
        dev_pred = fit_predict(x_tr[~in_dev], train_y[~in_dev], x_tr[in_dev])
        test_pred = fit_predict(x_tr, train_y, select_feature_mode(test_x, mode, args.n_bins))
        rows += [dict(method=method, split="dev", sample_id=s, prediction=int(p))
                 for s, p in zip(dev_ids, dev_pred)]
        rows += [dict(method=method, split="test", sample_id=s, prediction=int(p))
                 for s, p in zip(test_ids, test_pred)]
        print(f"{method}: {x_tr.shape[1]} features, dev/test predictions written")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    meta = {
        "methods": {m: 3 * (2 if mode == "both" else 1) * args.n_bins for mode, m in VARIANTS},
        "n_bins": args.n_bins,
        "train_images": len(train_ids),
        "dev_images": len(dev_ids),
        "test_images": len(test_ids),
    }
    args.output.with_suffix(".metadata.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
