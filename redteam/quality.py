"""Real PRISM reruns and stronger/weaker classifiers on the published split.

Feature extraction is the unchanged method function. Its deterministic output
is cached once to fit the audit budget. Every reference seed refits its own
standardizer and SVD LDA on all 28,000 training images. Candidate parameters
are selected strictly on an internal split of the published training set.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path('/home/user')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'redteam'))

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from data.genimage import shard_paths, _decode_image
from method.prism_features import extract_prism_features
from method.run_prism import select_feature_mode
from common import AUDIT, install_and_score, main_rows, with_controls


def extract_shard(path):
    import pyarrow.parquet as pq
    xs, ys = [], []
    started = time.monotonic()
    for batch in pq.ParquetFile(path).iter_batches(batch_size=32, columns=['image', 'generator']):
        for r in batch.to_pylist():
            xs.append(extract_prism_features(_decode_image(r['image']), n_bins=64))
            ys.append(int(r['generator']))
    print(f'{Path(path).name}: {len(ys)} images in {time.monotonic()-started:.1f}s', flush=True)
    return np.vstack(xs), np.asarray(ys)


def feature_cache():
    target = AUDIT / 'artifacts/features.npz'
    if not target.exists():
        arrays = {}
        for split in ['train', 'validation']:
            paths = shard_paths(ROOT / 'data/downloads/tiny-genimage', split)
            with ProcessPoolExecutor(max_workers=4) as pool:
                chunks = list(pool.map(extract_shard, paths))
            arrays[f'{split}_x'] = np.vstack([v[0] for v in chunks])
            arrays[f'{split}_y'] = np.concatenate([v[1] for v in chunks])
        np.savez(target, **arrays)
    d = np.load(target)
    return d['train_x'], d['train_y'], d['validation_x'], d['validation_y']


def candidate(name):
    if name == 'better_rbf':
        return SVC(C=10.0, gamma='scale', cache_size=1024)
    if name == 'better_extratrees':
        return ExtraTreesClassifier(n_estimators=400, min_samples_leaf=1,
                                    max_features=0.8, n_jobs=4, random_state=1729)
    if name == 'better_histgb':
        return HistGradientBoostingClassifier(max_iter=180, max_leaf_nodes=31,
                                              l2_regularization=1.0,
                                              random_state=1729)
    if name == 'better_shrinkage':
        return LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    return LinearDiscriminantAnalysis(solver='svd')


def run(name):
    x, y, tx, ty = feature_cache()
    seed = int(name.removeprefix('reference_seed')) if name.startswith('reference_seed') else 1729
    random.seed(seed)
    np.random.seed(seed)
    details = {'name': name, 'seed': seed, 'train_samples': len(y),
               'validation_samples': len(ty), 'train_classes': dict(Counter(map(int, y))),
               'feature_cache': 'Unchanged PRISM features, 64 bins; published split',
               'reference_randomness': 'SVD LDA and extraction are deterministic; all rows retained in original order'}
    if name == 'weaker_small_training':
        selected, _ = train_test_split(np.arange(len(y)), train_size=0.05,
                                       stratify=y, random_state=seed)
        x, y = x[selected], y[selected]
        details['train_samples'] = len(y)
    if name in ['weaker_magnitude_only', 'weaker_phase_only']:
        mode = 'magnitude' if 'magnitude' in name else 'phase'
        x = select_feature_mode(x, mode, 64)
        tx = select_feature_mode(tx, mode, 64)
    # Fit the same preprocessing only on training data.
    scaler = StandardScaler()
    sx = scaler.fit_transform(x)
    stx = scaler.transform(tx)
    model = candidate(name)
    if name.startswith('better_'):
        tr, va = train_test_split(np.arange(len(y)), test_size=0.2,
                                 stratify=y, random_state=1729)
        holdout_scaler = StandardScaler()
        model.fit(holdout_scaler.fit_transform(x[tr]), y[tr])
        p = model.predict(holdout_scaler.transform(x[va]))
        details['internal_accuracy'] = accuracy_score(y[va], p) * 100
        details['internal_balanced_accuracy'] = balanced_accuracy_score(y[va], p) * 100
        # The test is never read by model fitting or selection.
        model = candidate(name)
    started = time.monotonic()
    model.fit(sx, y)
    pred = model.predict(stx)
    details['fit_predict_seconds'] = time.monotonic() - started
    details['feature_count'] = sx.shape[1]
    details['classifier'] = str(model)
    out = AUDIT / name
    out.mkdir(exist_ok=True)
    (out / 'method_details.json').write_text(json.dumps(details, indent=2) + '\n')
    return install_and_score(name, with_controls(main_rows(pred, ty, range(len(ty)), sx.shape[1])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        run(args.name)
