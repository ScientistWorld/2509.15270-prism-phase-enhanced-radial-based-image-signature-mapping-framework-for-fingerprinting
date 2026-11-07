#!/usr/bin/env python3
"""Post-hardening audit: variants, shortcuts, seed spread and effect size.

Runs after scripts/reproduce.sh inside the job container. Every candidate
writes its own CSV under redteam/hardening/after/<name>/ (never under
outputs/, so the reference scores stay reference-only) and is scored by the
real evaluators (eval/train, eval/test) through subprocesses. Rank deltas use
scoring/reference.json coefficients against the reference `prism` entry of
scoring/scores_test.json.

The audit reads hidden labels only to (a) build the worst-case "shrunk test"
adversary, (b) bootstrap effect sizes, and (c) replay candidates through the
pre-hardening evaluator ("before" scores). No candidate's predictions are
derived from hidden labels except (a), which is the point of that trial.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "after"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from data.genimage import (  # noqa: E402
    PREDICTION_FIELDS, content_id, load_dev_split, test_shards, train_shards,
)
from method.run_prism import select_feature_mode  # noqa: E402
import before_scores  # noqa: E402

REF_CFG = json.loads((ROOT / "scoring/reference.json").read_text())["experiments"]
CYCLE = [1, 2, 3, 4, 6, 7, 8]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def write_rows(path: Path, rows: list[dict], fields=PREDICTION_FIELDS) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path


def run_eval(slice_: str, csv_path: Path, out: Path) -> tuple[int, dict | None, str]:
    proc = subprocess.run([sys.executable, f"eval/{slice_}/evaluate.py", str(csv_path), "--output", str(out)],
                          cwd=ROOT, capture_output=True, text=True)
    data = json.loads(out.read_text()) if proc.returncode == 0 and out.exists() else None
    return proc.returncode, data, proc.stderr.strip()[-400:]


def rank(scores: dict | None, method: str, ref: dict) -> float | None:
    if scores is None:
        return None
    total = 0.0
    for exp, spec in REF_CFG.items():
        cur = scores["experiments"][exp]["results"].get(method)
        base = ref["experiments"][exp]["results"]["prism"]
        if cur is None:
            return None
        total += spec["weight"] * sum(m["coefficient"] * (cur[k] - base[k]) for k, m in spec["metrics"].items())
    return total


def headline(scores: dict | None, method: str) -> dict:
    if scores is None:
        return {}
    a = scores["experiments"]["genimage_tiny_attribution"]["results"].get(method, {})
    b = scores["experiments"]["genimage_tiny_binary"]["results"].get(method, {})
    return {"attr_accuracy": a.get("accuracy"), "attr_balanced_accuracy": a.get("balanced_accuracy"),
            "attr_worst_class_recall": a.get("worst_class_recall"), "binary_accuracy": b.get("accuracy"),
            "failed": method in scores.get("_failed_methods", {}),
            "failure": scores.get("_failed_methods", {}).get(method)}


def score_candidate(name: str, csv_path: Path, method: str, ref: dict, results: list, extra=None) -> dict:
    folder = OUT / name
    folder.mkdir(parents=True, exist_ok=True)
    rc_tr, tr, _ = run_eval("train", csv_path, folder / "scores_train.json")
    rc_te, te, err = run_eval("test", csv_path, folder / "scores_test.json")
    status = {"name": name, "method": method, "train_rc": rc_tr, "test_rc": rc_te,
              "train": headline(tr, method), "test": headline(te, method),
              "rank_delta": rank(te, method, ref)}
    if rc_te:
        status["test_error"] = err
    if extra:
        status.update(extra)
    (folder / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    results.append(status)
    log(json.dumps(status))
    return status


# ---------------------------------------------------------------- before (old evaluator)
def before_replay(name: str, test_pred: dict[str, int], labels: dict[str, int], feature_count: int) -> dict:
    index_map = ROOT / "eval/test/downloads/index_map.csv"
    rows = []
    with index_map.open() as f:
        for r in csv.DictReader(f):
            sid = r["sample_id"]
            y, p = labels[sid], test_pred[sid]
            for exp, t, q in [("genimage_tiny_attribution", y, p), ("genimage_tiny_binary", int(y > 0), int(p > 0))]:
                rows.append(dict(experiment=exp, method="prism", split="validation",
                                 sample_id=f"validation/{r['public_index']}", y_true=t, y_pred=q,
                                 y_true_name="", y_pred_name="", feature_count=feature_count))
    fields = ["experiment", "method", "split", "sample_id", "y_true", "y_pred", "y_true_name",
              "y_pred_name", "feature_count"]
    path = write_rows(before_scores.OUT / name / "predictions.csv", rows, fields)
    return before_scores.score(name, path)


# ---------------------------------------------------------------- data helpers
def raw_iter(shards, columns):
    for shard in shards:
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=64, columns=columns):
            yield from batch.to_pylist()


def image_signature(raw: bytes) -> tuple:
    try:
        with Image.open(BytesIO(raw)) as im:
            return (im.format, im.mode, im.size[0], im.size[1])
    except Exception:  # noqa: BLE001 - probe only
        return ("unreadable", "", 0, 0)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    ref = json.loads((ROOT / "scoring/scores_test.json").read_text())
    labels = {r["sample_id"]: int(r["generator"]) for r in csv.DictReader((ROOT / "eval/test/labels.csv").open())}
    dev = load_dev_split()
    feats = dict(np.load(ROOT / "outputs/cache/prism_features.npz"))
    train_ids = feats["train_ids"].tolist()
    test_ids = feats["test_ids"].tolist()
    x, y, tx = feats["train_x"], feats["train_y"], feats["test_x"]
    in_dev = np.array([s in dev for s in train_ids])
    dev_ids = [s for s, d in zip(train_ids, in_dev) if d]
    reference_rows = list(csv.DictReader((ROOT / "outputs/prism_predictions.csv").open()))
    prism_rows = [r for r in reference_rows if r["method"] == "prism"]
    ref_test_pred = {r["sample_id"]: int(r["prediction"]) for r in prism_rows if r["split"] == "test"}

    # Reference through the same harness, plus its pre-hardening replay.
    score_candidate("reference", ROOT / "outputs/prism_predictions.csv", "prism", ref, results,
                    {"before": before_replay("reference", ref_test_pred, labels, 384)})

    # ------------------------------------------------------------ seeds
    seed_preds = []
    for seed in (0, 1, 2):
        np.random.seed(seed)
        sc = StandardScaler()
        lda = LinearDiscriminantAnalysis(solver="svd").fit(sc.fit_transform(x), y)
        seed_preds.append(lda.predict(sc.transform(tx)))
    ref_vec = np.array([ref_test_pred[s] for s in test_ids])
    seed_info = {"identical_to_reference": [bool((p == ref_vec).all()) for p in seed_preds],
                 "accuracy": [float(100 * np.mean(p == np.array([labels[s] for s in test_ids]))) for p in seed_preds]}
    log(f"seeds: {seed_info}")

    # ------------------------------------------------------------ variants
    def variant(name, make, mode="both", subsample=None):
        t0 = time.time()
        xtr, xte = select_feature_mode(x, mode, 64), select_feature_mode(tx, mode, 64)

        def fit(idx):
            if subsample:
                idx, _ = train_test_split(idx, train_size=subsample, stratify=y[idx], random_state=0)
            sc = StandardScaler().fit(xtr[idx])
            return sc, make().fit(sc.transform(xtr[idx]), y[idx])

        sc, model = fit(np.where(~in_dev)[0])
        dev_pred = model.predict(sc.transform(xtr[in_dev]))
        sc, model = fit(np.arange(len(y)))
        test_pred = model.predict(sc.transform(xte))
        rows = [dict(method=name, split="dev", sample_id=s, prediction=int(p)) for s, p in zip(dev_ids, dev_pred)]
        rows += [dict(method=name, split="test", sample_id=s, prediction=int(p)) for s, p in zip(test_ids, test_pred)]
        path = write_rows(OUT / name / "predictions.csv", rows)
        tp = {s: int(p) for s, p in zip(test_ids, test_pred)}
        score_candidate(name, path, name, ref, results,
                        {"seconds": round(time.time() - t0, 1), "feature_count": xtr.shape[1],
                         "before": before_replay(name, tp, labels, xtr.shape[1])})
        return test_pred

    variant_preds = {}
    variant_preds["better_shrinkage_lda"] = variant(
        "better_shrinkage_lda", lambda: LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
    variant_preds["better_histgb"] = variant(
        "better_histgb", lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
                                                                l2_regularization=1.0, random_state=0))
    variant_preds["better_rbf_svm"] = variant(
        "better_rbf_svm", lambda: SVC(C=10.0, gamma="scale", cache_size=2000))
    variant_preds["weaker_small_training"] = variant(
        "weaker_small_training", lambda: LinearDiscriminantAnalysis(solver="svd"), subsample=0.05)
    for mode, name in [("magnitude", "prism_magnitude_only"), ("phase", "prism_phase_only")]:
        rows = [r for r in reference_rows if r["method"] == name]
        path = write_rows(OUT / name / "predictions.csv", rows)
        tp = {r["sample_id"]: int(r["prediction"]) for r in rows if r["split"] == "test"}
        variant_preds[name] = np.array([tp[s] for s in test_ids])
        score_candidate(name, path, name, ref, results, {"before": before_replay(name, tp, labels, 192)})

    # ------------------------------------------------------------ bootstrap effect sizes (test)
    truth = np.array([labels[s] for s in test_ids])
    rng = np.random.default_rng(0)
    boots = rng.integers(0, len(truth), size=(2000, len(truth)))
    effects = {}
    for name, pred in variant_preds.items():
        diff = (pred == truth).astype(float) - (ref_vec == truth).astype(float)
        samples = diff[boots].mean(axis=1) * 100
        effects[f"{name} - prism"] = {"accuracy_diff": float(diff.mean() * 100),
                                      "ci95": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]}
    log(f"bootstrap: {effects}")

    # ------------------------------------------------------------ shortcuts
    test_cols = pq.ParquetFile(test_shards()[0]).schema_arrow.names
    dev_rows = [dict(method="sc", split="dev", sample_id=s, prediction=dev[s]) for s in dev_ids]

    def const_rows(method, value=0, **extra):
        rows = [dict(method=method, split="dev", sample_id=s, prediction=value, **extra) for s in dev_ids]
        rows += [dict(method=method, split="test", sample_id=s, prediction=value, **extra) for s in test_ids]
        return rows

    # 1 visible labels: only dev labels are visible; test shards carry no label column.
    rows = [dict(r, method="shortcut_visible_labels") for r in dev_rows]
    rows += [dict(method="shortcut_visible_labels", split="test", sample_id=s, prediction=0) for s in test_ids]
    score_candidate("shortcut_visible_labels", write_rows(OUT / "shortcut_visible_labels/predictions.csv", rows),
                    "shortcut_visible_labels", ref, results, {"test_shard_columns": test_cols})

    # 2 forged truth columns
    rows = const_rows("shortcut_forged_truth", 0, y_true=0, y_pred=0)
    score_candidate("shortcut_forged_truth",
                    write_rows(OUT / "shortcut_forged_truth/predictions.csv", rows, PREDICTION_FIELDS + ["y_true", "y_pred"]),
                    "shortcut_forged_truth", ref, results)

    # 3 shrunk test (worst case: oracle drops every wrong test row)
    rows = [dict(r, method="shortcut_shrunk_test") for r in prism_rows
            if r["split"] == "dev" or labels[r["sample_id"]] == int(r["prediction"])]
    score_candidate("shortcut_shrunk_test", write_rows(OUT / "shortcut_shrunk_test/predictions.csv", rows),
                    "shortcut_shrunk_test", ref, results, {"rows_kept": len(rows)})

    # 4 false resource count
    rows = [dict(r, method="shortcut_feature_zero", feature_count=0) for r in prism_rows]
    score_candidate("shortcut_feature_zero",
                    write_rows(OUT / "shortcut_feature_zero/predictions.csv", rows, PREDICTION_FIELDS + ["feature_count"]),
                    "shortcut_feature_zero", ref, results)

    # 5 impossible values
    for tag, value in [("99", 99), ("neg", -1), ("nan", "nan"), ("frac", "3.7")]:
        name = f"shortcut_impossible_values_{tag}"
        rows = const_rows(name, value, feature_count=-10000)
        score_candidate(name, write_rows(OUT / name / "predictions.csv", rows, PREDICTION_FIELDS + ["feature_count"]),
                        name, ref, results)

    # 6 score overwrite: forged perfect file at the output path, then the scorer runs
    name = "shortcut_score_overwrite"
    folder = OUT / name
    folder.mkdir(parents=True, exist_ok=True)
    forged = {"experiments": {e: {"results": {"prism": {m: 100.0 for m in REF_CFG[e]["metrics"]}}}} for e in REF_CFG}
    (folder / "scores_test.json").write_text(json.dumps(forged))
    rc, data, _ = run_eval("test", ROOT / "outputs/prism_predictions.csv", folder / "scores_test.json")
    overwritten = data is not None and data["experiments"]["genimage_tiny_attribution"]["results"]["prism"]["accuracy"] != 100.0
    results.append({"name": name, "test_rc": rc, "forged_file_replaced": overwritten,
                    "test": headline(data, "prism"), "rank_delta": rank(data, "prism", ref)})
    log(json.dumps(results[-1]))

    # 7 baseline replacement: evaluator never executes baseline code
    name = "shortcut_baseline_replacement"
    bpath = ROOT / "baseline/majority.py"
    original = bpath.read_bytes()
    bpath.write_text('raise RuntimeError("shortcut baseline replaced")\n')
    try:
        score_candidate(name, ROOT / "outputs/prism_predictions.csv", "prism", ref, results)
    finally:
        bpath.write_bytes(original)

    # 8 empty output
    name = "shortcut_empty_output"
    score_candidate(name, write_rows(OUT / name / "predictions.csv", []), name, ref, results)

    # 9 null / NaN resource counts with constant predictions
    for tag in ("null", "nan"):
        name = f"shortcut_{tag}_feature_count"
        rows = const_rows(name, 0, feature_count=tag)
        score_candidate(name, write_rows(OUT / name / "predictions.csv", rows, PREDICTION_FIELDS + ["feature_count"]),
                        name, ref, results)

    # 10 index position: the public validation order is a period-14 label cycle
    name = "shortcut_index_position"
    order = [r["sample_id"] for r in raw_iter(test_shards(), ["sample_id"])]
    pos_pred = {s: (0 if i % 2 == 0 else CYCLE[(i // 2) % 7]) for i, s in enumerate(order)}
    rows = [dict(method=name, split="dev", sample_id=s, prediction=0) for s in dev_ids]
    rows += [dict(method=name, split="test", sample_id=s, prediction=pos_pred[s]) for s in test_ids]
    score_candidate(name, write_rows(OUT / name / "predictions.csv", rows), name, ref, results)

    # 11 hedging: every label for every image
    name = "shortcut_hedge_all_labels"
    rows = [dict(method=name, split=sp, sample_id=s, prediction=k)
            for sp, ids in (("dev", dev_ids), ("test", test_ids)) for s in ids for k in range(9)]
    score_candidate(name, write_rows(OUT / name / "predictions.csv", rows), name, ref, results)

    # 12 duplicated rows (identical copies of the reference rows)
    name = "shortcut_duplicate_rows"
    rows = [dict(r, method=name) for r in prism_rows for _ in range(3)]
    score_candidate(name, write_rows(OUT / name / "predictions.csv", rows), name, ref, results)

    # 13 probe: file-format / resolution lookup table (no pixels)
    log("metadata probe: reading image headers")
    sig_train, seen = [], Counter()
    for row in raw_iter(train_shards(), ["image", "generator"]):
        raw = row["image"]["bytes"]
        base = content_id("tr", raw)
        k = seen[base]
        seen[base] += 1
        sid = base if k == 0 else f"{base}-dup{k}"
        sig_train.append((sid, image_signature(raw), int(row["generator"])))
    sig_test = {r["sample_id"]: image_signature(r["image"]) for r in raw_iter(test_shards(), ["sample_id", "image"])}

    def lookup(entries):
        full, coarse = defaultdict(Counter), defaultdict(Counter)
        for _, sig, lab in entries:
            full[sig][lab] += 1
            coarse[(sig[0], sig[1])][lab] += 1
        glob = Counter(lab for *_, lab in entries).most_common(1)[0][0]
        return lambda sig: (full[sig].most_common(1)[0][0] if sig in full else
                            coarse[(sig[0], sig[1])].most_common(1)[0][0] if (sig[0], sig[1]) in coarse else glob)

    fit_lookup = lookup([e for e in sig_train if e[0] not in dev])
    full_lookup = lookup(sig_train)
    dev_sig = {sid: sig for sid, sig, _ in sig_train if sid in dev}
    name = "probe_format_resolution_lookup"
    rows = [dict(method=name, split="dev", sample_id=s, prediction=fit_lookup(dev_sig[s])) for s in dev_ids]
    rows += [dict(method=name, split="test", sample_id=s, prediction=full_lookup(sig_test[s])) for s in test_ids]
    tp = {r["sample_id"]: r["prediction"] for r in rows if r["split"] == "test"}
    sig_counts = Counter((sig[0], sig[2], sig[3], lab) for _, sig, lab in sig_train)
    score_candidate(name, write_rows(OUT / name / "predictions.csv", rows), name, ref, results,
                    {"before": before_replay(name, tp, labels, 0),
                     "top_train_signatures": [list(map(str, k)) + [v] for k, v in sig_counts.most_common(25)]})

    summary = {"seeds": seed_info, "bootstrap": effects, "candidates": results,
               "reference_test": headline(ref, "prism")}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    log("done")


if __name__ == "__main__":
    main()
