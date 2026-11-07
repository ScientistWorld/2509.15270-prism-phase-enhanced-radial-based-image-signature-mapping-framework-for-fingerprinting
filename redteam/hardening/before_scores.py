#!/usr/bin/env python3
"""Score the redteam's shortcut CSVs with the PRE-hardening test evaluator.

The pre-hardening evaluator could not import `eval.metrics` when launched as a
script (REDTEAM.md). To learn what it *would* have paid, this script runs an
unmodified snapshot of it (``old_eval/``) with only the import path repaired
(PYTHONPATH). Stdlib only; light enough for a login node.

Rank delta = sum_exp weight * sum_metric coefficient * (score - reference),
using the pre-hardening reference.json and the archived reference test scores,
for the proposed method key ``prism``.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OLD = HERE / "old_eval"
OUT = HERE / "before"
CYCLE = [1, 2, 3, 4, 6, 7, 8]


def rank(scores: dict) -> float:
    ref = json.loads((OLD / "ref_scores_test.json").read_text())["experiments"]
    cfg = json.loads((OLD / "reference.json").read_text())["experiments"]
    total = 0.0
    for exp, spec in cfg.items():
        cur = scores.get("experiments", {}).get(exp, {}).get("results", {}).get("prism")
        base = ref[exp]["results"]["prism"]
        if cur is None:
            continue
        total += spec["weight"] * sum(
            m["coefficient"] * (cur[k] - base[k]) for k, m in spec["metrics"].items() if k in cur
        )
    return total


def score(name: str, csv_path: Path) -> dict:
    folder = OUT / name
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / "scores_test.json"
    env = dict(os.environ, PYTHONPATH=str(OLD))
    proc = subprocess.run(
        [sys.executable, str(OLD / "eval/test/evaluate.py"), str(csv_path),
         "--output", str(out), "--train-scores", str(folder / "absent.json")],
        env=env, capture_output=True, text=True,
    )
    status = {"name": name, "returncode": proc.returncode}
    if proc.returncode == 0:
        scores = json.loads(out.read_text())
        prism = {
            exp: scores["experiments"].get(exp, {}).get("results", {}).get("prism")
            for exp in ("genimage_tiny_attribution", "genimage_tiny_binary")
        }
        status["prism"] = prism
        status["rank_delta"] = rank(scores)
    else:
        status["error"] = proc.stderr.strip().splitlines()[-1:] if proc.stderr else []
    (folder / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    return status


def synth(name: str, rows: list[dict]) -> Path:
    path = OUT / name / "predictions.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return path


def main() -> None:
    original = list(csv.DictReader((ROOT / "redteam/reference_cached/predictions.csv").open()))
    results = []
    for name in ["reference_cached", "baseline_majority", "constant_output_control",
                 "shortcut_visible_labels", "shortcut_forged_truth", "shortcut_shrunk_test",
                 "shortcut_feature_zero", "shortcut_impossible_values",
                 "shortcut_score_overwrite", "shortcut_baseline_replacement",
                 "shortcut_empty_output", "shortcut_null_feature_count",
                 "shortcut_nan_feature_count"]:
        results.append(score(name, ROOT / "redteam" / name / "predictions.csv"))

    # Newly identified: the public validation parquet is ordered in a fixed
    # period-14 label cycle, so the row index inside `validation/<idx>`
    # determines the label without looking at any pixel.
    leak = []
    for r in original:
        r = dict(r)
        if r["method"] == "prism":
            idx = int(r["sample_id"].split("/")[1])
            gen = 0 if idx % 2 == 0 else CYCLE[(idx // 2) % 7]
            r["y_pred"] = gen if r["experiment"] == "genimage_tiny_attribution" else int(gen > 0)
        leak.append(r)
    results.append(score("shortcut_index_position", synth("shortcut_index_position", leak)))

    # Newly identified: duplicated rows are all counted, so repeating the
    # correctly classified rows inflates every metric.
    dup = list(original)
    for r in original:
        if r["method"] == "prism" and r["y_true"] == r["y_pred"]:
            dup.extend([r] * 9)
    results.append(score("shortcut_duplicate_correct_rows", synth("shortcut_duplicate_correct_rows", dup)))

    (OUT / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    for s in results:
        p = s.get("prism") or {}
        a = (p.get("genimage_tiny_attribution") or {}).get("accuracy")
        b = (p.get("genimage_tiny_binary") or {}).get("accuracy")
        print(f"{s['name']:<34} rc={s['returncode']} attr_acc={a} bin_acc={b} rank={s.get('rank_delta')}")


if __name__ == "__main__":
    main()
