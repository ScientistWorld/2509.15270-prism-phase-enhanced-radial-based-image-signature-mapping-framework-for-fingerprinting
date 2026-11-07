#!/usr/bin/env python3
"""Payoff of the left-open image-size cue, combined with the reference.

Uses saved prediction CSVs only (stdlib; light):
* the reference PRISM predictions (outputs/prism_predictions.csv), and
* the image-size lookup probe (after/probe_resolution_lookup/predictions.csv),
  which maps each image's width x height to the majority training label.

Rule "size_consistency": if the size is not a generator size -> real; if
PRISM says real but the size is a generator size -> the size's label; if the
size identifies one generator (128x128 BigGAN, 1024x1024 Midjourney) -> it;
if PRISM's generator is in a different size group -> the size's label;
otherwise keep PRISM. Scored with the real evaluators.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
GROUP = {1: "256", 3: "256", 7: "256", 6: "512", 8: "512", 2: "128", 4: "1024"}


def preds(path: Path, method: str) -> dict[str, tuple[str, int]]:
    with path.open() as f:
        return {r["sample_id"]: (r["split"], int(r["prediction"])) for r in csv.DictReader(f) if r["method"] == method}


def main() -> None:
    prism = preds(ROOT / "outputs/prism_predictions.csv", "prism")
    size = preds(HERE / "after/probe_resolution_lookup/predictions.csv", "probe_resolution_lookup")
    name = "shortcut_size_consistency"
    rows = []
    for sid, (split, p) in prism.items():
        r = size[sid][1]
        if r == 0:
            q = 0
        elif p == 0 or r in (2, 4) or GROUP.get(p) != GROUP[r]:
            q = r
        else:
            q = p
        rows.append(dict(method=name, split=split, sample_id=sid, prediction=q))
    folder = HERE / "after" / name
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "predictions.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "split", "sample_id", "prediction"])
        w.writeheader()
        w.writerows(rows)
    out = {}
    for s in ("train", "test"):
        subprocess.run([sys.executable, f"eval/{s}/evaluate.py", str(folder / "predictions.csv"),
                        "--output", str(folder / f"scores_{s}.json")], cwd=ROOT, check=True, capture_output=True)
        out[s] = json.loads((folder / f"scores_{s}.json").read_text())
    ref = json.loads((ROOT / "scoring/scores_test.json").read_text())
    cfg = json.loads((ROOT / "scoring/reference.json").read_text())["experiments"]
    rank = sum(spec["weight"] * sum(m["coefficient"] * (out["test"]["experiments"][e]["results"][name][k]
                                                        - ref["experiments"][e]["results"]["prism"][k])
                                    for k, m in spec["metrics"].items()) for e, spec in cfg.items())
    status = {"name": name, "rank_delta": rank,
              "train": {e: out["train"]["experiments"][e]["results"][name] for e in cfg},
              "test": {e: out["test"]["experiments"][e]["results"][name] for e in cfg}}
    (folder / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
