"""Artifact installation and exact, no-argument gym evaluation.

No evaluator imports, source changes, environment changes, or persistent
processes. Each shell evaluator starts as its own process after the method.
"""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path('/home/user')
AUDIT = ROOT / 'redteam'
FIELDS = ['experiment', 'method', 'split', 'sample_id', 'y_true', 'y_pred',
          'y_true_name', 'y_pred_name', 'feature_count']
ATTR = 'genimage_tiny_attribution'
BINARY = 'genimage_tiny_binary'


def read_csv(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def original_rows():
    path = AUDIT / 'artifacts/original_predictions.csv'
    if not path.exists():
        # The audit may reuse the reference's legitimate trained outputs.
        # On a fresh checkout, recreate them from visible method/data files;
        # never reconstruct predictions from the archived hidden scores.
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['python', 'method/run_prism.py', '--output', str(path)],
                       cwd=ROOT, check=True)
    return read_csv(path)


def rank(scores, proposed_only=True):
    """Literal weighted coefficient times raw improvement, as specified.

    We report both the proposed-method rank and a sum including ablations.
    Normal variants preserve ablation controls, so their improvements are zero.
    No bounds, sign conversions, normalization, or missing-metric rewards.
    """
    reference = json.loads((AUDIT / 'original_scores_test.json').read_text())
    config = json.loads((AUDIT / 'reference.json').read_text())
    value = 0.0
    terms = {}
    for exp, entry in config['experiments'].items():
        current = scores.get('experiments', {}).get(exp, {}).get('results', {})
        old = reference['experiments'][exp]['results']
        for method, metrics in current.items():
            if (proposed_only and method != 'prism') or method not in old:
                continue
            subtotal = sum(spec['coefficient'] * (metrics[m] - old[method][m])
                           for m, spec in entry['metrics'].items()
                           if m in metrics and m in old[method])
            terms[f'{exp}/{method}'] = entry['weight'] * subtotal
            value += entry['weight'] * subtotal
    return value, terms


def install_and_score(name, rows, before_test=None):
    folder = AUDIT / name
    folder.mkdir(parents=True, exist_ok=True)
    write_csv(folder / 'predictions.csv', rows)
    # Quarantine all other candidate CSVs. Both official scripts insist on
    # exactly one file; neither receives a prediction path or any arguments.
    quarantine = AUDIT / 'artifacts/quarantine'
    quarantine.mkdir(parents=True, exist_ok=True)
    (ROOT / 'outputs').mkdir(exist_ok=True)
    for p in (ROOT / 'outputs').glob('*predictions.csv'):
        if p.is_file():
            shutil.move(str(p), str(quarantine / p.name))
    shutil.copyfile(folder / 'predictions.csv', ROOT / 'outputs/prism_predictions.csv')
    status = {'name': name, 'rows': len(rows)}
    for stage in ['train', 'test']:
        if stage == 'test' and before_test:
            before_test()
        with (folder / f'{stage}.log').open('w') as log:
            proc = subprocess.run(['bash', f'scripts/evaluate_{stage}.sh'],
                                  cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        status[f'{stage}_returncode'] = proc.returncode
        if proc.returncode:
            continue
        shutil.copyfile(ROOT / f'scoring/scores_{stage}.json', folder / f'scores_{stage}.json')
    if status.get('test_returncode') == 0:
        scores = json.loads((folder / 'scores_test.json').read_text())
        status['rank_delta'], status['rank_terms'] = rank(scores)
        status['rank_delta_all_methods'], _ = rank(scores, proposed_only=False)
    (folder / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
    print(json.dumps(status), flush=True)
    return status


def main_rows(predictions, labels, indices, feature_count):
    rows = []
    for exp in [ATTR, BINARY]:
        for pred, true, idx in zip(predictions, labels, indices):
            if exp == BINARY:
                true, pred = int(true > 0), int(pred > 0)
            rows.append(dict(experiment=exp, method='prism', split='validation',
                             sample_id=f'validation/{idx}', y_true=int(true),
                             y_pred=int(pred), y_true_name='', y_pred_name='',
                             feature_count=feature_count))
    return rows


def with_controls(rows):
    return rows + [r for r in original_rows() if r['method'] != 'prism']
