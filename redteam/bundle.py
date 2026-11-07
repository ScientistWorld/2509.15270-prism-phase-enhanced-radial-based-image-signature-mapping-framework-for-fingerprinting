"""Run the audit as a single compute job; save every result immediately."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path('/home/user')
AUDIT = ROOT / 'redteam'

ORDER = [
    'reference_cached',
    'shortcut_visible_labels', 'shortcut_forged_truth', 'shortcut_shrunk_test',
    'shortcut_feature_zero', 'shortcut_impossible_values',
    'shortcut_score_overwrite', 'shortcut_baseline_replacement',
    'constant_output_control', 'shortcut_empty_output',
    'shortcut_null_feature_count', 'shortcut_nan_feature_count',
    'baseline_majority',
    'reference_seed0', 'reference_seed1', 'reference_seed2',
    'weaker_magnitude_only', 'weaker_phase_only', 'weaker_small_training',
    'better_rbf', 'better_shrinkage', 'better_extratrees', 'better_histgb',
]


def restore():
    # Only files this audit writes, never unrelated initial worktree changes.
    for name in ['scores_train.json', 'scores_test.json']:
        subprocess.run(['git', 'restore', '--source=HEAD', '--', f'scoring/{name}'], cwd=ROOT, check=True)
    for p in (ROOT / 'outputs').glob('*predictions.csv'):
        p.unlink()
    shutil.copyfile(AUDIT / 'artifacts/original_predictions.csv', ROOT / 'outputs/prism_predictions.csv')


def main(interface_only=False):
    summary = []
    order = [name for name in ORDER if not name.startswith(('reference_seed', 'weaker_', 'better_'))] if interface_only else ORDER
    try:
        for name in order:
            folder = AUDIT / name
            if (folder / 'status.json').exists():
                summary.append(json.loads((folder / 'status.json').read_text()))
                continue
            started = time.monotonic()
            print(f'=== {name} ===', flush=True)
            try:
                with (folder / 'run.log').open('w') as log:
                    p = subprocess.run(['bash', str(folder / 'run.sh')], cwd=ROOT,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       timeout=5100 if name.startswith('reference_seed') else 2400)
                returncode = p.returncode
            except subprocess.TimeoutExpired:
                returncode = 124
            item = {'name': name, 'returncode': returncode, 'seconds': time.monotonic()-started}
            if (folder / 'status.json').exists():
                item.update(json.loads((folder / 'status.json').read_text()))
            summary.append(item)
            (AUDIT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
            print(json.dumps(item), flush=True)
            if name == 'reference_cached' and item.get('test_returncode') != 0 and not interface_only:
                # If the unmodified no-argument interface is broken, record
                # the failure instead of repairing its process/environment.
                break
    finally:
        restore()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--interface-only', action='store_true',
                        help='Exercise inexpensive candidates despite the broken reference interface; do not train unscorable models.')
    main(interface_only=parser.parse_args().interface_only)
