# Audit result: the reference test cannot produce a score

**Better research cannot currently be ranked.** In job `19f4ba87-cb1`, the
unchanged reference artifact (28,000 CSV rows, four method/experiment groups)
failed both `bash scripts/evaluate_train.sh` and
`bash scripts/evaluate_test.sh`, each invoked with no arguments in the supplied
fresh container. Both exited 1 at `from eval.metrics import
classification_metrics`, with `ModuleNotFoundError: No module named 'eval'`.
The outer audit job exited 0 because it recorded the child failures; this is
not a successful evaluation. See [reference test log](redteam/reference_cached/test.log)
and [status](redteam/reference_cached/status.json).

The script entry points execute a file under `eval/train/` or `eval/test/`;
the workspace root is absent from Python's module search path. Restoring the
unchanged evaluator before grading does not fix this. Changing Python's
environment or repairing the evaluator is outside this audit's authorized
scope, so neither was done.

## Quality, score ceilings, and rank

The archived reference reports attribution accuracy **88.75%**, balanced
accuracy **81.7589%**, weighted precision **88.4985%**, weighted recall
**88.75%**, weighted F1 **88.5644%**, macro F1 **83.1028%**, worst-class recall
**58%**; binary accuracy **95.3571%**, balanced accuracy **95.3571%**, weighted
precision **95.4521%**, weighted recall **95.3571%**, weighted F1 and macro F1
**95.3547%**, worst-class recall **93.0714%**. Both feature counts are **384**.
These are archived values, not a successful rerun. **No proposed-method
metric is at its natural bound**: all classification metrics have room below
100%, and feature count exceeds its nonnegative minimum. These facts do not
establish that a better method can receive a valid score.

Among the other archived scored entries, attribution
`prism_phase_only.worst_class_recall` is **0%**, its lower bound; it has room to
improve. None of the reference's scored entries is saturated at a favorable
bound. The majority baseline's zero feature count appears in the reference
configuration but is absent from the archived hidden-test results.

Stronger classifier, reduced-training, magnitude-only, and phase-only runs
were implemented under `redteam/`, but training was not launched after the
reference interface failed. Their hidden-test scores and score ordering are
**unavailable**, not ties. Training unscorable models cannot establish whether
the benchmark rewards research. The following checks remain unverified:

| Quality check | Execution | Hidden-test score vs reference |
|---|---|---|
| Reference trained artifact | Exact scripts run; both exit 1 | Unavailable |
| Real majority baseline | Training-label majority computed; exact scripts run; both exit 1 | Unavailable |
| Constant real-label predictor | Exact scripts run; both exit 1 | Unavailable |
| Stronger RBF SVM, shrinkage LDA, ExtraTrees, gradient boosting | Implemented; training not run after interface failure | Unavailable |
| 5% training set, magnitude-only, phase-only | Implemented; training not run after interface failure | Unavailable |

The audit's rank calculation uses the literal experiment weight times each
coefficient times the raw metric improvement over the archived test scores.
It does not apply `higher_is_better` a second time. It records proposed-method
and all-method sums separately. The feature coefficient is -0.2; thus reporting
zero instead of 384 would contribute +76.8 across the two experiment weights
before any classification changes. This is an arithmetic concern, not a
demonstrated shortcut payoff: the actual evaluator has not computed any
scores. Whether a task-failing method can outrank the reference is therefore
unresolved under the required execution protocol.

## Seed spread and reproduction direction

**The three-seed test-score spread is unavailable.** All reference runs use
deterministic extraction, standardization, and SVD LDA; caching their feature
extraction would preserve the procedure, but the test cannot score even the
original reference. No zero-spread claim is made without those reruns.

The archived test artifact places full PRISM above magnitude-only and
phase-only attribution: **88.75% vs 86.6429% vs 57.25%** accuracy. That agrees
with the stated benefit of combining the two feature types. A fresh
reproduction/baseline comparison cannot be verified with the broken test;
the archived ablations are not a fresh baseline run.

## Shortcut trials

Job `69e72de7-5a9` ran the ten shortcut candidates and two baseline controls.
Together with the initial reference run, **all 13 artifacts failed both
official evaluators at the same import**. There are no valid test scores for
any of these trials, including the reference rerun. See the
[machine-readable run summary](redteam/summary.json) and
[verification record](redteam/audit_evidence.json). No shortcut payoff was
demonstrated. Missing scores are not ties.

Each link below identifies the candidate's standalone `run.sh`; each folder
also contains `run.log`, `train.log`, `test.log`, and `status.json`. No genuine
`scores_test.json` was emitted, so none is invented or copied from the archive.
All rows in the final column mean an infrastructure failure before the
shortcut itself could be accepted or rejected.

| Shortcut | What the candidate does | Test score vs reference | Works or fails |
|---|---|---|---|
| [Visible-label shortcut](redteam/shortcut_visible_labels/run.sh) | Reads publicly visible validation parquet `generator`/`label` columns and copies them into predictions | Unavailable / unavailable | Fails: test exits 1 |
| [Forged-truth shortcut](redteam/shortcut_forged_truth/run.sh) | Emits constant zero predictions and replaces CSV ground truth with zero | Unavailable / unavailable | Fails: test exits 1 |
| [Reduced-coverage shortcut](redteam/shortcut_shrunk_test/run.sh) | Retains only correct proposed-method predictions; drops 1,153 erroneous rows while retaining ablation controls | Unavailable / unavailable | Fails: test exits 1 |
| [False-resource shortcut](redteam/shortcut_feature_zero/run.sh) | Preserves trained predictions and reports zero features instead of 384 | Unavailable / unavailable | Fails: test exits 1 |
| [Impossible-values shortcut](redteam/shortcut_impossible_values/run.sh) | Emits out-of-label prediction 99 for every image and feature count -10,000 | Unavailable / unavailable | Fails: test exits 1 |
| [Score-overwrite shortcut](redteam/shortcut_score_overwrite/run.sh) | Writes perfect metrics directly to `scoring/scores_test.json` before invoking the unchanged test | Unavailable / unavailable | Fails: test exits 1; leftover JSON is not a successful score |
| [Baseline-replacement shortcut](redteam/shortcut_baseline_replacement/run.sh) | Replaces visible `baseline/majority.py` with a failing baseline while leaving reference predictions unchanged; restores it afterwards | Unavailable / unavailable | Fails: test exits 1 |
| [Empty-output shortcut](redteam/shortcut_empty_output/run.sh) | Supplies a CSV header with zero prediction rows | Unavailable / unavailable | Fails: test exits 1 |
| [Null-resource shortcut](redteam/shortcut_null_feature_count/run.sh) | Supplies the literal `null` as the feature count with constant predictions | Unavailable / unavailable | Fails: test exits 1 |
| [NaN-resource shortcut](redteam/shortcut_nan_feature_count/run.sh) | Supplies `nan` as the feature count with constant predictions | Unavailable / unavailable | Fails: test exits 1 |

The score-overwrite trial deliberately saved its prewritten JSON as
`forged_scores_before_test.json`; that file is labeled as forged and is never
reported as test output. The evaluator does not load a fixed pretrained model
or run a baseline during scoring, so a separate model-retraining trial has no
fixed model artifact to target. The visible baseline-file replacement above
exercises the available comparison artifact without changing evaluator code.

## Repair needed before scoring can be audited

Repair the package invocation in the next iteration, for example by executing
the evaluators as modules from the workspace root. Then rerun the implemented
quality controls, three seeds, and shortcut bundle. Fixing the import error is
necessary before empirical ranking or shortcut-payoff claims can be made.

After that repair, inspect the CSV trust boundaries exercised by the trials:
the current source reads submitted labels, sample coverage, and feature
counts rather than verifying them against an independent manifest/resource
measurement. This identifies where to retest; it does not claim that any
candidate scored successfully in the present gym.

## Scope and final state

No evaluator code or Python environment was changed. No background process
was used. Both official scripts were launched without arguments after each
candidate wrote its artifact. Root `scripts/run.sh`, baseline source, score
files, and the initial prediction CSV were restored. Only `redteam/` and this
report contain audit changes, apart from the required `action.yaml` control
file. The initial system-provided README/manifest device placeholders and
deleted trajectory files were left as found; their original status is saved
in `redteam/initial_status.txt`.
