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

Stronger classifier, reduced-training, magnitude-only, and phase-only runs
were implemented under `redteam/`, but training was not launched after the
reference interface failed. Their hidden-test scores and score ordering are
**unavailable**, not ties. The real majority baseline is included in the
interface-only follow-up; its evaluation status will be recorded alongside
the shortcuts.

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

The follow-up job runs the cheap trials through the immutable no-argument
interface. No shortcut is called successful without a successful hidden-test
run; failures and missing scores are not ties. Each folder contains its own
`run.sh` that constructs the candidate and runs both official scripts.

Results will be finalized from the follow-up job logs.

## Repair needed before scoring can be audited

Repair the package invocation in the next iteration, for example by executing
the evaluators as modules from the workspace root. Then rerun the implemented
quality controls, three seeds, and shortcut bundle. Fixing the import error is
necessary before empirical ranking or shortcut-payoff claims can be made.
