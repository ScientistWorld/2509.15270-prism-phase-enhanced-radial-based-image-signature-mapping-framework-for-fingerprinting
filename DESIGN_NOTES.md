# Train/Test Evaluation Design Notes

## Checked

- Read `scripts/evaluate_train.sh` and `scripts/evaluate_test.sh` side by side. The train wrapper calls `eval/train/evaluate.py` and writes `scoring/scores_train.json`; the test wrapper calls `eval/test/evaluate.py` and writes `scoring/scores_test.json`.
- Checked the evaluators for method detachment. Both consume the prediction CSV artifact produced by a method and do not import from `method/`.
- Grepped for train/test cross-import leakage. No cross-imports remain between `eval/train/` and `eval/test/`.
- Checked `scoring/reference.json` for paper-agreement metric names and formulas. The active metrics are scientific quantities, not deviations from reported values.
- Confirmed no `scoring/scores_train.json` or `scoring/scores_test.json` stubs exist.
- Ran static validation only: `bash -n` for both split wrappers, `python -m json.tool scoring/reference.json`, and Python syntax compilation for both split evaluators.

## Changed

- Fixed the strongest leakage pattern in the first pass: the visible train evaluator used a deterministic 50/50 split rule whose complement would reveal the likely test slice. The train evaluator now exposes only a label-stratified 40% visible train slice. The hidden test evaluator selects a disjoint label-stratified 40% slice from the non-train examples with a separate hidden salt, leaving an unused buffer so train membership does not identify test membership exactly.
- Added a top-level `slice` marker to train/test score payloads.
- Replaced the method-specific default prediction path in `scripts/evaluate_train.sh` and `scripts/evaluate_test.sh` with discovery of a single `*predictions.csv` artifact under `outputs/`, while still allowing an explicit CSV path argument.
- Added anti-shortcut metrics to both split evaluators: `balanced_accuracy`, `f1_macro`, and `worst_class_recall`. These complement the existing weighted metrics and make it harder to improve by sacrificing hard generator classes or one side of the binary task.
- Updated `scoring/reference.json` so every metric has `higher_is_better`, `coefficient`, and `description`, and added the new metrics to both existing experiments.

## Runtime Concerns For The Next Iteration

- I did not run either evaluator or regenerate score files, per the design-only instruction. The runner iteration should verify that the new 40/40/buffer split yields enough samples per class and that the train/test scores are close for the reproduced method but not byte-identical.
- The main evaluation script `scripts/evaluate.sh` and `eval/score_predictions.py` remain unchanged by scope. They compute the original full-slice metrics, while the train/test evaluators compute the expanded metric set used by the split harness.
- The split relies on prediction artifacts carrying stable `experiment`, `method`, `sample_id`, `y_true`, `y_pred`, and optional `feature_count` columns. Future methods must keep that artifact interface.
