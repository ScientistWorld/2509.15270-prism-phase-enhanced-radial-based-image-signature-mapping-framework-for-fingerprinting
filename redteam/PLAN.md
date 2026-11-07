# Job plan

1. Re-evaluate the exact reference CSV through both no-argument scripts.
2. Exercise metadata answer copying, forged truth, coverage reduction, false
   resource counts, impossible values, direct score overwrite, a baseline
   replacement, constant outputs, empty output, null and NaN feature counts.
3. Run an actual majority baseline.
4. Extract unmodified PRISM features once for all published images; refit
   standardization and LDA at seeds 0, 1, and 2. The reference is deterministic.
5. Remove phase/magnitude or 95% of training data as weaker scientific controls.
6. Test RBF SVM, shrinkage LDA, ExtraTrees, and gradient boosting at the same
   384-dimensional input size, each fitted on training data exclusively.

Stop if the exact reference evaluator fails; do not patch the evaluator,
Python environment, or process to make an otherwise broken test pass.

After the job, compare raw metrics and literal weighted coefficient ranks;
document ceilings, reproduction direction, seed spread, and paying shortcuts.
Restore scripts/run.sh, scores, and method/baseline files before completion.
