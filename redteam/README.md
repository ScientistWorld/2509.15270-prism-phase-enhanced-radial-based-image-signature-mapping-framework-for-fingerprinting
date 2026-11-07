# Scoring audit

Run `python redteam/bundle.py` in the supplied compute environment. Each named
folder has one `run.sh`; it prepares its method artifact and executes the
unaltered `bash scripts/evaluate_train.sh` and then
`bash scripts/evaluate_test.sh`, both with **no arguments**. Each is a fresh
process. Neither evaluator code nor its Python environment is changed.

`scores_test.json`, `scores_train.json`, `status.json`, and logs in each folder
are the evidence. Large deterministic feature caches and prediction CSVs are
ignored; each run can regenerate its CSV. If the reference cache is absent,
it is recreated by running the visible method on the published training split.
Reference/quality runs use the
actual PRISM implementation in `method/` and reuse its deterministic extracted
features, while refitting the classifier for each seed. Shortcut code only
uses visible metadata, the CSV contract, or legitimately trained predictions.
It never reads hidden split seeds, hidden test labels, or hidden score files.

Ranks use the literal coefficients and weights specified in the audit request.
`rank_delta` covers the proposed `prism` method across both tasks, relative to
the initial `scoring/scores_test.json`. `rank_delta_all_methods` also sums the
ablation entries, for transparency. All normal variants retain the reference
ablation controls. There is no double application of `higher_is_better`.

The initial tree already had device placeholders at README.md and manifest.json
and deleted trajectory files. `initial_status.txt` records this system state;
the audit does not change those paths.

Both submitted jobs found an import error in the unchanged evaluators before
any scoring. `REDTEAM.md` records that blocker. The first job stopped at the
reference; `python redteam/bundle.py --interface-only` then exercised all cheap
candidates. Standalone scripts now propagate evaluator failures as nonzero
exit codes; the job collector itself completes after preserving their logs.
