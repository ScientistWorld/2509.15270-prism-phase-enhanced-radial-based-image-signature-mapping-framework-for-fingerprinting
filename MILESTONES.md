# Reproduction Milestones

**Current: secondary_claims**

## Progress Log

### [2026-05-27 02:24] - secondary_claims
- Completed final verification without requiring another compute submission.
- Confirmed validation, shell syntax, Python syntax, method/eval separation, and verified download URLs.
- Tightened method-agnostic documentation and set the final action to `done`.

### [2026-05-27 02:20] - secondary_claims
- Prepared a follow-up packaged run that scores the declared majority baseline alongside PRISM artifacts.
- Extended the canonical evaluators to accept multiple prediction CSVs while preserving method/eval separation.
- Left the milestone at `secondary_claims`; the current scored artifacts remain valid and the combined baseline path is available for future packaged runs.

### [2026-05-27 02:11] - secondary_claims
- Final verification passed after score regeneration.
- `scoring/scores.json` now includes the full declared metric set, and binary detection scores come from attribution-derived predictions.
- Method-agnostic docs were tightened to avoid leaking the paper's implementation details to from-scratch agents.

### [2026-05-27 02:06] - secondary_claims
- Tightened the canonical evaluator so `scoring/scores.json` reports all metrics declared in `scoring/reference.json`.
- Updated the binary Tiny GenImage experiment to collapse the attribution model's source predictions into real/fake labels, matching the paper's no task-specific retraining claim.
- Prepared a fresh CPU submission to rerun PRISM and regenerate score artifacts.

### [2026-05-27 01:49] - secondary_claims
- Completed the ablation CPU job successfully.
- Added scored magnitude-only and phase-only PRISM variants for Tiny GenImage attribution.
- Validated the updated scoring contract and reproduced scores with `python validate.py --compare`.

### [2026-05-27 01:04] - secondary_claims
- Validated the packaged PRISM reproduction with `python validate.py --compare`.
- Reproduced Tiny GenImage attribution within 0.2 percentage points of the paper's reported accuracy.
- Reproduced the secondary real-vs-fake GenImage experiment within about 0.8 percentage points of the paper's reported accuracy.

### [2026-05-27 00:58] - core_claim_plus
- Completed the paper's PRISM radial Fourier feature extraction and LDA pipeline on Tiny GenImage.
- Produced method prediction artifacts and scored them through the method-agnostic evaluator.

### [2026-05-27 00:23] - method_runs
- Verified that the PRISM method executes end-to-end on local downloaded benchmark data.

## Stop Justification
- Completed at milestone `secondary_claims`.
