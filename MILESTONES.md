# Reproduction Milestones

**Current: core_claim**

<!-- Milestone levels (update "Current" above as you progress):
  none             — just started, no meaningful progress yet
  method_runs      — the paper's method executes end-to-end without errors
  core_claim       — minimum experiment supports the paper's central claim
  core_claim_plus  — core claim reproduced on additional settings
  secondary_claims — secondary results or contributions reproduced
  majority         — more than half of reported results reproduced
  near_complete    — most results reproduced, only minor gaps remain
  full             — all reported results reproduced
-->

## Progress Log

### [2026-05-26 23:57] - none
- Read the paper summary and implementation details for PRISM, including the radial DFT magnitude/phase feature extraction and LDA classifier.
- Implemented the paper's actual PRISM computation in `method/`, a method-agnostic Tiny GenImage loader in `data/`, and reusable evaluation code in `eval/`.
- Added scripts, environment setup, scoring references, and documentation needed to submit the first offline reproduction job.

### [2026-05-27 00:22] - none
- Cancelled the first CPU job after logs showed it was progressing correctly but would likely exceed the one-hour CPU-job limit because it extracted PRISM features separately for attribution and binary evaluation.
- Refactored the method runner to extract PRISM features once and reuse them for both LDA evaluations.

### [2026-05-27 01:00] - core_claim
- Completed the optimized CPU reproduction job on Tiny GenImage.
- Reproduced PRISM source attribution at 88.30% accuracy and 88.11 weighted F1 versus the paper's 88.47% accuracy and 88.25 weighted F1.
- Reproduced PRISM real-vs-generated detection at 94.31% accuracy and 94.31 weighted F1 versus the paper's 95.06% accuracy and 95.05 weighted F1.
- Preserved the paper's compact 384-feature representation as the constraint metric.

## Stop Justification
- Reproduction reached `core_claim`. The remaining unreproduced scope is broader benchmark coverage beyond the public Tiny GenImage experiment.
