# Reproduction Milestones

**Current: none**

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

## Stop Justification
- Not stopped. The first reproduction job is being prepared.
