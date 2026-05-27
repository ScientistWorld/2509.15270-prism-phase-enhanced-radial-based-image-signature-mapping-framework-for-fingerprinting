# Overview

PRISM is a black-box image provenance method for attributing generated images to their source model. It uses compact radial Fourier magnitude and phase features, followed by an LDA classifier.

This workspace reproduces the core computation on Tiny GenImage, one of the paper's reported public benchmarks. The environment includes method code, dataset loading, prediction scoring, reference metrics, and scripts for download, reproduction, and evaluation.

---

# Problem Definition

The task is to identify the origin of an image in a black-box setting where only the final image is available. The main problem is multi-class source attribution: given an image from Tiny GenImage, predict whether it is real or which generator produced it.

The secondary task is binary real-vs-fake detection on the same benchmark. This tests whether the attribution setting, which is harder than binary detection, also preserves performance on the conventional synthetic-image detection task.

Inputs are RGB images. Outputs are class labels from the benchmark metadata. Evaluation must use the published train/validation split and must not use generator internals, prompts, model weights, or image-generation APIs.

---

# Method

PRISM extracts handcrafted frequency-domain signatures from each image and classifies them with Linear Discriminant Analysis.

For each RGB channel, the method computes a two-dimensional DFT, shifts the zero-frequency component to the center, and derives both log magnitude and phase. It then averages these spectra over 64 evenly spaced radial annuli. Magnitude features are arithmetic means of log magnitude values. Phase features are the cosine of the circular mean angle within each annulus.

The three color channels are concatenated, producing `3 * 2 * 64 = 384` features per image. Features are standardized using the training split, then LDA is trained as the supervised projection and classifier. For the secondary real-vs-fake task, this reproduction collapses the same source-attribution predictions into binary labels.

The original paper evaluates this pipeline on PRISM-36K and four public benchmarks. This reproduction uses the cited Tiny GenImage benchmark because the PRISM-36K reduced release URL was not reachable from this node.

---

# Evaluation Protocol

Evaluation consumes a prediction CSV written by a method run. The CSV contains one row per validation image with experiment name, method name, sample id, true label, and predicted label.

Metrics are accuracy plus weighted precision, weighted recall, and weighted F1, reported as percentages. The reusable evaluator in `eval/` also reports balanced accuracy, macro F1, worst-class recall, and the number of per-image values consumed by the classifier so future methods are scored on both average performance and class-coverage constraints. It reads only prediction artifacts and writes `scoring/scores.json`.

The reproduced benchmark uses Tiny GenImage's published training and validation split: 28,000 training images and 7,000 validation images. The attribution experiment predicts source labels. The binary experiment predicts real versus fake labels by collapsing the attribution model's source prediction into real versus generated, rather than fitting a separate task-specific binary classifier.

The main constraint metric is representation dimensionality. Improvements should not simply replace a compact image-level representation with a much larger one unless that resource increase is reported and scored.

---

# Reproduction Log

### Iteration 1: gpt-5.5
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 18m | **GPU**: 0.0h
- **Jobs**: 2 total (1 completed, 0 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 2: gpt-5.5
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 2m | **GPU**: 0.0h
- **Jobs**: 2 total (1 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-05-27 01:22] - secondary_claims
- Audited the inherited implementation and confirmed it implements PRISM rather than a surrogate: radial DFT log-magnitude and cosine circular phase features, 64 bins per RGB channel, standardized features, and LDA classification.
- Confirmed evaluator separation: `eval/` reads prediction CSV artifacts and does not import from `method/`.
- Revalidated `scoring/scores.json` with `python validate.py --compare`.
- Confirmed the same PRISM run reproduces the paper's secondary GenImage real-vs-fake detection claim at 94.31% accuracy vs. 95.06% reported.

### [2026-05-27 01:08] - core_claim
- Implemented PRISM's 384-dimensional radial DFT magnitude and cosine phase feature extractor with 64 annular bins per RGB channel.
- Ran the LDA attribution and binary detection pipeline on the public Tiny GenImage benchmark split.
- Reproduced the central claim on GenImage: attribution accuracy 88.30% vs. 88.47% reported, and binary detection accuracy 94.31% vs. 95.06% reported.
- `python validate.py --compare` passes with method/eval separation intact.

### [2026-05-27 00:58] - method_runs
- The paper's PRISM pipeline executed end-to-end and wrote reusable prediction artifacts under `outputs/`.
- `scripts/evaluate.sh` scored the prediction CSV without importing from `method/`.

</details>

### Iteration 3: gpt-5.5
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 6m | **GPU**: 0.0h
- **Jobs**: 3 total (2 completed, 0 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 4: gpt-5.5 (designer_stage_1)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 3m | **GPU**: 0.0h
- **Jobs**: 3 total (2 completed, 0 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 5: gpt-5.5 (designer_stage_2)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 4m | **GPU**: 0.0h
- **Jobs**: 3 total (2 completed, 0 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 6: gpt-5.5 (designer_stage_3)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 1m | **GPU**: 0.0h
- **Jobs**: 4 total (3 completed, 0 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 7: gpt-5.5 (packager)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 2m | **GPU**: 0.0h
- **Jobs**: 4 total (3 completed, 0 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 8: gpt-5.5 (verifier)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 9m | **GPU**: 0.0h
- **Jobs**: 4 total (3 completed, 0 failed)

<details>
<summary>Progress Log</summary>

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

</details>


---

# Reproduction Milestones

**Current: secondary_claims**

## Stop Justification
- Completed at milestone `secondary_claims`.
