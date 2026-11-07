# Overview

PRISM is a black-box image provenance method for attributing generated images to their source model. It uses compact radial Fourier magnitude and phase features, followed by an LDA classifier.

This workspace reproduces the core computation on Tiny GenImage, one of the paper's reported public benchmarks. The environment includes method code, dataset loading, prediction scoring, reference metrics, and scripts for download, reproduction, and evaluation.

---

# Problem Definition

Identify the origin of an image in a black-box setting where only the final image file is available. The main task is multi-class source attribution: given an image from Tiny GenImage, predict whether it is real or which of seven generators produced it (ADM, BigGAN, GLIDE, Midjourney, Stable Diffusion 1.5, VQDM, Wukong). The class label space is the dataset's generator column, integers 0..8 (0 = Real; label 5, SD14, is declared by the dataset card but never occurs).

The secondary task is real-vs-fake detection on the same images. It is not a separate prediction: a source prediction of 0 counts as "real", any generator as "fake". It tests whether solving the harder attribution problem also solves conventional synthetic-image detection.

Data (Tiny GenImage, public Hugging Face mirror, published split):

- Training split: 28,000 labelled images (14,000 real, 2,000 per generator) in `data/downloads/tiny-genimage/data/`.
- Visible development slice: 5,600 of those training images (2,800 real, 400 per generator), listed with labels in `data/splits/dev.csv`.
- Held-out test set: the 7,000 images of the published validation split (3,500 real, 500 per generator), in `data/downloads/tiny-genimage/test/` as `sample_id` + image. Their labels are hidden. Each test image is served as a lossless PNG of its decoded RGB pixels: the pixel array matches what `PIL.Image.open(f).convert("RGB")` returns for the published file. The original file container and metadata are not kept.

`data/genimage.py` iterates both splits, decodes images to RGB and assigns every image a content-derived `sample_id`. Work from decoded pixels: the training files keep their original containers (real photographs are JPEG files, generated images PNG files), but the test inputs do not, so cues taken from the file container do not transfer.

Methods may only use the image content and the training labels. They must not use generator internals, prompts, model weights, image-generation APIs, or any external copy of the benchmark's labels.

---

# Method

PRISM extracts handcrafted frequency-domain signatures from each image and classifies them with Linear Discriminant Analysis.

For each RGB channel, the method computes a two-dimensional DFT, shifts the zero-frequency component to the center, and derives both log magnitude (`log(|F|+1)`) and phase. It then aggregates these spectra over 64 evenly spaced radial annuli. Magnitude features are arithmetic means of log magnitude values. Phase features are the cosine of the circular mean angle within each annulus.

The three color channels are concatenated, producing `3 * 2 * 64 = 384` features per image. Features are standardized on the training data, then LDA (SVD solver) is trained as the supervised projection and classifier. For the real-vs-fake task the same source predictions are collapsed into real/fake, with no retraining, as the paper does.

Reproduction protocol (`method/run_prism.py`):
- Test predictions: fit on all 28,000 training images and predict the 7,000 held-out test images (the paper's published 28k/7k GenImage split).
- Dev predictions: fit on the 22,400 training images outside `data/splits/dev.csv` and predict the 5,600 dev images.
- Ablations from the paper's Fig. 5 reuse the same features: `prism_magnitude_only` (192 magnitude features) and `prism_phase_only` (192 phase features).
- Features are cached in `outputs/cache/prism_features.npz`, so a re-run with a different classifier skips the FFT pass, which takes about 40 minutes on one CPU.

The pipeline is deterministic: refitting under different random seeds gives identical predictions.

The original paper evaluates this pipeline on PRISM-36K and four public benchmarks. This gym uses the cited Tiny GenImage benchmark because the PRISM-36K release URL was not reachable.

---

# Evaluation Protocol

## Submission format

A method writes one or more CSV files under `outputs/` with at least these columns:

| column | meaning |
|---|---|
| `method` | name of the method/variant (one file may hold several methods) |
| `sample_id` | id from `data/genimage.py` (`tr-...` for dev images, `te-...` for test images) |
| `prediction` | predicted source label, an integer in 0..8 |

A `split` column (`dev`/`test`) is conventional but not read. Any other column (true labels, feature counts, scores, confidences) is ignored. Every CSV under `outputs/` with the three columns is read; files without them are skipped.

For dev predictions, the model must not have been fitted on the dev images: fit on the training images that are not in `data/splits/dev.csv`. For test predictions you may refit on the whole training split, as the published protocol does.

## Scorers

- `bash scripts/evaluate_train.sh` scores every method on the 5,600 dev images and writes `scoring/scores_train.json`.
- `bash scripts/evaluate_test.sh` scores every method on the 7,000 held-out test images against hidden labels and writes `scoring/scores_test.json`. The hidden test is the ranking signal.
- `bash scripts/evaluate.sh` runs both when the hidden scorer is present and mirrors the result into `scoring/scores.json`.

Both scorers run without arguments (or take explicit CSV paths) and use the same interface and metrics.

## Validity rules

A method is scored as failed, with every metric 0, if on the scored ids it
- omits any image,
- predicts anything other than an integer label 0..8, or
- gives two different predictions for the same image.

Byte-identical duplicate rows are harmless. Failures are listed under `_failed_methods` in the score file.

## Metrics (percentages)

`genimage_tiny_attribution` (8 true classes; weight 0.65): accuracy, balanced accuracy (mean per-class recall), weighted precision, weighted F1, macro F1, worst-class recall.

`genimage_tiny_binary` (derived real vs fake, balanced; weight 0.35): accuracy, weighted precision, weighted F1, worst-class recall (the lower of real recall and fake recall).

The rank sums, over experiments, the weight times the coefficient-weighted improvement of each metric over the reference (coefficients in `scoring/reference.json`). Accuracy is the primary metric. Half of the images are real, so the class-balanced metrics stop a method from scoring well by giving up the rarer generator classes.

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

### Iteration 9: gpt-5.5 (verifier)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 13m | **GPU**: 0.0h
- **Jobs**: 4 total (3 completed, 0 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 10: gpt-6.1-sol (redteam)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 7m | **GPU**: 0.0h

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 11: claude-opus-5-5 (hardener)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 24m | **GPU**: 0.0h
- **Jobs**: 3 total (2 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-11-07] - secondary_claims (hardening)
- Hidden test now scores all 7,000 published validation images against labels kept only in `eval/test/`; methods get label-free, hash-ordered, PNG-re-encoded test inputs.
- Reference rerun from final code: attribution 88.30 (paper 88.47), real-vs-fake 95.23 (paper 95.06); ablation order and majority floor preserved.
- Repaired shortcuts (visible labels, row-index leak, forged truth, coverage drop, feature-count, impossible values, duplicates, container format) now fail or gain nothing; image-size cue left open and documented in `HARDENING.md`.

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

</details>


---

# Reproduction Milestones

**Current: secondary_claims**

## Stop Justification
- Completed at milestone `secondary_claims`.
