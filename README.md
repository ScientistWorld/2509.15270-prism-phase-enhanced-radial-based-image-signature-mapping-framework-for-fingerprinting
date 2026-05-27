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

The three color channels are concatenated, producing `3 * 2 * 64 = 384` features per image. Features are standardized using the training split, then LDA is trained as the supervised projection and classifier.

The original paper evaluates this pipeline on PRISM-36K and four public benchmarks. This reproduction uses the cited Tiny GenImage benchmark because the PRISM-36K reduced release URL was not reachable from this node.

---

# Evaluation Protocol

Evaluation consumes a prediction CSV written by a method run. The CSV contains one row per validation image with experiment name, method name, sample id, true label, and predicted label.

Metrics are weighted accuracy, precision, recall, and F1, reported as percentages. The reusable evaluator in `eval/` reads only prediction artifacts and writes `scoring/scores.json`.

The reproduced benchmark uses Tiny GenImage's published training and validation split: 28,000 training images and 7,000 validation images. The attribution experiment predicts source labels. The binary experiment predicts real versus fake labels.

The main constraint metric is feature dimensionality. The paper emphasizes that radial reduction makes the frequency representation compact and practical, so improvements should not simply replace the compact representation with a much larger image descriptor.

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


---

# Reproduction Milestones

**Current: core_claim**

## Stop Justification
- Completed at milestone `core_claim`.
