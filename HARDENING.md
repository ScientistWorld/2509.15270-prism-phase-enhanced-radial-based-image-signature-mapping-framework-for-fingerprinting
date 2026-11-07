# Hardening report: PRISM on Tiny GenImage

**Before this iteration the hidden test produced no score at all.** Both scorers died on `ModuleNotFoundError: No module named 'eval'` (REDTEAM.md). With only that import repaired, the old scorer paid out on most of the shortcuts the redteam proposed, plus two new ones found here: a row-index leak and duplicate rows. It also ranked a weaker ablation far above the reference.

**After hardening:**
- The test scores all 7,000 published validation images against hidden labels.
- Better variants rank higher and weaker ones lower.
- Every repaired shortcut fails or gains nothing.
- The reference still reproduces the paper: 88.30 vs 88.47 attribution accuracy, 95.23 vs 95.06 real-vs-fake.

**One major shortcut is left open:** the image-size cue, worth rank +14.4. It is explained below.

Evidence: `redteam/hardening/` holds the harnesses (`before_scores.py`, `after_scores.py`, `size_hybrid.py`) and per-candidate `status.json` / `scores_*.json`. Run summaries are in `before/summary.json`, `v1_original_inputs/summary.json` and `after/summary.json`. Final job: `cbc0a98c-3f8` (CPU, exit 0).

## What changed

| Area | Before | After |
|---|---|---|
| Hidden labels | The labelled validation parquet sat in `data/downloads/`, readable by any method. Rows follow a fixed label cycle (real, ADM, real, BigGAN, …), so the `validation/<idx>` sample id alone gave every label. The parquet `path` field also names the source (`0_adm_85.PNG`). | `scripts/download.sh` puts the labelled validation shards under `eval/test/downloads/` only. `eval/test/export_inputs.py` writes label-free test shards to `data/downloads/tiny-genimage/test/`: `sample_id` = `te-` + SHA-256 of the image bytes, rows sorted by id, no `path`, and each image re-encoded as a lossless PNG of its decoded RGB pixels. Answers live only in `eval/test/labels.csv`. |
| Ground truth / coverage | `y_true`, sample membership and test size were read from the submitted CSV. | Fixed by `eval/test/`: 7,000 ids and labels, label space 0..8. Only `method`, `sample_id` and `prediction` are read. |
| Validity | Out-of-range labels scored as wrong but feature count still paid. Empty or `null` feature counts crashed; `nan` produced a NaN rank. Duplicates were counted repeatedly. | A missing id, a prediction that is not an integer 0..8, or conflicting duplicates mark the method **failed** (all metrics 0). Identical duplicates are harmless. No predictions at all → exit 1. |
| Self-reported numbers | `feature_count` came from the CSV, coefficient −0.2 (fewer features = higher rank). | Removed from scoring because it cannot be verified from predictions. It remains a reviewed policy constraint (`scoring/CONSTRAINTS.md`). |
| Binary task | A separate set of rows with its own `y_true`. | Derived by the scorer from the source prediction (0 = real), as in the paper's "without retraining" setting. |
| Train side | A hash-selected 40% of the *validation* predictions, with the test slice a disjoint 40% of the same file. | A fixed, stratified 20% of the *training* split (`data/splits/dev.csv`, 5,600 images). Dev predictions come from a model fit on the other 80%. |
| Code | The train and test scorers shared `eval/metrics.py`, which broke on import. | Two independent stdlib-only scorers. No-argument runs discover every `*.csv` under `outputs/` and score every method in them. |
| Data revision | `resolve/main`. | Pinned to revision `89c4fe9…`. |

## Do better results now score higher?

"Before" replays each candidate's test predictions through an unmodified copy of the old scorer (`redteam/hardening/old_eval/`), with only the import path repaired, on its archived 40% validation slice and with the old coefficients. The real old test produced no number for anything. "After" is the hardened hidden test. Rank = Σ weight × coefficient × (metric − reference).

| Variant (all on the same 384 PRISM features unless stated) | Before: acc / rank | After: test acc / rank | After: dev acc |
|---|---|---|---|
| Reference: PRISM (standardise + SVD-LDA) | 88.75 / 0 | **88.30 / 0** | 88.43 |
| Better: gradient boosting (HistGB) | 88.96 / +0.90 | **89.24 / +3.19** | 88.20 |
| Shrinkage LDA | 87.89 / −3.36 | 87.70 / −2.12 | 87.39 |
| RBF SVM (C=10, untuned) | 87.00 / −8.28 | 86.71 / −5.81 | 85.36 |
| Weaker: LDA on 5% of training data | 79.93 / −30.17 | 79.10 / −28.91 | 77.96 |
| Weaker: magnitude-only (192 features) | 86.64 / **+31.53** (wrong sign) | 86.44 / **−4.65** | 85.38 |
| Weaker: phase-only (192 features) | 57.25 / −118.82 | 56.69 / −134.08 | 55.96 |

- The ordering is now monotone in quality. The old scorer would have ranked the magnitude-only ablation +31.5 *above* the reference purely because it reports fewer features.
- **Effect vs noise.** PRISM is deterministic: refitting under seeds 0, 1, 2 gives identical predictions, so seed spread is 0. Paired bootstrap 95% intervals for test-accuracy differences (2,000 resamples of 7,000 images):
  - HistGB +0.94 [+0.16, +1.73]
  - shrinkage −0.60 [−0.99, −0.21]
  - SVM −1.59 [−2.39, −0.77]
  - magnitude-only −1.86 [−2.40, −1.33] (the paper's phase-helps claim)
  - 5% data −9.2 [−10.0, −8.4]
  - phase-only −31.6 [−32.8, −30.4]

  All exclude 0.
- **Train/test agreement.** Honest methods land within about 1 point between dev and test; dev models train on 80% of the data, so dev runs slightly lower. Methods tuned to the visible side lose on test: copied dev labels score 100 on dev and 50 on test, and the container rule 92.3 on dev and 41.8 on test.

## Shortcuts: fix and score before → after

Rank is relative to the reference. "Failed" means every metric is 0, so rank −267.55.

| Shortcut | Fix | Before: test acc / rank | After: test acc / rank |
|---|---|---|---|
| Copy visible validation labels (redteam) | Labelled shards moved to `eval/test/`. Methods see only dev labels and label-free test shards (columns: `sample_id`, `image`). | 100 / +46.96 | 50.00 / −179.13 (dev 100, test chance) |
| Row-index leak (`validation/<idx>` → period-14 label cycle) | Content-hash ids; test rows sorted by hash | 100 / +46.96 | 28.96 / −176.88 |
| Forged `y_true` column | Truth comes only from `eval/test/labels.csv` | 100 / +46.96 | 50.00 / −179.13 |
| Drop wrongly predicted rows | Fixed id list; any missing id → failed | 100 / +46.96 | failed / −267.55 |
| Report 0 features | Feature count not scored | 88.75 / +76.80 | 88.30 / 0.00 |
| Impossible labels (99, −1, `nan`, `3.7`) + feature count −10,000 | Non-integer or out-of-range label → failed | 0.00 / **+1743.76** | failed / −267.55 |
| `null` / `nan` feature count | Column ignored | crash / NaN rank | 50.00 / −179.13 (constant predictor) |
| Repeat correct rows | Ids de-duplicated; conflicting duplicates → failed | 98.75 / +40.90 | 88.30 / 0.00 (identical copies); every label per id → failed |
| Empty output | No rows → exit 1, no score | crash | exit 1 (no score) |
| Pre-write a perfect `scores_test.json` | The scorer always rewrites the file from predictions | (no score) | 88.30 / 0.00 (forged file replaced) |
| Replace `baseline/majority.py` | The scorer never runs baselines; reference values are fixed in `scoring/` | (no score) | 88.30 / 0.00 |
| PRISM + "JPEG file ⇒ real" (every real image is a JPEG, every fake a PNG) | Test inputs re-encoded as lossless PNG of the decoded pixels. PRISM features stay bit-identical on all 7,000 images (max diff 0.0), so reference scores are unchanged. | 91.84 / **+12.54** (scored on original files) | 41.84 / −146.08 |
| Format + size lookup table, no pixels | Same PNG re-encoding | 85.71 / +52.26 (old scorer) | 28.57 / −181.00 |

## Shortcuts left open

1. **Image size: major, pays +14.36.**
   - **The cue.** Each Tiny GenImage generator has a fixed output size: BigGAN 128², ADM/GLIDE/VQDM 256², SD1.5/Wukong 512², Midjourney 1024². Real ImageNet photos almost never have those sizes. A size-only lookup gets 78.54 attribution and 99.97 real-vs-fake. On its own it ranks −44.51, because worst-class recall is 0 when one size is shared by several generators.
   - **The payoff.** Combined with PRISM by a hand rule (`size_hybrid.py`: trust size for real/BigGAN/Midjourney and for the size group, PRISM within a group), it reaches 92.87 / 99.97 and rank **+14.36**.
   - **Why it is left open.** Size is a property of the pixel array, present in training and test alike. Removing it means cropping or resampling every image, which changes the data PRISM is measured on and breaks comparability with the paper's Table 1. PRISM itself carries resolution implicitly through its radial bins, so no score can separate honest frequency methods from this cue.
   - **Mitigation.** `scoring/CONSTRAINTS.md` and `scoring/DIRECTION.md` name image size as a dataset artefact reviewed as a shortcut. The post-hoc grader should look for explicit use of image dimensions.
2. **JPEG traces inside real-image pixels: minor.** The container cue is closed, but every real photo passed through JPEG, so 8×8 compression traces remain in its pixels. A simple blockiness detector plus PRISM does not pay (calibration accuracy 64%, rank −105.83). A stronger detector might pay, but the size cue already gives near-perfect real-vs-fake. Flagged in the same constraint.
3. **Public benchmark.** The test images are the public Hugging Face validation split, so anyone with internet can hash-match them and recover labels. This cannot be closed for a public dataset; it is listed as out of scope.
4. **Representation compactness is not scored.** The paper's 384-value descriptor (<0.1 s per image) is a policy constraint. Large pretrained encoders would be a different resource regime and must be caught by review.
5. **Visible dev side is honour-based.** Dev labels are ordinary training labels, so training on dev inflates only the visible score. The hidden test is unaffected.

## Reference score before → after

| | Attribution acc / bal. acc / worst-class recall | Real-vs-fake acc / worst-class recall |
|---|---|---|
| Paper (Tables 1–2) | 88.47 (F1 88.25) | 95.06 (F1 95.05) |
| Before: archived old 40% validation slice (2,800 images); the old test could not run | 88.75 / 81.76 / 58.00 | 95.36 / 93.07 |
| **After: hidden test, all 7,000 validation images** | **88.30 / 81.35 / 56.20** (F1 88.11) | **95.23 / 92.89** (F1 95.23) |
| After: visible dev slice (5,600 training images) | 88.43 / 82.29 / 56.75 | 94.86 / 93.11 |

PRISM's predictions are unchanged; only the scored slice and metric set changed. The reference, its ablations and the majority baseline were rerun end to end from the final code, with fresh feature extraction on the PNG test inputs (`scripts/run.sh`). `scoring/scores_test.json`, `scores_train.json` and `scores.json` contain only those reference runs. `scoring/reference.json` keeps the paper's numbers: 88.47/88.25, 95.06/95.05, 82.20 (GenImage paper detector), 12.5 (random guess).

## Headroom and baselines

- No scored metric is at a bound for the reference:
  - attribution: accuracy 88.3, balanced accuracy 81.3, macro F1 82.7, worst-class recall 56.2 (SD1.5/Wukong confusion);
  - real-vs-fake: 95.2, worst-class recall 92.9.

  An honest compact method (HistGB) already beats the reference significantly. No extra experiment was needed.
- Baselines sit clearly below with room above the reference:
  - majority class: 50.0 accuracy, 12.5 balanced, 0 worst-class; binary 50.0;
  - phase-only: 56.7;
  - magnitude-only: 86.4.
- Nothing depends on wall-clock time, and every metric is recomputed from predictions.

## Weights and coefficients

Experiment weights are unchanged: attribution 0.65, real-vs-fake 0.35. Attribution is the paper's main contribution; real-vs-fake is its secondary claim and is now derived from the same predictions. Removed:
- `feature_count` (unverifiable; it let a failed submission rank +1,744);
- `recall_weighted` (identical to accuracy, so double-counted);
- binary `balanced_accuracy` and `f1_macro` (identical to accuracy / weighted F1 on the balanced 3,500/3,500 test).

The remaining coefficients are as before:
- attribution: accuracy 1.0, F1-w 0.7, precision-w 0.3, balanced accuracy 0.6, macro F1 0.5, worst-class 0.4;
- binary: accuracy 1.0, F1-w 0.7, precision-w 0.3, worst-class 0.4.

All scored metrics are percentages in [0, 100], so a failed submission (all 0) can never outrank a working one.

## Could not fix / caveats

- **The image-size shortcut (+14.36) stays open.** It is documented above and flagged for review.
- **PRISM-36K is still unavailable.** The gym covers the paper's Tiny GenImage results only, as before.
- **The test-input export needs pyarrow and pillow.** `scripts/download.sh` uses `uv run --with pyarrow --with pillow` when the host Python lacks them. The export ran inside the job container, and the pinned URLs return 200. A full from-empty download was not repeated: the 8.4 GB is already present.
- **Infrastructure note.** The first audit job stalled for 45+ minutes in gradient boosting, from OpenMP oversubscription on the single allotted CPU. `scripts/run.sh` now pins native thread pools to 1. Scoring is unaffected.
