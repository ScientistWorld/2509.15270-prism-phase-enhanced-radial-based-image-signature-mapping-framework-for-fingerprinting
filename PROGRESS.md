# Progress

## What Works

- Paper analysis is complete enough to implement the core PRISM experiment.
- The PRISM feature extractor implements 64 radial bins over centered 2D FFT magnitude and phase for each RGB channel, producing the paper's 384-dimensional representation.
- The method pipeline trains an LDA classifier on extracted features and writes prediction artifacts for source attribution.
- The method runner now derives real-vs-generated predictions by collapsing the source-attribution predictions into binary labels.
- The runner now also emits attribution ablations for magnitude-only and phase-only PRISM feature slices without rerunning FFT extraction.
- The binary Tiny GenImage output now collapses the source-attribution PRISM predictions into real/fake labels instead of training a separate binary classifier.
- Evaluation is separated from the method and scores prediction CSV artifacts without importing from `method/`.
- The canonical evaluator reports the full declared metric set: accuracy, balanced accuracy, weighted precision/recall/F1, macro F1, worst-class recall, and feature count.
- The evaluation entry points now accept multiple prediction CSVs, so the packaged run can score PRISM, ablations, and the majority baseline together.
- The majority baseline reads label columns directly rather than decoding images, making it cheap enough to include in the standard reproduction path.
- The completed reproduction reaches `secondary_claims` on the paper's public Tiny GenImage benchmark: source attribution plus the secondary real-vs-fake detection result.

## Results

- `scoring/scores.json` contains reproduced PRISM metrics from the completed CPU job.
- Tiny GenImage attribution:
  - Paper: 88.47 accuracy, 88.25 weighted F1, 384 features.
  - Reproduction: 88.30 accuracy, 88.11 weighted F1, 384 features.
- Tiny GenImage attribution ablation:
  - Full PRISM: 88.30 accuracy, 88.11 weighted F1, 384 features.
  - Magnitude-only PRISM: 86.44 accuracy, 86.14 weighted F1, 192 features.
  - Phase-only PRISM: 56.69 accuracy, 46.14 weighted F1, 192 features.
  - Full magnitude-plus-phase PRISM improves attribution accuracy by 1.86 percentage points over magnitude alone on this public benchmark.
- Tiny GenImage binary detection:
  - Paper: 95.06 accuracy, 95.05 weighted F1, 384 features.
  - Reproduction: 95.23 accuracy, 95.23 weighted F1, 384 features.
- `python validate.py --compare` passes and shows the reproduced metrics are within about one percentage point of the paper's reported values.
- Current milestone: `secondary_claims`.

## Audit

- Implementation fidelity: passed. The method code contains the paper's PRISM computation rather than a learned surrogate or hardcoded output.
- Evaluation honesty: passed. The evaluator consumes `outputs/prism_predictions.csv`; `scoring/scores.json` is recomputed from 14,000 prediction rows and is not copied from the paper.
- Deliverables: passed. Briefing, scoring targets, scripts, download script, and container setup are present. The login-node Python lacks `scikit-learn`, but the container setup log shows the package is installed in the packaged job environment where the reproduction ran.

## Remaining

- Broader paper coverage remains: PRISM-36K and additional forensic/deepfake benchmarks were not reproduced.
- A future expansion could add more public datasets and additional baselines, using the current evaluator contract.
- A CPU verification job is pending to regenerate canonical score files with the majority baseline included.

## Issues

- The paper's PRISM-36K download link was not reachable from this environment, so the reproduction targets the paper's public Tiny GenImage evaluation instead.
- Tiny GenImage is large enough that the initial setup depends on completing the Hugging Face dataset download before submitting the offline job.
- The first CPU job was cancelled after 20 minutes because it was healthy but too slow for the one-hour CPU job limit when attribution and binary extraction were duplicated. The runner has been refactored to avoid that duplicated FFT pass.
- The second CPU job completed successfully in the one-hour limit after sharing feature extraction across tasks.
- The third CPU job completed successfully and added magnitude-only and phase-only ablation scores to `scoring/scores.json`.
- Final verification regenerated the score files after tightening binary detection to use attribution-derived real/fake predictions and after extending the canonical score file to emit the full declared metric set.
- A login-node smoke run of the new baseline path was not possible because the host Python does not have runtime dependencies such as `tqdm`; syntax validation and `validate.py --compare` passed, and the packaged container has the dependencies in `environment/setup.sh`.

## Deviations from Paper

- Dataset: the paper's PRISM-36K link was unavailable, so the runnable core experiment uses the paper's Tiny GenImage benchmark via the public Hugging Face mirror `TheKernel01/Tiny-GenImage`.
- Label set: the public Tiny GenImage mirror includes an SD14 label in addition to the generators listed in the paper's summarized Tiny GenImage table. The first run keeps the mirror's labels intact and documents this as a dataset-source deviation, not an algorithm change.
- Algorithm: no method substitution. The reproduction implements PRISM's radial Fourier magnitude and phase features followed by LDA.
