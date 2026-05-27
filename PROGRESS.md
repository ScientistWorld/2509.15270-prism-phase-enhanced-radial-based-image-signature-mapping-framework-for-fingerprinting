# Progress

## What Works

- Paper analysis is complete enough to implement the core PRISM experiment.
- The PRISM feature extractor implements 64 radial bins over centered 2D FFT magnitude and phase for each RGB channel, producing the paper's 384-dimensional representation.
- The method pipeline trains an LDA classifier on extracted features and writes prediction artifacts for both source attribution and real-vs-generated detection.
- The method runner now extracts PRISM features once and reuses the same representation for attribution and binary LDA classifiers.
- Evaluation is separated from the method and scores prediction CSV artifacts without importing from `method/`.

## Results

- No reproduced metrics yet. The first compute job will run PRISM on Tiny GenImage and write `scoring/scores.json`.
- Paper reference numbers recorded in `scoring/reference.json`:
  - Tiny GenImage attribution: PRISM accuracy 88.47, weighted F1 88.25.
  - Tiny GenImage binary detection: PRISM accuracy 95.06, weighted F1 95.05.

## Remaining

- Finish the Tiny GenImage download.
- Submit the first offline run through `scripts/run.sh`.
- Inspect `scoring/scores.json` after the job and compare against `scoring/reference.json`.
- If the run is valid and the method separates generators substantially above trivial baselines, update the milestone to `core_claim`; otherwise diagnose and resubmit.

## Issues

- The paper's PRISM-36K download link was not reachable from this environment, so the reproduction targets the paper's public Tiny GenImage evaluation instead.
- Tiny GenImage is large enough that the initial setup depends on completing the Hugging Face dataset download before submitting the offline job.
- The first CPU job was cancelled after 20 minutes because it was healthy but too slow for the one-hour CPU job limit when attribution and binary extraction were duplicated. The runner has been refactored to avoid that duplicated FFT pass.

## Deviations from Paper

- Dataset: the paper's PRISM-36K link was unavailable, so the runnable core experiment uses the paper's Tiny GenImage benchmark via the public Hugging Face mirror `TheKernel01/Tiny-GenImage`.
- Label set: the public Tiny GenImage mirror includes an SD14 label in addition to the generators listed in the paper's summarized Tiny GenImage table. The first run keeps the mirror's labels intact and documents this as a dataset-source deviation, not an algorithm change.
- Algorithm: no method substitution. The reproduction implements PRISM's radial Fourier magnitude and phase features followed by LDA.
