# Evaluation Targets

## Primary: genimage_tiny_attribution

Tests whether a system can identify the image source on Tiny GenImage under the benchmark's published split.

- **Metric**: `accuracy` and `f1_weighted` (higher is better)
- **Constraint**: `feature_count` (lower is better for comparable accuracy)
- **Ablation check**: compare `prism` with `prism_magnitude_only` and `prism_phase_only`; full PRISM should outperform magnitude-only, and phase-only should be substantially weaker
- **How to run**: `bash scripts/method.sh`, then inspect `scoring/scores.json`

## Secondary: genimage_tiny_binary

Tests whether the same prediction pipeline distinguishes real images from synthetic images on Tiny GenImage.

- **Metric**: `accuracy` and `f1_weighted` (higher is better)
- **Constraint**: `feature_count` (lower is better for comparable accuracy)
- **How to run**: `bash scripts/method.sh`, then inspect `scoring/scores.json`
