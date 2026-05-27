# Evaluation Targets

## Primary: genimage_tiny_attribution

Tests whether a system can identify the image source on Tiny GenImage under the benchmark's published split.

- **Metric**: `accuracy` and `f1_weighted` (higher is better)
- **Constraint**: `feature_count` (lower is better for comparable accuracy)
- **Comparison check**: compare any submitted variants under the same prediction-artifact interface; improvements should increase accuracy without relying on unreported resource growth
- **How to run**: produce a prediction CSV and pass it to `bash scripts/evaluate.sh`

## Secondary: genimage_tiny_binary

Tests whether a submission distinguishes real images from synthetic images on Tiny GenImage.

- **Metric**: `accuracy` and `f1_weighted` (higher is better)
- **Constraint**: `feature_count` (lower is better for comparable accuracy)
- **How to run**: produce a prediction CSV and pass it to `bash scripts/evaluate.sh`
