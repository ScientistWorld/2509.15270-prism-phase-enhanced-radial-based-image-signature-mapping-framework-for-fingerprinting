# Evaluation Targets

## Primary: genimage_tiny_attribution

Tests whether a system can identify the image source on Tiny GenImage under the benchmark's published split.

- **Metric**: `accuracy` and `f1_weighted` (higher is better)
- **Constraint**: `feature_count` (lower is better for comparable accuracy)
- **Comparison check**: compare the main submission with any lower-information variants included in the prediction artifact; the full submission should preserve a clear accuracy advantage at a compact representation size
- **How to run**: produce a prediction CSV and pass it to `bash scripts/evaluate.sh`

## Secondary: genimage_tiny_binary

Tests whether the same prediction pipeline distinguishes real images from synthetic images on Tiny GenImage.

- **Metric**: `accuracy` and `f1_weighted` (higher is better)
- **Constraint**: `feature_count` (lower is better for comparable accuracy)
- **How to run**: produce a prediction CSV and pass it to `bash scripts/evaluate.sh`
