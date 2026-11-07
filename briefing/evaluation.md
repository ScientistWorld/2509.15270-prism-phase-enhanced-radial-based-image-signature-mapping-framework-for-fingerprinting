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
