# Evaluation Protocol

Evaluation consumes a prediction CSV written by a method run. The CSV contains one row per validation image with experiment name, method name, sample id, true label, and predicted label.

Metrics are weighted accuracy, precision, recall, and F1, reported as percentages. The reusable evaluator in `eval/` reads only prediction artifacts and writes `scoring/scores.json`.

The reproduced benchmark uses Tiny GenImage's published training and validation split: 28,000 training images and 7,000 validation images. The attribution experiment predicts source labels. The binary experiment predicts real versus fake labels.

The main constraint metric is feature dimensionality. The paper emphasizes that radial reduction makes the frequency representation compact and practical, so improvements should not simply replace the compact representation with a much larger image descriptor.
