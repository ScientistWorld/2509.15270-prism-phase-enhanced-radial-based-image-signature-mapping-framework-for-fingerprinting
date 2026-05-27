# Evaluation Protocol

Evaluation consumes a prediction CSV written by a method run. The CSV contains one row per validation image with experiment name, method name, sample id, true label, and predicted label.

Metrics are accuracy plus weighted precision, weighted recall, and weighted F1, reported as percentages. The reusable evaluator in `eval/` also reports balanced accuracy, macro F1, worst-class recall, and the number of per-image values consumed by the submission so future methods are scored on both average performance and class-coverage constraints. It reads only prediction artifacts and writes `scoring/scores.json`.

The reproduced benchmark uses Tiny GenImage's published training and validation split: 28,000 training images and 7,000 validation images. The attribution experiment predicts source labels. The binary experiment predicts real versus fake labels. Methods may emit either or both experiment outputs as long as they follow the prediction CSV contract.

The main constraint metric is representation dimensionality. Improvements should not simply replace a compact image-level representation with a much larger one unless that resource increase is reported and scored.
