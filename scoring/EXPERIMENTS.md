# Experiment Overview

Both experiments are scored from one artifact: a source prediction for every dev image (visible scorer) and every held-out test image (hidden scorer). Ground truth, test set, label space and metric definitions are fixed by `eval/`.

## genimage_tiny_attribution (weight 0.65)

Tests source attribution on a public benchmark that contains real photographs and images from seven generators. It asks whether a system trained on the published training split can recover each held-out image's source from the image alone. Half of the test images are real and the seven generators have 500 images each. Accuracy and weighted F1 measure the overall result, and balanced accuracy, macro F1 and worst-class recall check that no generator is sacrificed.

Reference baselines in the score files: the paper's method with only one of its two feature families (ablations, to check that combining them helps) and a majority-class predictor (floor).

## genimage_tiny_binary (weight 0.35)

Tests real-vs-fake detection on the same held-out images. It is derived from the source prediction (real vs any generator) and is not trained or submitted separately. It checks that the fine-grained attribution task also solves the conventional detection task. The test is balanced (3,500 / 3,500), so accuracy and balanced accuracy coincide, and worst-class recall guards against trading real recall for fake recall.
