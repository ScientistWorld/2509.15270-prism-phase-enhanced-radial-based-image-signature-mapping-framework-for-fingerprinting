# Experiment Overview

## genimage_tiny_attribution

Tests source attribution on a public benchmark containing real images and images produced by multiple generators. The experiment asks whether the system can recover the image source from image content alone on held-out validation data.

All submitted variants for this experiment are scored through the same prediction-artifact interface, so future agents can compare source-attribution performance without changing the benchmark.

## genimage_tiny_binary

Tests real-vs-fake detection on the same benchmark split. The experiment asks whether a submission can distinguish authentic images from generated images on held-out validation data.
