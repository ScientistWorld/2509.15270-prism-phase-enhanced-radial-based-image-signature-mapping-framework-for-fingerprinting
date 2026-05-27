# Experiment Overview

## genimage_tiny_attribution

Tests source attribution on a public benchmark containing real images and images produced by multiple generators. The experiment asks whether the system can recover the image source from image content alone on held-out validation data.

The submitted baseline includes lower-information comparison variants for this experiment. They are scored under the same prediction-artifact interface so future agents can check whether a richer compact representation improves source attribution without changing the benchmark.

## genimage_tiny_binary

Tests real-vs-fake detection on the same benchmark split. The experiment asks whether a source-attribution-oriented system also performs well on the simpler binary authenticity task when source predictions are grouped into real versus generated labels.
