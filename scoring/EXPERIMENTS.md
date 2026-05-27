# Experiment Overview

## genimage_tiny_attribution

Tests source attribution on a public benchmark containing real images and images produced by multiple generators. The experiment asks whether the system can recover the image source from image content alone on held-out validation data.

The submitted method also reports magnitude-only and phase-only variants for this experiment so future agents can check whether the phase-enhanced full descriptor is doing useful work beyond magnitude alone. These are scored as comparison variants under the same prediction-artifact interface.

## genimage_tiny_binary

Tests real-vs-fake detection on the same benchmark split. The experiment asks whether a source-attribution-oriented system also performs well on the simpler binary authenticity task when source predictions are grouped into real versus generated labels.
