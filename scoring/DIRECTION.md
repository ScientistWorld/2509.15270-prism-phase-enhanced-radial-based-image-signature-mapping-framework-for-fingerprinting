# Research Direction

Improve black-box source attribution of AI-generated images on Tiny GenImage from compact image-level representations. The reference reaches about 88% accuracy, but its worst generator classes (SD1.5 and Wukong, which share the same latent-diffusion backbone and resolution) stay near 55-60% recall. Real-vs-fake detection is about 95%. Promising directions:

- richer but still compact frequency-domain summaries (other radial or angular aggregations, cross-channel statistics, bin counts)
- better normalisation, and robustness to resolution and JPEG differences between sources
- lightweight non-linear or regularised classifiers on compact descriptors
- class-imbalance handling, since half the data is real

Out of scope: generator internals, prompts, paid APIs, watermark keys, large pretrained visual encoders that replace the compact fingerprint, using held-out test labels from any source, or exploiting file order, container or metadata instead of image content. Dataset-assembly artefacts are reviewed as shortcuts, not progress: JPEG compression traces present only in the real images, and image size, which identifies several generators. See `scoring/CONSTRAINTS.md`.
