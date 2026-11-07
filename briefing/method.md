# Method

PRISM extracts handcrafted frequency-domain signatures from each image and classifies them with Linear Discriminant Analysis.

For each RGB channel, the method computes a two-dimensional DFT, shifts the zero-frequency component to the center, and derives both log magnitude (`log(|F|+1)`) and phase. It then aggregates these spectra over 64 evenly spaced radial annuli. Magnitude features are arithmetic means of log magnitude values. Phase features are the cosine of the circular mean angle within each annulus.

The three color channels are concatenated, producing `3 * 2 * 64 = 384` features per image. Features are standardized on the training data, then LDA (SVD solver) is trained as the supervised projection and classifier. For the real-vs-fake task the same source predictions are collapsed into real/fake, with no retraining, as the paper does.

Reproduction protocol (`method/run_prism.py`):
- Test predictions: fit on all 28,000 training images and predict the 7,000 held-out test images (the paper's published 28k/7k GenImage split).
- Dev predictions: fit on the 22,400 training images outside `data/splits/dev.csv` and predict the 5,600 dev images.
- Ablations from the paper's Fig. 5 reuse the same features: `prism_magnitude_only` (192 magnitude features) and `prism_phase_only` (192 phase features).
- Features are cached in `outputs/cache/prism_features.npz`, so a re-run with a different classifier skips the FFT pass, which takes about 40 minutes on one CPU.

The pipeline is deterministic: refitting under different random seeds gives identical predictions.

The original paper evaluates this pipeline on PRISM-36K and four public benchmarks. This gym uses the cited Tiny GenImage benchmark because the PRISM-36K release URL was not reachable.
