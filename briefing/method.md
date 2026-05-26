# Method

PRISM extracts handcrafted frequency-domain signatures from each image and classifies them with Linear Discriminant Analysis.

For each RGB channel, the method computes a two-dimensional DFT, shifts the zero-frequency component to the center, and derives both log magnitude and phase. It then averages these spectra over 64 evenly spaced radial annuli. Magnitude features are arithmetic means of log magnitude values. Phase features are the cosine of the circular mean angle within each annulus.

The three color channels are concatenated, producing `3 * 2 * 64 = 384` features per image. Features are standardized using the training split, then LDA is trained as the supervised projection and classifier.

The original paper evaluates this pipeline on PRISM-36K and four public benchmarks. This reproduction uses the cited Tiny GenImage benchmark because the PRISM-36K reduced release URL was not reachable from this node.
