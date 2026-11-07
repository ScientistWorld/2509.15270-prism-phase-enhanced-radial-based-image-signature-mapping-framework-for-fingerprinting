# Problem Definition

Identify the origin of an image in a black-box setting where only the final image file is available. The main task is multi-class source attribution: given an image from Tiny GenImage, predict whether it is real or which of seven generators produced it (ADM, BigGAN, GLIDE, Midjourney, Stable Diffusion 1.5, VQDM, Wukong). The class label space is the dataset's generator column, integers 0..8 (0 = Real; label 5, SD14, is declared by the dataset card but never occurs).

The secondary task is real-vs-fake detection on the same images. It is not a separate prediction: a source prediction of 0 counts as "real", any generator as "fake". It tests whether solving the harder attribution problem also solves conventional synthetic-image detection.

Data (Tiny GenImage, public Hugging Face mirror, published split):

- Training split: 28,000 labelled images (14,000 real, 2,000 per generator) in `data/downloads/tiny-genimage/data/`.
- Visible development slice: 5,600 of those training images (2,800 real, 400 per generator), listed with labels in `data/splits/dev.csv`.
- Held-out test set: the 7,000 images of the published validation split (3,500 real, 500 per generator), in `data/downloads/tiny-genimage/test/` as `sample_id` + raw image bytes. Their labels are hidden.

`data/genimage.py` iterates both splits and assigns every image a content-derived `sample_id`.

Methods may only use the image content and the training labels. They must not use generator internals, prompts, model weights, image-generation APIs, or any external copy of the benchmark's labels.
