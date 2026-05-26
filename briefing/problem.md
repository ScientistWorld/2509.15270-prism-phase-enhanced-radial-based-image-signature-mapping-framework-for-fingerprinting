# Problem Definition

The task is to identify the origin of an image in a black-box setting where only the final image is available. The main problem is multi-class source attribution: given an image from Tiny GenImage, predict whether it is real or which generator produced it.

The secondary task is binary real-vs-fake detection on the same benchmark. This tests whether the attribution setting, which is harder than binary detection, also preserves performance on the conventional synthetic-image detection task.

Inputs are RGB images. Outputs are class labels from the benchmark metadata. Evaluation must use the published train/validation split and must not use generator internals, prompts, model weights, or image-generation APIs.
