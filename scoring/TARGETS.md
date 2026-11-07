# Evaluation Targets

## Primary: genimage_tiny_attribution

Source attribution (Real + 7 generators) on the 7,000 held-out Tiny GenImage validation images, trained on the published 28,000-image training split.

- **Benefit metrics**: `accuracy` (primary), `f1_weighted`, `precision_weighted` (higher is better)
- **Constraint metrics**: `balanced_accuracy`, `f1_macro`, `worst_class_recall`. Half of the test set is real, so these keep a method from gaining accuracy by giving up minority generators.
- **Paper**: 88.47 accuracy / 88.25 weighted F1 (Table 1, GenImage)
- **Ablation check**: the reference's magnitude-only and phase-only variants must score below the full method (paper Fig. 5: phase alone is insufficient, phase plus magnitude beats magnitude alone)
- **Floor**: majority-class predictor (50% accuracy, 12.5% balanced accuracy)
- **How to run**: write predictions under `outputs/` (see `briefing/evaluation.md`), then `bash scripts/evaluate_train.sh` (dev) / `bash scripts/evaluate_test.sh` (hidden test)

## Secondary: genimage_tiny_binary

Real-vs-fake detection on the same images, derived from the source prediction without a separate detector.

- **Benefit metrics**: `accuracy` (primary), `f1_weighted`, `precision_weighted`
- **Constraint metric**: `worst_class_recall` (the lower of real and fake recall)
- **Paper**: 95.06 accuracy / 95.05 weighted F1 (Table 2), against 82.20 for the GenImage paper's detector
- **How to run**: scored automatically from the same prediction files

## Policy constraint (not scored)

Compact, black-box image-level representation: see `scoring/CONSTRAINTS.md`. The representation size cannot be verified from predictions, so it is enforced by review, not by the score.
