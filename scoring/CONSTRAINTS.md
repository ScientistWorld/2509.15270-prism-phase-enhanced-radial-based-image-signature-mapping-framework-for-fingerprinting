# Constraints

Scientists should keep the Tiny GenImage dataset and published train/validation split fixed. Evaluation must use the prediction CSV interface consumed by `scripts/evaluate.sh`.

The primary resource constraint is descriptor size. PRISM uses 384 features per image after radial reduction. Methods that improve accuracy by using very large image embeddings or end-to-end vision backbones are outside the intended comparison unless they report the increased feature count and are treated as a different resource regime.

Evaluation code must remain method-agnostic and must not import from `method/`. Data-loading utilities may be shared from `data/`, but the scorer must read method outputs rather than rerunning a method internally.
