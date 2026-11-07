"""Method-agnostic classification metrics for prediction artifacts."""

from __future__ import annotations


def classification_metrics(
    y_true: list[int],
    y_pred: list[int],
    feature_count: float,
) -> dict[str, float]:
    labels = sorted(set(y_true))
    total = len(y_true)
    if total == 0:
        raise ValueError("Cannot score an empty prediction set")

    correct = sum(int(t == p) for t, p in zip(y_true, y_pred))
    per_label = {}
    for label in labels:
        tp = sum(int(t == label and p == label) for t, p in zip(y_true, y_pred))
        fp = sum(int(t != label and p == label) for t, p in zip(y_true, y_pred))
        fn = sum(int(t == label and p != label) for t, p in zip(y_true, y_pred))
        support = sum(int(t == label) for t in y_true)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_label[label] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    weighted_precision = sum(v["precision"] * v["support"] for v in per_label.values()) / total
    weighted_recall = sum(v["recall"] * v["support"] for v in per_label.values()) / total
    weighted_f1 = sum(v["f1"] * v["support"] for v in per_label.values()) / total
    recalls = [v["recall"] for v in per_label.values()]
    f1_values = [v["f1"] for v in per_label.values()]

    return {
        "accuracy": 100.0 * correct / total,
        "balanced_accuracy": 100.0 * sum(recalls) / len(recalls),
        "precision_weighted": 100.0 * weighted_precision,
        "recall_weighted": 100.0 * weighted_recall,
        "f1_weighted": 100.0 * weighted_f1,
        "f1_macro": 100.0 * sum(f1_values) / len(f1_values),
        "worst_class_recall": 100.0 * min(recalls),
        "feature_count": float(feature_count),
    }
