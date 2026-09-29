"""Accuracy, macro-F1, weighted-F1, per-class scores, confusion matrix."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Sequence[str],
) -> dict:
    """Return the metric block the project spec requires for every system."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    label_ids = list(range(len(labels)))

    report = classification_report(
        y_true,
        y_pred,
        labels=label_ids,
        target_names=list(labels),
        output_dict=True,
        zero_division=0,
    )
    per_class = {
        label: {
            "precision": report[label]["precision"],
            "recall": report[label]["recall"],
            "f1": report[label]["f1-score"],
            "support": int(report[label]["support"]),
        }
        for label in labels
    }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=label_ids, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=label_ids, average="weighted", zero_division=0)),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=label_ids).tolist(),
        "labels": list(labels),
        "n_examples": int(len(y_true)),
    }


def format_metrics(metrics: dict) -> str:
    return (
        f"accuracy={metrics['accuracy']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f} "
        f"weighted_f1={metrics['weighted_f1']:.4f}"
    )
