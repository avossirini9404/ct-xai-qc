"""Classification performance."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


def balanced_accuracy(y_true: np.ndarray, probs: np.ndarray) -> float:
    return float(balanced_accuracy_score(y_true, probs.argmax(axis=1)))


def macro_auc(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Macro one-vs-rest AUC, or the ordinary AUC for two classes.

    Returns NaN when a split does not contain every class, which happens with the
    small subsets used in the smoke tests.
    """
    y_true = np.asarray(y_true)
    n_classes = probs.shape[1]
    if len(np.unique(y_true)) < n_classes:
        return float("nan")
    if n_classes == 2:
        return float(roc_auc_score(y_true, probs[:, 1]))
    return float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))


def summarise_performance(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    return {
        "balanced_accuracy": balanced_accuracy(y_true, probs),
        "macro_auc": macro_auc(y_true, probs),
    }
