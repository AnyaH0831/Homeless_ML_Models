"""Decision-focused metrics for ranking people for limited intervention."""

from __future__ import annotations

import numpy as np


def top_k_metrics(y_true, proba, fraction: float = 0.10) -> dict[str, float | int]:
    """Return precision and recall among the highest-risk fraction."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    labels = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(proba, dtype=float)
    if labels.shape[0] != probabilities.shape[0]:
        raise ValueError("y_true and proba must have equal length")
    if labels.size == 0 or labels.sum() == 0:
        return {
            "fraction": fraction,
            "n_selected": 0,
            "precision_at_k": float("nan"),
            "recall_at_k": float("nan"),
            "positive_rate_selected": float("nan"),
        }

    n_selected = max(1, int(np.ceil(labels.size * fraction)))
    selected = np.argsort(-probabilities, kind="mergesort")[:n_selected]
    positives_selected = int(labels[selected].sum())
    total_positives = int(labels.sum())
    return {
        "fraction": fraction,
        "n_selected": n_selected,
        "precision_at_k": positives_selected / n_selected,
        "recall_at_k": positives_selected / total_positives,
        "positive_rate_selected": positives_selected / n_selected,
    }


def top_k_recall(y_true, proba, fractions=(0.05, 0.10, 0.20)) -> dict[str, float]:
    """Convenience mapping for reporting intervention coverage at several cutoffs."""
    return {
        f"recall_at_{int(fraction * 100)}pct": float(
            top_k_metrics(y_true, proba, fraction)["recall_at_k"]
        )
        for fraction in fractions
    }
