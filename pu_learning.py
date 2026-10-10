"""Positive-Unlabeled learning utilities for Lanark note-derived variables.

The caller must provide a defensible estimate of c = P(S=1 | Y=1). The current
cleaned files do not contain a synthetic observation-mask column, so this
module refuses to invent c from prevalence alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import ClassifierMixin, clone


@dataclass
class ElkanNotoPU:
    """Correct a classifier trained on positive-vs-unlabeled labels.

    ``base_estimator`` is fitted with S=1 for explicitly observed positives and
    S=0 for unlabeled rows. The corrected estimate is P(Y=1|x) = P(S=1|x)/c,
    clipped to [0, 1].
    """

    base_estimator: ClassifierMixin
    c: float

    def __post_init__(self) -> None:
        if not 0.0 < self.c <= 1.0:
            raise ValueError("c must satisfy 0 < c <= 1")
        self.model_ = clone(self.base_estimator)

    def fit(self, X, observed_positive) -> "ElkanNotoPU":
        labels = np.asarray(observed_positive, dtype=int)
        if not set(np.unique(labels)).issubset({0, 1}):
            raise ValueError("observed_positive must contain only 0/1 values")
        if labels.sum() == 0 or labels.sum() == len(labels):
            raise ValueError("PU fitting requires both positives and unlabeled rows")
        self.model_.fit(X, labels)
        return self

    def predict_proba(self, X) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("Call fit before predict_proba")
        selected_probability = self.model_.predict_proba(X)[:, 1]
        corrected = np.clip(selected_probability / self.c, 0.0, 1.0)
        return np.column_stack([1.0 - corrected, corrected])


def estimate_c_from_positive_holdout(selected_probability, positive_mask) -> float:
    """Estimate c from held-out known positives only.

    This is the standard Elkan-Noto estimator. ``positive_mask`` identifies
    examples known to have Y=1 in a calibration set; it must not be inferred
    from the same rows used to fit the classifier.
    """
    probabilities = np.asarray(selected_probability, dtype=float)
    mask = np.asarray(positive_mask, dtype=bool)
    if mask.sum() == 0:
        raise ValueError("At least one known positive is required to estimate c")
    c = float(probabilities[mask].mean())
    return float(np.clip(c, 1e-6, 1.0))


def require_observation_mask(frame, variable: str) -> np.ndarray:
    """Return an explicit S mask or fail rather than treating missing as zero."""
    candidates = [f"{variable}_observed", f"{variable}_labeled", f"{variable}_selection"]
    for column in candidates:
        if column in frame:
            return frame[column].fillna(0).astype(int).to_numpy()
    raise ValueError(
        f"No explicit observation mask found for {variable}. Add one of: {candidates}; "
        "missing values alone cannot identify the PU labeling propensity c."
    )
