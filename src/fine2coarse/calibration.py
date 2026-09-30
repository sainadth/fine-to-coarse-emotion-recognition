"""Post-hoc decision offsets and seed ensembling, fitted on dev only.

MELD is ~47% neutral, so an argmax classifier trained with cross-entropy leans
towards the majority class. Adding one offset per class to the log-probabilities
and choosing the offsets that maximise *dev* macro-F1 moves that decision
boundary without retraining. The test split is only ever transformed with offsets
already fixed on dev, which keeps this within the "no hyperparameter selection on
test" rule.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import f1_score

EPSILON = 1e-12


def apply_offsets(probabilities: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Argmax of log-probabilities shifted by one additive offset per class."""
    return (np.log(np.asarray(probabilities) + EPSILON) + offsets).argmax(axis=1)


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_labels: int) -> float:
    return f1_score(y_true, y_pred, labels=list(range(num_labels)), average="macro", zero_division=0)


def fit_offsets(
    y_true: Sequence[int],
    probabilities: np.ndarray,
    bound: float = 3.0,
    step: float = 0.1,
    rounds: int = 3,
) -> np.ndarray:
    """Coordinate ascent on dev macro-F1 over a grid of per-class offsets.

    Returns offsets centred to mean zero (argmax is shift-invariant, so centring
    only makes them easier to read). Ties keep the smaller move, so a class whose
    offset does not matter stays at 0.
    """
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    num_labels = probabilities.shape[1]
    grid = np.round(np.arange(-bound, bound + step / 2, step), 6)
    grid = grid[np.argsort(np.abs(grid), kind="stable")]  # try small moves first

    offsets = np.zeros(num_labels)
    best = _macro_f1(y_true, apply_offsets(probabilities, offsets), num_labels)
    for _ in range(rounds):
        improved = False
        for k in range(num_labels):
            for value in grid:
                trial = offsets.copy()
                trial[k] = value
                score = _macro_f1(y_true, apply_offsets(probabilities, trial), num_labels)
                if score > best + 1e-12:
                    best, offsets, improved = score, trial, True
        if not improved:
            break
    return offsets - offsets.mean()


def average_probabilities(runs: Sequence[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Mean class probabilities over runs that predicted the same examples in the same order."""
    if not runs:
        raise ValueError("no runs to average")
    y_true = np.asarray(runs[0]["y_true"])
    for run in runs[1:]:
        if not np.array_equal(np.asarray(run["y_true"]), y_true):
            raise ValueError("runs disagree on y_true; they did not predict the same split in the same order")
    return y_true, np.mean([np.asarray(run["probs"]) for run in runs], axis=0)
