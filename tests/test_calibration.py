import numpy as np

from fine2coarse.calibration import (
    apply_offsets,
    average_probabilities,
    fit_offsets,
    pooled_predictions,
    select_by_dev,
)
from fine2coarse.metrics import compute_metrics


def _skewed(n_per_class=(80, 10, 10), seed=0):
    """Probabilities that systematically over-predict class 0."""
    rng = np.random.default_rng(seed)
    y_true = np.concatenate([np.full(n, k) for k, n in enumerate(n_per_class)])
    logits = rng.normal(0, 1, (len(y_true), 3))
    logits[np.arange(len(y_true)), y_true] += 1.0
    logits[:, 0] += 1.5
    probabilities = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    return y_true, probabilities


def test_zero_offsets_are_plain_argmax():
    _, probabilities = _skewed()
    assert (apply_offsets(probabilities, np.zeros(3)) == probabilities.argmax(axis=1)).all()


def test_fitted_offsets_do_not_lower_dev_macro_f1():
    y_true, probabilities = _skewed()
    labels = ("a", "b", "c")
    before = compute_metrics(y_true.tolist(), probabilities.argmax(1).tolist(), labels)["macro_f1"]
    offsets = fit_offsets(y_true, probabilities)
    after = compute_metrics(y_true.tolist(), apply_offsets(probabilities, offsets).tolist(), labels)["macro_f1"]
    assert after > before


def test_offsets_are_centred_and_bounded():
    y_true, probabilities = _skewed()
    offsets = fit_offsets(y_true, probabilities)
    assert abs(offsets.mean()) < 1e-9
    assert offsets.max() - offsets.min() <= 6.0 + 1e-9


def test_average_probabilities_requires_matching_labels():
    a = {"y_true": [0, 1], "probs": [[0.9, 0.1], [0.2, 0.8]]}
    b = {"y_true": [0, 1], "probs": [[0.5, 0.5], [0.4, 0.6]]}
    y_true, mean = average_probabilities([a, b])
    assert y_true.tolist() == [0, 1]
    assert np.allclose(mean, [[0.7, 0.3], [0.3, 0.7]])
    try:
        average_probabilities([a, {"y_true": [1, 1], "probs": b["probs"]}])
    except ValueError:
        return
    raise AssertionError("expected ValueError for mismatched y_true")


def _runs(probs_per_run, y_true):
    return [{"y_true": list(y_true), "probs": p} for p in probs_per_run]


def test_pooled_predictions_is_argmax_of_the_mean():
    y_true = [0, 1]
    runs = _runs([[[0.9, 0.1], [0.4, 0.6]], [[0.3, 0.7], [0.2, 0.8]]], y_true)
    pooled_y, pooled_pred = pooled_predictions(runs)
    assert pooled_y.tolist() == y_true
    assert pooled_pred.tolist() == [0, 1]  # means are [0.6,0.4] and [0.3,0.7]


def test_select_by_dev_picks_the_highest_scoring_candidate():
    y_true, probabilities = _skewed()
    good = _runs([probabilities.tolist()], y_true)
    # a candidate that predicts class 0 for everything scores far worse
    flat = np.zeros_like(probabilities)
    flat[:, 0] = 1.0
    bad = _runs([flat.tolist()], y_true)
    best, scores = select_by_dev({"good": good, "bad": bad}, num_labels=3)
    assert best == "good"
    assert scores["good"] > scores["bad"]


def test_select_by_dev_breaks_ties_towards_fewer_runs():
    y_true = [0, 1, 0, 1]
    probs = [[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]]
    small = _runs([probs], y_true)
    large = _runs([probs, probs, probs], y_true)
    best, scores = select_by_dev({"large": large, "small": small}, num_labels=2)
    assert scores["large"] == scores["small"]
    assert best == "small"
