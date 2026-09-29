from fine2coarse.metrics import compute_metrics

LABELS = ("a", "b", "c")


def test_perfect_predictions():
    metrics = compute_metrics([0, 1, 2], [0, 1, 2], LABELS)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["per_class"]["b"]["support"] == 1


def test_missing_class_does_not_crash_macro_f1():
    metrics = compute_metrics([0, 0, 1], [0, 0, 0], LABELS)
    assert metrics["per_class"]["c"]["f1"] == 0.0
    assert metrics["confusion_matrix"][1] == [1, 0, 0]
    assert 0.0 < metrics["macro_f1"] < 1.0
