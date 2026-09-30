import torch

from fine2coarse.engine import IGNORE_INDEX, FocalLoss, build_criterion


def test_focal_with_zero_gamma_matches_cross_entropy():
    torch.manual_seed(0)
    logits = torch.randn(8, 7)
    target = torch.randint(0, 7, (8,))
    focal = FocalLoss(gamma=0.0)(logits, target)
    plain = torch.nn.CrossEntropyLoss()(logits, target)
    assert torch.allclose(focal, plain, atol=1e-6)


def test_focal_down_weights_confident_correct_predictions():
    confident = torch.tensor([[10.0, 0.0, 0.0]])
    unsure = torch.tensor([[0.6, 0.5, 0.5]])
    target = torch.tensor([0])
    loss = FocalLoss(gamma=2.0)
    ratio_focal = loss(unsure, target) / loss(confident, target)
    plain = torch.nn.CrossEntropyLoss()
    ratio_plain = plain(unsure, target) / plain(confident, target)
    assert ratio_focal > ratio_plain


def test_ignored_positions_do_not_contribute():
    logits = torch.randn(4, 7)
    target = torch.tensor([3, IGNORE_INDEX, 1, IGNORE_INDEX])
    loss = FocalLoss(gamma=2.0)(logits, target)
    kept = FocalLoss(gamma=2.0)(logits[[0, 2]], target[[0, 2]])
    assert torch.allclose(loss, kept, atol=1e-6)


def test_all_ignored_returns_zero_without_nan():
    logits = torch.randn(2, 7, requires_grad=True)
    target = torch.full((2,), IGNORE_INDEX)
    loss = FocalLoss(gamma=2.0)(logits, target)
    loss.backward()
    assert loss.item() == 0.0 and not torch.isnan(logits.grad).any()


def test_class_weights_reach_the_focal_loss():
    logits = torch.tensor([[0.2, 0.9, 0.1]])
    target = torch.tensor([0])
    light = FocalLoss(gamma=1.0, weight=torch.tensor([1.0, 1.0, 1.0]))(logits, target)
    heavy = FocalLoss(gamma=1.0, weight=torch.tensor([5.0, 1.0, 1.0]))(logits, target)
    assert heavy > light


def test_build_criterion_rejects_unknown_loss():
    assert isinstance(build_criterion("focal", 2.0, None), FocalLoss)
    try:
        build_criterion("hinge", 2.0, None)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_label_smoothing_reaches_cross_entropy():
    criterion = build_criterion("ce", 2.0, None, label_smoothing=0.1)
    assert isinstance(criterion, torch.nn.CrossEntropyLoss)
    assert criterion.label_smoothing == 0.1
    assert criterion.ignore_index == IGNORE_INDEX


def test_label_smoothing_with_focal_is_rejected():
    try:
        build_criterion("focal", 2.0, None, label_smoothing=0.1)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
