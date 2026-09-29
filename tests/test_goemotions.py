from fine2coarse.config import GOEMOTIONS_ALL_LABELS, GOEMOTIONS_LABELS, NEUTRAL_INDEX
from fine2coarse.data.goemotions import remap_labels


def test_neutral_is_excluded_and_ids_are_contiguous():
    assert len(GOEMOTIONS_LABELS) == 27
    assert "neutral" not in GOEMOTIONS_LABELS
    assert NEUTRAL_INDEX == 27


def test_remap_drops_neutral_and_keeps_order():
    assert remap_labels([0, 27]) == [0]
    assert remap_labels([27]) == []
    assert remap_labels([26, 27]) == [26]
    assert remap_labels([2, 11]) == [2, 11]


def test_remapped_ids_point_at_the_same_emotion():
    for original, name in enumerate(GOEMOTIONS_ALL_LABELS):
        if original == NEUTRAL_INDEX:
            continue
        [remapped] = remap_labels([original])
        assert GOEMOTIONS_LABELS[remapped] == name
