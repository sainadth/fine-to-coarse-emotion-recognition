import pandas as pd
import pytest

from fine2coarse.config import MELD_CSV_NAMES, MELD_LABELS
from fine2coarse.data.meld import (
    DialogueDataset,
    _relative_speaker_ids,
    load_meld_dialogues,
    load_meld_frame,
)

ROWS = [
    # dialogue 0: five turns, two speakers
    (0, 0, "Joey", "Hey!", "joy"),
    (0, 1, "Ross", "Not now.", "anger"),
    (0, 2, "Joey", "Why not?", "surprise"),
    (0, 3, "Ross", "Because.", "neutral"),
    (0, 4, "Joey", "Fine.", "sadness"),
    # dialogue 1: two turns
    (1, 0, "Monica", "Look at this.", "surprise"),
    (1, 1, "Chandler", "Gross.", "disgust"),
]


@pytest.fixture()
def meld_dir(tmp_path):
    frame = pd.DataFrame(
        [
            {
                "Sr No.": i,
                "Utterance": text,
                "Speaker": speaker,
                "Emotion": emotion,
                "Sentiment": "neutral",
                "Dialogue_ID": dialogue,
                "Utterance_ID": turn,
            }
            for i, (dialogue, turn, speaker, text, emotion) in enumerate(ROWS)
        ]
    )
    for name in MELD_CSV_NAMES.values():
        frame.to_csv(tmp_path / name, index=False)
    return tmp_path


def test_frame_maps_emotions_to_ids(meld_dir):
    frame = load_meld_frame("train", data_dir=meld_dir)
    assert len(frame) == len(ROWS)
    assert frame.loc[0, "label"] == MELD_LABELS.index("joy")


def test_dialogues_keep_boundaries_and_order(meld_dir):
    dialogues = load_meld_dialogues("train", max_turns=24, data_dir=meld_dir)
    assert [len(d) for d in dialogues] == [5, 2]
    assert dialogues[0].utterances[0] == "Hey!"
    assert dialogues[0].utterance_ids == [0, 1, 2, 3, 4]


def test_long_dialogues_are_windowed_without_losing_turns(meld_dir):
    dialogues = load_meld_dialogues("train", max_turns=2, data_dir=meld_dir)
    assert [len(d) for d in dialogues] == [2, 2, 1, 2]
    assert sum(len(d) for d in dialogues) == len(ROWS)


def test_relative_speaker_ids_start_at_one():
    assert _relative_speaker_ids(["A", "B", "A", "C"]) == [1, 2, 1, 3]


def _cached_tokenizer():
    """roberta-base tokenizer if it is already cached locally, else None."""
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained("roberta-base")
    except Exception:
        return None


def test_collate_pads_turns_and_masks_labels(meld_dir):
    tokenizer = _cached_tokenizer()
    if tokenizer is None:
        pytest.skip("roberta-base tokenizer unavailable")
    dialogues = load_meld_dialogues("train", max_turns=24, data_dir=meld_dir)
    dataset = DialogueDataset(dialogues, tokenizer, max_length=16)
    batch = dataset.collate(dialogues)

    assert batch["input_ids"].shape[:2] == (2, 5)
    assert batch["turn_mask"][1].tolist() == [True, True, False, False, False]
    assert batch["labels"][1, 2:].tolist() == [-100, -100, -100]
    assert batch["speaker_ids"][0].tolist() == [1, 2, 1, 2, 1]
    assert (batch["attention_mask"][1, 2:] == 0).all()


def test_speaker_prefix_is_off_by_default(meld_dir):
    dialogues = load_meld_dialogues("train", max_turns=24, data_dir=meld_dir)
    assert dialogues[0].utterances[0] == "Hey!"


def test_speaker_prefix_names_only_the_current_speaker(meld_dir):
    dialogues = load_meld_dialogues("train", max_turns=24, data_dir=meld_dir, speaker_prefix=True)
    assert dialogues[0].utterances[:2] == ["Joey: Hey!", "Ross: Not now."]
    assert dialogues[1].utterances == ["Monica: Look at this.", "Chandler: Gross."]


def test_utterance_view_uses_the_same_prefix(meld_dir):
    from fine2coarse.data.meld import meld_utterance_examples

    plain = meld_utterance_examples("train", data_dir=meld_dir)
    prefixed = meld_utterance_examples("train", data_dir=meld_dir, speaker_prefix=True)
    assert plain[1].text == "Not now."
    assert prefixed[1].text == "Ross: Not now."
    assert [e.label for e in plain] == [e.label for e in prefixed]
