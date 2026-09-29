"""Stage II data: MELD as sequential multi-party dialogues.

MELD ships as three CSVs (train/dev/test) with one row per utterance and
``Dialogue_ID`` / ``Utterance_ID`` columns. Dialogue boundaries and utterance
order are preserved: rows are grouped by dialogue and sorted by utterance
index, and dialogues longer than ``max_turns`` are split into consecutive
non-overlapping windows so no turn is dropped and no window crosses a
dialogue boundary.

The same CSVs feed two views:

* utterance view (systems M1, M2) -- each utterance classified independently;
* dialogue view (systems M3, M4) -- a padded [B, T, L] tensor plus a turn mask.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from fine2coarse.config import (
    DATA_DIR,
    MELD_CSV_NAMES,
    MELD_CSV_URLS,
    MELD_LABEL_TO_ID,
    MELD_LABELS,
)
from fine2coarse.data.goemotions import TextClassificationDataset, TextExample
from fine2coarse.utils import get_logger

LOGGER = get_logger()

SPLITS = ("train", "dev", "test")
_QUOTE_FIXES = {
    "\x92": "'", "\x93": '"', "\x94": '"', "\x96": "-", "\x97": "-",
    "\x85": "...", "’": "'", "“": '"', "”": '"',
}


@dataclass
class Dialogue:
    """One dialogue (or one window of a long dialogue)."""

    dialogue_id: int
    utterances: list[str]
    labels: list[int]
    speakers: list[str] = field(default_factory=list)
    utterance_ids: list[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.utterances)


def meld_dir() -> Path:
    return DATA_DIR / "meld"


def download_meld(force: bool = False) -> dict[str, Path]:
    """Fetch the three MELD CSVs from the official repository if missing."""
    target = meld_dir()
    target.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split, url in MELD_CSV_URLS.items():
        path = target / MELD_CSV_NAMES[split]
        if force or not path.exists():
            LOGGER.info("downloading MELD %s -> %s", split, path)
            urllib.request.urlretrieve(url, path)
        paths[split] = path
    return paths


def _clean(text: str) -> str:
    text = str(text)
    for bad, good in _QUOTE_FIXES.items():
        text = text.replace(bad, good)
    return text.strip()


def load_meld_frame(split: str, data_dir: Path | None = None) -> pd.DataFrame:
    """Read one MELD split as a dataframe with clean text and integer labels."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    directory = Path(data_dir) if data_dir else meld_dir()
    path = directory / MELD_CSV_NAMES[split]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/prepare_data.py --dataset meld` first."
        )
    try:
        frame = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        frame = pd.read_csv(path, encoding="latin-1")
    frame = frame.rename(columns=lambda c: c.strip())
    frame["Utterance"] = frame["Utterance"].map(_clean)
    frame["Emotion"] = frame["Emotion"].str.strip().str.lower()
    unknown = set(frame["Emotion"]) - set(MELD_LABELS)
    if unknown:
        raise ValueError(f"unexpected MELD emotions in {split}: {sorted(unknown)}")
    frame["label"] = frame["Emotion"].map(MELD_LABEL_TO_ID)
    frame = frame[frame["Utterance"].str.len() > 0].reset_index(drop=True)
    return frame


def load_meld_dialogues(
    split: str,
    max_turns: int = 24,
    data_dir: Path | None = None,
) -> list[Dialogue]:
    """Group a split into dialogues, preserving order and boundaries."""
    frame = load_meld_frame(split, data_dir=data_dir)
    dialogues: list[Dialogue] = []
    for dialogue_id, group in frame.groupby("Dialogue_ID", sort=True):
        group = group.sort_values("Utterance_ID")
        utterances = group["Utterance"].tolist()
        labels = group["label"].tolist()
        speakers = group["Speaker"].map(_clean).tolist()
        turn_ids = group["Utterance_ID"].tolist()
        for start in range(0, len(utterances), max_turns):
            stop = start + max_turns
            dialogues.append(
                Dialogue(
                    dialogue_id=int(dialogue_id),
                    utterances=utterances[start:stop],
                    labels=labels[start:stop],
                    speakers=speakers[start:stop],
                    utterance_ids=turn_ids[start:stop],
                )
            )
    LOGGER.info(
        "MELD %s: %d dialogues/windows, %d utterances",
        split, len(dialogues), sum(len(d) for d in dialogues),
    )
    return dialogues


class DialogueDataset(Dataset):
    """Padded dialogue batches: input_ids [B, T, L], turn_mask [B, T]."""

    def __init__(self, dialogues: Sequence[Dialogue], tokenizer, max_length: int = 64):
        self.dialogues = list(dialogues)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.dialogues)

    def __getitem__(self, index: int) -> Dialogue:
        return self.dialogues[index]

    def collate(self, batch: Sequence[Dialogue]) -> dict[str, torch.Tensor]:
        batch_size = len(batch)
        max_t = max(len(d) for d in batch)
        flat = [utterance for dialogue in batch for utterance in dialogue.utterances]
        encoded = self.tokenizer(
            flat,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        seq_len = encoded["input_ids"].shape[1]
        pad_id = self.tokenizer.pad_token_id

        input_ids = torch.full((batch_size, max_t, seq_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_t, seq_len), dtype=torch.long)
        labels = torch.full((batch_size, max_t), -100, dtype=torch.long)
        turn_mask = torch.zeros((batch_size, max_t), dtype=torch.bool)
        speaker_ids = torch.zeros((batch_size, max_t), dtype=torch.long)

        cursor = 0
        for b, dialogue in enumerate(batch):
            turns = len(dialogue)
            input_ids[b, :turns] = encoded["input_ids"][cursor : cursor + turns]
            attention_mask[b, :turns] = encoded["attention_mask"][cursor : cursor + turns]
            labels[b, :turns] = torch.tensor(dialogue.labels, dtype=torch.long)
            turn_mask[b, :turns] = True
            speaker_ids[b, :turns] = torch.tensor(
                _relative_speaker_ids(dialogue.speakers), dtype=torch.long
            )
            cursor += turns

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "turn_mask": turn_mask,
            "speaker_ids": speaker_ids,
        }


def _relative_speaker_ids(speakers: Sequence[str], max_speakers: int = 9) -> list[int]:
    """Map speaker names to 1..max_speakers in order of first appearance (0 = pad)."""
    mapping: dict[str, int] = {}
    ids: list[int] = []
    for speaker in speakers:
        if speaker not in mapping:
            mapping[speaker] = min(len(mapping) + 1, max_speakers)
        ids.append(mapping[speaker])
    return ids


def build_meld_dialogue_dataloaders(
    tokenizer,
    batch_size: int = 4,
    eval_batch_size: int | None = None,
    max_length: int = 64,
    max_turns: int = 24,
    num_workers: int = 0,
    limit: int | None = None,
    data_dir: Path | None = None,
) -> dict[str, DataLoader]:
    loaders: dict[str, DataLoader] = {}
    for split in SPLITS:
        dialogues = load_meld_dialogues(split, max_turns=max_turns, data_dir=data_dir)
        if limit:
            dialogues = dialogues[:limit]
        dataset = DialogueDataset(dialogues, tokenizer, max_length=max_length)
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size if split == "train" else (eval_batch_size or batch_size),
            shuffle=split == "train",
            collate_fn=dataset.collate,
            num_workers=num_workers,
        )
    return loaders


def build_meld_utterance_dataloaders(
    tokenizer,
    batch_size: int = 32,
    eval_batch_size: int | None = None,
    max_length: int = 64,
    num_workers: int = 0,
    limit: int | None = None,
    data_dir: Path | None = None,
) -> dict[str, DataLoader]:
    """Independent-utterance view used by systems M1 and M2."""
    loaders: dict[str, DataLoader] = {}
    for split in SPLITS:
        frame = load_meld_frame(split, data_dir=data_dir)
        examples = [
            TextExample(text=row.Utterance, label=int(row.label))
            for row in frame.itertuples()
        ]
        if limit:
            examples = examples[:limit]
        dataset = TextClassificationDataset(examples, tokenizer, max_length=max_length)
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size if split == "train" else (eval_batch_size or batch_size * 2),
            shuffle=split == "train",
            collate_fn=dataset.collate,
            num_workers=num_workers,
        )
    return loaders


def class_weights(split: str = "train", data_dir: Path | None = None) -> torch.Tensor:
    """Inverse-frequency weights over the 7 MELD classes (optional, off by default)."""
    frame = load_meld_frame(split, data_dir=data_dir)
    counts = torch.zeros(len(MELD_LABELS))
    for label, count in frame["label"].value_counts().items():
        counts[int(label)] = float(count)
    weights = counts.sum() / (len(MELD_LABELS) * counts.clamp(min=1.0))
    return weights
