"""Stage I data: GoEmotions, 27 fine-grained classes, neutral excluded.

GoEmotions is multi-label. The project spec trains a single-label 27-way
classifier with cross-entropy, so examples are reduced to one label:

* ``multilabel_policy="single"`` (default) keeps only examples that carry
  exactly one non-neutral label -- the cleanest single-label reading.
* ``multilabel_policy="first"`` keeps every example with at least one
  non-neutral label and takes the lowest-index label.

Examples whose only label is ``neutral`` are dropped either way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from fine2coarse.config import GOEMOTIONS_LABELS, NEUTRAL_INDEX
from fine2coarse.utils import get_logger

LOGGER = get_logger()


def remap_labels(label_ids: Iterable[int]) -> list[int]:
    """Drop neutral and shift the remaining GoEmotions ids into 0..26."""
    return [i if i < NEUTRAL_INDEX else i - 1 for i in label_ids if i != NEUTRAL_INDEX]


@dataclass
class TextExample:
    text: str
    label: int


class TextClassificationDataset(Dataset):
    def __init__(self, examples: Sequence[TextExample], tokenizer, max_length: int = 64):
        self.examples = list(examples)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> TextExample:
        return self.examples[index]

    def collate(self, batch: Sequence[TextExample]) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            [example.text for example in batch],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor([example.label for example in batch], dtype=torch.long)
        return encoded


def load_goemotions(
    multilabel_policy: str = "single",
    cache_dir: str | None = None,
) -> dict[str, list[TextExample]]:
    """Load the `go_emotions` simplified config and reduce it to 27 classes."""
    from datasets import load_dataset

    # Namespaced id: newer huggingface_hub rejects the bare legacy name "go_emotions".
    raw = load_dataset("google-research-datasets/go_emotions", "simplified", cache_dir=cache_dir)
    splits: dict[str, list[TextExample]] = {}
    for split in ("train", "validation", "test"):
        examples: list[TextExample] = []
        dropped_neutral = 0
        dropped_multi = 0
        for row in raw[split]:
            labels = remap_labels(row["labels"])
            if not labels:
                dropped_neutral += 1
                continue
            if len(labels) > 1:
                if multilabel_policy == "single":
                    dropped_multi += 1
                    continue
                if multilabel_policy != "first":
                    raise ValueError(f"unknown multilabel_policy {multilabel_policy!r}")
            examples.append(TextExample(text=row["text"], label=labels[0]))
        LOGGER.info(
            "GoEmotions %s: kept %d, dropped %d neutral-only, dropped %d multi-label",
            split, len(examples), dropped_neutral, dropped_multi,
        )
        splits[split] = examples
    return splits


def build_goemotions_dataloaders(
    tokenizer,
    batch_size: int = 32,
    eval_batch_size: int | None = None,
    max_length: int = 64,
    multilabel_policy: str = "single",
    num_workers: int = 0,
    limit: int | None = None,
    cache_dir: str | None = None,
) -> dict[str, DataLoader]:
    splits = load_goemotions(multilabel_policy=multilabel_policy, cache_dir=cache_dir)
    loaders: dict[str, DataLoader] = {}
    for split, examples in splits.items():
        if limit:
            examples = examples[:limit]
        dataset = TextClassificationDataset(examples, tokenizer, max_length=max_length)
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size if split == "train" else (eval_batch_size or batch_size * 2),
            shuffle=split == "train",
            collate_fn=dataset.collate,
            num_workers=num_workers,
            drop_last=False,
        )
    return loaders


def label_names() -> tuple[str, ...]:
    return GOEMOTIONS_LABELS
