"""Label vocabularies, paths, and small config helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("F2C_DATA_DIR", PROJECT_ROOT / "data"))
RESULTS_DIR = Path(os.environ.get("F2C_RESULTS_DIR", PROJECT_ROOT / "results"))
CHECKPOINT_DIR = Path(os.environ.get("F2C_CHECKPOINT_DIR", PROJECT_ROOT / "checkpoints"))

BASE_MODEL_ID = os.environ.get("F2C_BASE_MODEL", "roberta-base")

# GoEmotions "simplified" label order as published on the Hub. Index 27 is
# `neutral`, which the project spec excludes -> Stage I uses indices 0..26.
GOEMOTIONS_ALL_LABELS: tuple[str, ...] = (
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization", "relief",
    "remorse", "sadness", "surprise", "neutral",
)
NEUTRAL_INDEX = GOEMOTIONS_ALL_LABELS.index("neutral")
GOEMOTIONS_LABELS: tuple[str, ...] = tuple(
    label for i, label in enumerate(GOEMOTIONS_ALL_LABELS) if i != NEUTRAL_INDEX
)
NUM_FINE_LABELS = len(GOEMOTIONS_LABELS)  # 27

# MELD coarse emotions, alphabetical; the CSV strings are lowercase already.
MELD_LABELS: tuple[str, ...] = (
    "anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise",
)
NUM_COARSE_LABELS = len(MELD_LABELS)  # 7
MELD_LABEL_TO_ID = {label: i for i, label in enumerate(MELD_LABELS)}

MELD_CSV_URLS = {
    "train": "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/train_sent_emo.csv",
    "dev": "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/dev_sent_emo.csv",
    "test": "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/test_sent_emo.csv",
}
MELD_CSV_NAMES = {
    "train": "train_sent_emo.csv",
    "dev": "dev_sent_emo.csv",
    "test": "test_sent_emo.csv",
}


@dataclass
class RunPaths:
    """Where one training run writes its checkpoint, metrics, and predictions."""

    name: str

    @property
    def checkpoint(self) -> Path:
        return CHECKPOINT_DIR / self.name

    @property
    def results(self) -> Path:
        return RESULTS_DIR / self.name

    def make(self) -> "RunPaths":
        self.checkpoint.mkdir(parents=True, exist_ok=True)
        self.results.mkdir(parents=True, exist_ok=True)
        return self


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def merge_overrides(config: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply `key=value` CLI overrides onto a config dict (values parsed as YAML)."""
    merged = dict(config)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like key=value, got {item!r}")
        key, raw = item.split("=", 1)
        merged[key.strip()] = yaml.safe_load(raw)
    return merged
