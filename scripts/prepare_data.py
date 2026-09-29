#!/usr/bin/env python3
"""Download and sanity-check the datasets (GoEmotions cache + MELD CSVs)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fine2coarse.config import GOEMOTIONS_LABELS, MELD_LABELS  # noqa: E402
from fine2coarse.data.goemotions import load_goemotions  # noqa: E402
from fine2coarse.data.meld import download_meld, load_meld_dialogues, load_meld_frame  # noqa: E402
from fine2coarse.utils import get_logger  # noqa: E402

LOGGER = get_logger()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["all", "goemotions", "meld"], default="all")
    parser.add_argument("--force", action="store_true", help="re-download MELD CSVs")
    parser.add_argument("--multilabel-policy", choices=["single", "first"], default="single")
    args = parser.parse_args()

    if args.dataset in ("all", "goemotions"):
        splits = load_goemotions(multilabel_policy=args.multilabel_policy)
        LOGGER.info("GoEmotions labels (%d): %s", len(GOEMOTIONS_LABELS), ", ".join(GOEMOTIONS_LABELS))
        for split, examples in splits.items():
            LOGGER.info("  %s: %d examples", split, len(examples))

    if args.dataset in ("all", "meld"):
        download_meld(force=args.force)
        for split in ("train", "dev", "test"):
            frame = load_meld_frame(split)
            dialogues = load_meld_dialogues(split)
            counts = frame["Emotion"].value_counts().to_dict()
            LOGGER.info(
                "MELD %s: %d utterances, %d dialogue windows, counts=%s",
                split, len(frame), len(dialogues), {k: counts.get(k, 0) for k in MELD_LABELS},
            )


if __name__ == "__main__":
    main()
