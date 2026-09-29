#!/usr/bin/env python3
"""Stage I: fine-tune RoBERTa-base on GoEmotions (27 classes, neutral excluded)."""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fine2coarse.config import load_yaml, merge_overrides  # noqa: E402
from fine2coarse.engine import Stage1Config, train_stage1  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage1.yaml")
    parser.add_argument("--set", nargs="*", dest="overrides", help="key=value overrides")
    args = parser.parse_args()

    raw = merge_overrides(load_yaml(args.config), args.overrides)
    known = {f.name for f in fields(Stage1Config)}
    unknown = set(raw) - known
    if unknown:
        raise SystemExit(f"unknown config keys: {sorted(unknown)}")
    train_stage1(Stage1Config(**raw))


if __name__ == "__main__":
    main()
