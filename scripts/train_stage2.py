#!/usr/bin/env python3
"""Stage II: train one of the four MELD systems (M1-M4)."""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fine2coarse.config import load_yaml, merge_overrides  # noqa: E402
from fine2coarse.engine import Stage2Config, train_stage2  # noqa: E402

CONFIG_FOR_SYSTEM = {
    "M1": "configs/m1_base_noute.yaml",
    "M2": "configs/m2_goemotions_noute.yaml",
    "M3": "configs/m3_base_ute.yaml",
    "M4": "configs/m4_goemotions_ute.yaml",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=sorted(CONFIG_FOR_SYSTEM), help="shortcut for --config")
    parser.add_argument("--config", help="path to a Stage-II YAML config")
    parser.add_argument("--set", nargs="*", dest="overrides", help="key=value overrides")
    args = parser.parse_args()

    if not args.config and not args.system:
        parser.error("pass --system M1|M2|M3|M4 or --config PATH")
    config_path = args.config or CONFIG_FOR_SYSTEM[args.system]

    raw = merge_overrides(load_yaml(config_path), args.overrides)
    known = {f.name for f in fields(Stage2Config)}
    unknown = set(raw) - known
    if unknown:
        raise SystemExit(f"unknown config keys: {sorted(unknown)}")
    train_stage2(Stage2Config(**raw))


if __name__ == "__main__":
    main()
