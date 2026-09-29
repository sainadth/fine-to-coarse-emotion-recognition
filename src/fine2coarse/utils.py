"""Seeding, device selection, logging, and JSON helpers."""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

LOGGER_NAME = "fine2coarse"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def resolve_device(requested: str = "auto") -> torch.device:
    """cuda > mps > cpu, unless the caller pins a device explicitly."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_logger(name: str = LOGGER_NAME) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_jsonable)
        handle.write("\n")


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def count_trainable(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
