"""Training and evaluation loops for Stage I and Stage II."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from fine2coarse.config import (
    BASE_MODEL_ID,
    GOEMOTIONS_LABELS,
    MELD_LABELS,
    NUM_COARSE_LABELS,
    NUM_FINE_LABELS,
    RunPaths,
)
from fine2coarse.data.goemotions import build_goemotions_dataloaders
from fine2coarse.data.meld import (
    build_meld_dialogue_dataloaders,
    build_meld_utterance_dataloaders,
    class_weights,
)
from fine2coarse.metrics import compute_metrics, format_metrics
from fine2coarse.models.stage1 import FineGrainedClassifier
from fine2coarse.models.stage2 import SequentialEmotionModel
from fine2coarse.utils import count_trainable, get_logger, resolve_device, set_seed, write_json

LOGGER = get_logger()
IGNORE_INDEX = -100


@dataclass
class Stage1Config:
    run_name: str = "stage1_goemotions"
    model_name: str = BASE_MODEL_ID
    seed: int = 42
    epochs: int = 4
    batch_size: int = 32
    eval_batch_size: int = 64
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    max_length: int = 64
    dropout: float = 0.1
    multilabel_policy: str = "single"
    grad_clip: float = 1.0
    device: str = "auto"
    num_workers: int = 0
    limit: int | None = None
    amp: bool = True


@dataclass
class Stage2Config:
    run_name: str = "M4_goemotions_ute"
    system: str = "M4"
    base_model: str = BASE_MODEL_ID
    from_stage1: str | None = None
    use_ute: bool = True
    view: str = "dialogue"  # "dialogue" for M3/M4, "utterance" for M1/M2
    seed: int = 42
    epochs: int = 5
    batch_size: int = 4
    eval_batch_size: int = 8
    learning_rate: float = 1e-5       # backbone
    head_learning_rate: float = 1e-4  # UTE + classifier
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_length: int = 64
    max_turns: int = 24
    dropout: float = 0.1
    ute_layers: int = 2
    ute_heads: int = 8
    ute_ff: int = 2048
    ute_dropout: float = 0.1
    freeze_backbone: bool = False
    use_fine_logits: bool = False
    speaker_aware: bool = False
    class_weighted_loss: bool = False
    grad_clip: float = 1.0
    device: str = "auto"
    num_workers: int = 0
    limit: int | None = None
    amp: bool = True
    extras: dict[str, Any] = field(default_factory=dict)


def _autocast(device: torch.device, enabled: bool):
    if device.type == "cuda" and enabled:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return torch.autocast(device_type=device.type, enabled=False)


def _optimizer_and_schedule(parameters, learning_rate, weight_decay, steps, warmup_ratio):
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(steps * warmup_ratio), num_training_steps=steps
    )
    return optimizer, scheduler


# --------------------------------------------------------------------------- Stage I


@torch.no_grad()
def predict_stage1(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[list[int], list[int]]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    for batch in tqdm(loader, desc="eval", leave=False):
        labels = batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["input_ids"], batch["attention_mask"])
        y_pred.extend(logits.argmax(dim=-1).cpu().tolist())
        y_true.extend(labels.tolist())
    return y_true, y_pred


def train_stage1(config: Stage1Config) -> dict:
    set_seed(config.seed)
    device = resolve_device(config.device)
    paths = RunPaths(config.run_name).make()
    LOGGER.info("Stage I on %s -> %s", device, paths.checkpoint)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    loaders = build_goemotions_dataloaders(
        tokenizer,
        batch_size=config.batch_size,
        eval_batch_size=config.eval_batch_size,
        max_length=config.max_length,
        multilabel_policy=config.multilabel_policy,
        num_workers=config.num_workers,
        limit=config.limit,
    )
    model = FineGrainedClassifier(
        config.model_name, num_labels=NUM_FINE_LABELS, dropout=config.dropout
    ).to(device)
    LOGGER.info("trainable parameters: %d", count_trainable(model))

    steps = max(1, len(loaders["train"]) * config.epochs)
    optimizer, scheduler = _optimizer_and_schedule(
        model.parameters(), config.learning_rate, config.weight_decay, steps, config.warmup_ratio
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")
    criterion = nn.CrossEntropyLoss()

    best_macro_f1 = -1.0
    history: list[dict] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        running = 0.0
        started = time.time()
        for batch in tqdm(loaders["train"], desc=f"stage1 epoch {epoch}"):
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, config.amp):
                logits = model(batch["input_ids"], batch["attention_mask"])
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item()

        y_true, y_pred = predict_stage1(model, loaders["validation"], device)
        metrics = compute_metrics(y_true, y_pred, GOEMOTIONS_LABELS)
        entry = {
            "epoch": epoch,
            "train_loss": running / max(1, len(loaders["train"])),
            "val": {k: metrics[k] for k in ("accuracy", "macro_f1", "weighted_f1")},
            "seconds": time.time() - started,
        }
        history.append(entry)
        LOGGER.info("epoch %d loss=%.4f val %s", epoch, entry["train_loss"], format_metrics(metrics))

        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            model.save_pretrained_parts(paths.checkpoint)
            tokenizer.save_pretrained(paths.checkpoint)
            LOGGER.info("new best val macro-F1 %.4f -> checkpoint saved", best_macro_f1)

    best = FineGrainedClassifier.from_saved(paths.checkpoint).to(device)
    y_true, y_pred = predict_stage1(best, loaders["test"], device)
    test_metrics = compute_metrics(y_true, y_pred, GOEMOTIONS_LABELS)
    LOGGER.info("Stage I test %s", format_metrics(test_metrics))

    payload = {
        "stage": "I",
        "config": asdict(config),
        "history": history,
        "best_val_macro_f1": best_macro_f1,
        "test": test_metrics,
        "checkpoint": str(paths.checkpoint),
    }
    write_json(paths.results / "metrics.json", payload)
    write_json(paths.results / "test_predictions.json", {"y_true": y_true, "y_pred": y_pred})
    return payload


# -------------------------------------------------------------------------- Stage II


def _stage2_loaders(tokenizer, config: Stage2Config) -> dict[str, DataLoader]:
    if config.view == "dialogue":
        return build_meld_dialogue_dataloaders(
            tokenizer,
            batch_size=config.batch_size,
            eval_batch_size=config.eval_batch_size,
            max_length=config.max_length,
            max_turns=config.max_turns,
            num_workers=config.num_workers,
            limit=config.limit,
        )
    if config.view == "utterance":
        return build_meld_utterance_dataloaders(
            tokenizer,
            batch_size=config.batch_size,
            eval_batch_size=config.eval_batch_size,
            max_length=config.max_length,
            num_workers=config.num_workers,
            limit=config.limit,
        )
    raise ValueError(f"view must be 'dialogue' or 'utterance', got {config.view!r}")


def _stage2_forward(model: SequentialEmotionModel, batch: dict, device: torch.device):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    turn_mask = batch.get("turn_mask")
    speaker_ids = batch.get("speaker_ids")
    if turn_mask is not None:
        turn_mask = turn_mask.to(device)
    if speaker_ids is not None:
        speaker_ids = speaker_ids.to(device)
    logits = model(input_ids, attention_mask, turn_mask=turn_mask, speaker_ids=speaker_ids)
    return logits.reshape(-1, model.num_labels), labels.reshape(-1)


@torch.no_grad()
def predict_stage2(model: SequentialEmotionModel, loader: DataLoader, device: torch.device):
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    for batch in tqdm(loader, desc="eval", leave=False):
        logits, labels = _stage2_forward(model, batch, device)
        keep = labels != IGNORE_INDEX
        y_pred.extend(logits[keep].argmax(dim=-1).cpu().tolist())
        y_true.extend(labels[keep].cpu().tolist())
    return y_true, y_pred


def build_stage2_model(config: Stage2Config) -> SequentialEmotionModel:
    return SequentialEmotionModel(
        base_model=config.base_model,
        from_stage1=config.from_stage1,
        num_labels=NUM_COARSE_LABELS,
        use_ute=config.use_ute,
        ute_layers=config.ute_layers,
        ute_heads=config.ute_heads,
        ute_ff=config.ute_ff,
        ute_dropout=config.ute_dropout,
        max_turns=max(config.max_turns, 64),
        dropout=config.dropout,
        freeze_backbone=config.freeze_backbone,
        use_fine_logits=config.use_fine_logits,
        speaker_aware=config.speaker_aware,
    )


def train_stage2(config: Stage2Config) -> dict:
    set_seed(config.seed)
    device = resolve_device(config.device)
    paths = RunPaths(config.run_name).make()
    LOGGER.info("Stage II system %s on %s -> %s", config.system, device, paths.checkpoint)

    tokenizer_source = config.from_stage1 or config.base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    loaders = _stage2_loaders(tokenizer, config)

    model = build_stage2_model(config).to(device)
    LOGGER.info(
        "backbone=%s use_ute=%s trainable=%d",
        model.backbone_source, config.use_ute, count_trainable(model),
    )

    steps = max(1, len(loaders["train"]) * config.epochs)
    optimizer, scheduler = _optimizer_and_schedule(
        model.parameter_groups(config.learning_rate, config.head_learning_rate, config.weight_decay),
        config.learning_rate,
        config.weight_decay,
        steps,
        config.warmup_ratio,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")
    weights = class_weights().to(device) if config.class_weighted_loss else None
    criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=IGNORE_INDEX)

    best_macro_f1 = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        running = 0.0
        started = time.time()
        for batch in tqdm(loaders["train"], desc=f"{config.system} epoch {epoch}"):
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, config.amp):
                logits, labels = _stage2_forward(model, batch, device)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item()

        y_true, y_pred = predict_stage2(model, loaders["dev"], device)
        metrics = compute_metrics(y_true, y_pred, MELD_LABELS)
        entry = {
            "epoch": epoch,
            "train_loss": running / max(1, len(loaders["train"])),
            "dev": {k: metrics[k] for k in ("accuracy", "macro_f1", "weighted_f1")},
            "seconds": time.time() - started,
        }
        history.append(entry)
        LOGGER.info("epoch %d loss=%.4f dev %s", epoch, entry["train_loss"], format_metrics(metrics))

        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, paths.checkpoint / "model.pt")
            tokenizer.save_pretrained(paths.checkpoint)
            LOGGER.info("new best dev macro-F1 %.4f -> checkpoint saved", best_macro_f1)

    if best_state is not None:
        model.load_state_dict(best_state)
    y_true, y_pred = predict_stage2(model, loaders["test"], device)
    test_metrics = compute_metrics(y_true, y_pred, MELD_LABELS)
    LOGGER.info("%s test %s", config.system, format_metrics(test_metrics))

    payload = {
        "stage": "II",
        "system": config.system,
        "config": asdict(config),
        "goemotions_transfer": config.from_stage1 is not None,
        "ute": config.use_ute,
        "history": history,
        "best_dev_macro_f1": best_macro_f1,
        "test": test_metrics,
        "checkpoint": str(paths.checkpoint),
    }
    write_json(paths.results / "metrics.json", payload)
    write_json(paths.results / "test_predictions.json", {"y_true": y_true, "y_pred": y_pred})
    return payload
