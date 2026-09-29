"""Stage I: RoBERTa-base + 27-way fine-grained head.

    x --RoBERTa--> h --W27--> p(y^f | x)

After Stage I the head W27 is discarded and only the backbone parameters
theta_G transfer to Stage II. `save_pretrained_parts` writes the backbone in
HuggingFace format (so Stage II can load it by path) and the head separately,
which the optional research extension re-loads.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, AutoModel

from fine2coarse.config import BASE_MODEL_ID, NUM_FINE_LABELS

BACKBONE_SUBDIR = "backbone"
HEAD_FILENAME = "fine_head.pt"


class FineGrainedClassifier(nn.Module):
    def __init__(
        self,
        model_name: str = BASE_MODEL_ID,
        num_labels: int = NUM_FINE_LABELS,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, num_labels)
        self.num_labels = num_labels

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Pooled utterance representation h: the <s> (CLS) token state."""
        output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return output.last_hidden_state[:, 0]

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.head(self.dropout(self.encode(input_ids, attention_mask)))

    def save_pretrained_parts(self, directory: str | Path) -> Path:
        directory = Path(directory)
        (directory / BACKBONE_SUBDIR).mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(directory / BACKBONE_SUBDIR)
        torch.save(self.head.state_dict(), directory / HEAD_FILENAME)
        return directory

    @classmethod
    def from_saved(cls, directory: str | Path, num_labels: int = NUM_FINE_LABELS) -> "FineGrainedClassifier":
        directory = Path(directory)
        config = AutoConfig.from_pretrained(directory / BACKBONE_SUBDIR)
        model = cls.__new__(cls)
        nn.Module.__init__(model)
        model.backbone = AutoModel.from_pretrained(directory / BACKBONE_SUBDIR, config=config)
        model.dropout = nn.Dropout(0.1)
        model.head = nn.Linear(config.hidden_size, num_labels)
        model.head.load_state_dict(torch.load(directory / HEAD_FILENAME, map_location="cpu"))
        model.num_labels = num_labels
        return model


def backbone_path(stage1_dir: str | Path) -> Path:
    return Path(stage1_dir) / BACKBONE_SUBDIR


def head_path(stage1_dir: str | Path) -> Path:
    return Path(stage1_dir) / HEAD_FILENAME
