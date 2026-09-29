"""Stage II: sequential coarse emotion recognition on MELD.

    U_t -> RoBERTa_G -> h_t -> UTE -> z_t -> W7 -> MELD emotion

One class covers all four required systems:

    M1: RoBERTa_base + no UTE     (from_stage1=None, use_ute=False)
    M2: RoBERTa_G    + no UTE     (from_stage1=path, use_ute=False)
    M3: RoBERTa_base + UTE        (from_stage1=None, use_ute=True)
    M4: RoBERTa_G    + UTE        (from_stage1=path, use_ute=True)

theta_G, theta_UTE and W7 are all trainable during Stage II (continued
fine-tuning, not a frozen feature extractor) unless `freeze_backbone` is set,
which exists only for the frozen-vs-trainable ablation named in the spec.

The optional research extension (`use_fine_logits`) keeps the Stage-I 27-class
head as a frozen predictor, concatenates its distribution with the utterance
representation, e_t = [h_t ; p_t^f], and projects back to d_model before the UTE.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel

from fine2coarse.config import BASE_MODEL_ID, NUM_COARSE_LABELS, NUM_FINE_LABELS
from fine2coarse.models.stage1 import backbone_path, head_path
from fine2coarse.models.ute import UtteranceLevelTransformerEncoder


class SequentialEmotionModel(nn.Module):
    def __init__(
        self,
        base_model: str = BASE_MODEL_ID,
        from_stage1: str | Path | None = None,
        num_labels: int = NUM_COARSE_LABELS,
        use_ute: bool = True,
        ute_layers: int = 2,
        ute_heads: int = 8,
        ute_ff: int = 2048,
        ute_dropout: float = 0.1,
        max_turns: int = 64,
        dropout: float = 0.1,
        freeze_backbone: bool = False,
        use_fine_logits: bool = False,
        speaker_aware: bool = False,
        num_speakers: int = 9,
    ) -> None:
        super().__init__()
        if from_stage1 is not None:
            self.backbone = AutoModel.from_pretrained(backbone_path(from_stage1))
            self.backbone_source = str(from_stage1)
        else:
            self.backbone = AutoModel.from_pretrained(base_model)
            self.backbone_source = base_model

        hidden = self.backbone.config.hidden_size
        self.use_ute = use_ute
        self.use_fine_logits = use_fine_logits
        self.dropout = nn.Dropout(dropout)

        self.fine_head: nn.Linear | None = None
        self.fine_projection: nn.Linear | None = None
        if use_fine_logits:
            if from_stage1 is None:
                raise ValueError("use_fine_logits requires a Stage-I checkpoint")
            self.fine_head = nn.Linear(hidden, NUM_FINE_LABELS)
            self.fine_head.load_state_dict(torch.load(head_path(from_stage1), map_location="cpu"))
            for parameter in self.fine_head.parameters():
                parameter.requires_grad = False
            self.fine_projection = nn.Linear(hidden + NUM_FINE_LABELS, hidden)

        self.ute = (
            UtteranceLevelTransformerEncoder(
                d_model=hidden,
                num_layers=ute_layers,
                num_heads=ute_heads,
                dim_feedforward=ute_ff,
                dropout=ute_dropout,
                max_turns=max_turns,
                num_speakers=num_speakers if speaker_aware else 0,
            )
            if use_ute
            else None
        )
        self.classifier = nn.Linear(hidden, num_labels)
        self.num_labels = num_labels

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
        self.freeze_backbone = freeze_backbone

    def encode_utterances(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """[B, T, L] -> [B, T, D] using the shared RoBERTa encoder."""
        batch, turns, length = input_ids.shape
        flat_ids = input_ids.reshape(batch * turns, length)
        flat_mask = attention_mask.reshape(batch * turns, length)
        output = self.backbone(input_ids=flat_ids, attention_mask=flat_mask)
        pooled = output.last_hidden_state[:, 0]
        return pooled.reshape(batch, turns, -1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        turn_mask: torch.Tensor | None = None,
        speaker_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns logits [B, T, num_labels]. 2-D inputs are treated as T=1."""
        squeezed = input_ids.dim() == 2
        if squeezed:
            input_ids = input_ids.unsqueeze(1)
            attention_mask = attention_mask.unsqueeze(1)
        if turn_mask is None:
            turn_mask = torch.ones(input_ids.shape[:2], dtype=torch.bool, device=input_ids.device)

        hidden = self.encode_utterances(input_ids, attention_mask)

        if self.use_fine_logits:
            fine_probabilities = torch.softmax(self.fine_head(hidden), dim=-1)
            hidden = self.fine_projection(torch.cat([hidden, fine_probabilities], dim=-1))

        if self.ute is not None:
            hidden = self.ute(hidden, turn_mask=turn_mask, speaker_ids=speaker_ids)

        logits = self.classifier(self.dropout(hidden))
        return logits.squeeze(1) if squeezed else logits

    def parameter_groups(self, backbone_lr: float, head_lr: float, weight_decay: float) -> list[dict]:
        """Lower learning rate for the pretrained backbone, higher for new modules."""
        backbone_parameters = [p for p in self.backbone.parameters() if p.requires_grad]
        new_modules = [self.classifier]
        if self.ute is not None:
            new_modules.append(self.ute)
        if self.fine_projection is not None:
            new_modules.append(self.fine_projection)
        new_parameters = [p for module in new_modules for p in module.parameters() if p.requires_grad]
        groups = [{"params": new_parameters, "lr": head_lr, "weight_decay": weight_decay}]
        if backbone_parameters:
            groups.insert(0, {"params": backbone_parameters, "lr": backbone_lr, "weight_decay": weight_decay})
        return groups
