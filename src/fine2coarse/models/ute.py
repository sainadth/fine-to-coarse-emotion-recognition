"""Utterance-Level Transformer Encoder (UTE).

Turn-level positional embeddings are added to the utterance representations,

    h~_t = h_t + p_t,

and a standard Transformer encoder contextualises the dialogue:

    (z_1, ..., z_T) = UTE(h~_1, ..., h~_T).
"""

from __future__ import annotations

import torch
from torch import nn


class UtteranceLevelTransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        num_layers: int = 2,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_turns: int = 64,
        num_speakers: int = 0,
    ) -> None:
        super().__init__()
        self.position = nn.Embedding(max_turns, d_model)
        self.speaker = nn.Embedding(num_speakers + 1, d_model) if num_speakers else None
        self.norm_in = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.max_turns = max_turns
        nn.init.normal_(self.position.weight, std=0.02)
        if self.speaker is not None:
            # Same scale as the positional embedding: both are added straight onto the
            # RoBERTa hidden states, and nn.Embedding's N(0, 1) default would swamp them.
            nn.init.normal_(self.speaker.weight, std=0.02)

    def forward(
        self,
        hidden: torch.Tensor,
        turn_mask: torch.Tensor,
        speaker_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """hidden [B, T, D], turn_mask [B, T] (True = real turn) -> z [B, T, D]."""
        batch, turns, _ = hidden.shape
        if turns > self.max_turns:
            raise ValueError(f"dialogue has {turns} turns but max_turns={self.max_turns}")
        positions = torch.arange(turns, device=hidden.device).unsqueeze(0).expand(batch, turns)
        hidden = hidden + self.position(positions)
        if self.speaker is not None and speaker_ids is not None:
            hidden = hidden + self.speaker(speaker_ids.clamp(max=self.speaker.num_embeddings - 1))
        hidden = self.dropout(self.norm_in(hidden))
        return self.encoder(hidden, src_key_padding_mask=~turn_mask)
