"""Transformer encoder model for keypoint-sequence sign language classification."""

import math

import torch
import torch.nn as nn
from typing import Optional


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 200, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class PoseTransformer(nn.Module):
    """
    Transformer encoder for pose/keypoint sequence classification.

    Architecture:
      input_proj → sinusoidal pos-enc → TransformerEncoder → masked mean pool → classifier
    """

    def __init__(
        self,
        input_dim: int = 225,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.3,
        num_classes: int = 100,
    ):
        super().__init__()
        self.d_model = d_model

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.pos_enc = SinusoidalPositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-LN: more stable training
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(
        self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
            lengths: (batch,) actual frame counts before padding

        Returns:
            logits: (batch, num_classes)
        """
        x = self.input_proj(x)       # (B, T, d_model)
        x = self.pos_enc(x)

        # Build padding mask: True = position to IGNORE (padding)
        if lengths is not None:
            B, T, _ = x.shape
            mask = torch.arange(T, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
        else:
            mask = None

        x = self.encoder(x, src_key_padding_mask=mask)  # (B, T, d_model)

        # Masked mean pooling over valid frames
        if lengths is not None:
            valid = (~mask).unsqueeze(-1).float()          # (B, T, 1)
            pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        else:
            pooled = x.mean(dim=1)                         # (B, d_model)

        return self.classifier(pooled)
