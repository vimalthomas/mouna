"""Fusion strategies for multimodal models."""

import torch
import torch.nn as nn


class LateFusion(nn.Module):
    """Late fusion: concatenate features and project."""

    def __init__(self, input_dim: int, output_dim: int):
        """
        Initialize late fusion.

        Args:
            input_dim: Combined input dimension (vision_dim + keypoint_dim).
            output_dim: Output dimension.
        """
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(
        self, vision_features: torch.Tensor, keypoint_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse features by concatenation.

        Args:
            vision_features: (batch, seq_len, dim)
            keypoint_features: (batch, seq_len, dim)

        Returns:
            Fused features (batch, output_dim)
        """
        # Pool temporal dimension
        vision_pooled = vision_features.mean(dim=1)  # (batch, dim)
        keypoint_pooled = keypoint_features.mean(dim=1)  # (batch, dim)

        # Concatenate
        concatenated = torch.cat([vision_pooled, keypoint_pooled], dim=1)

        # Project
        fused = self.projection(concatenated)

        return fused


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion between modalities."""

    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        """
        Initialize cross-attention fusion.

        Args:
            hidden_dim: Feature dimension.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
        """
        super().__init__()

        self.cross_attention_v2k = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.cross_attention_k2v = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(
        self, vision_features: torch.Tensor, keypoint_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply cross-attention between modalities.

        Args:
            vision_features: (batch, seq_len, dim)
            keypoint_features: (batch, seq_len, dim)

        Returns:
            Fused features (batch, dim)
        """
        # Vision attending to keypoints
        v_attend_k, _ = self.cross_attention_v2k(
            query=vision_features,
            key=keypoint_features,
            value=keypoint_features,
        )
        vision_enhanced = self.norm1(vision_features + v_attend_k)

        # Keypoints attending to vision
        k_attend_v, _ = self.cross_attention_k2v(
            query=keypoint_features,
            key=vision_features,
            value=vision_features,
        )
        keypoint_enhanced = self.norm2(keypoint_features + k_attend_v)

        # Combine both enhanced representations
        combined = (vision_enhanced + keypoint_enhanced) / 2

        # Feed-forward
        fused = self.norm3(combined + self.ffn(combined))

        # Temporal pooling
        fused_pooled = fused.mean(dim=1)  # (batch, dim)

        return fused_pooled


class EarlyFusion(nn.Module):
    """Early fusion: combine at feature level."""

    def __init__(self, vision_dim: int, keypoint_dim: int, output_dim: int):
        """
        Initialize early fusion.

        Args:
            vision_dim: Vision feature dimension.
            keypoint_dim: Keypoint feature dimension.
            output_dim: Output dimension.
        """
        super().__init__()

        self.vision_projection = nn.Linear(vision_dim, output_dim)
        self.keypoint_projection = nn.Linear(keypoint_dim, output_dim)

        self.fusion = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(
        self, vision_features: torch.Tensor, keypoint_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse features at early stage.

        Args:
            vision_features: (batch, seq_len, vision_dim)
            keypoint_features: (batch, seq_len, keypoint_dim)

        Returns:
            Fused features (batch, output_dim)
        """
        # Project to same dimension
        vision_proj = self.vision_projection(vision_features)
        keypoint_proj = self.keypoint_projection(keypoint_features)

        # Element-wise addition
        combined = vision_proj + keypoint_proj

        # Fusion layer
        fused = self.fusion(combined)

        # Temporal pooling
        fused_pooled = fused.mean(dim=1)

        return fused_pooled
