"""Multimodal sign recognition model (vision backbone + temporal model)."""

import torch
import torch.nn as nn
from typing import Optional
from transformers import TimesformerModel, VideoMAEModel


class MultimodalSignRecognition(nn.Module):
    """
    Multimodal model combining vision backbone and keypoint features.
    """

    def __init__(
        self,
        vision_backbone: str = "timesformer",
        num_classes: int = 2000,
        keypoint_dim: int = 1629,
        hidden_dim: int = 768,
        fusion_strategy: str = "cross_attention",
        dropout: float = 0.1,
    ):
        """
        Initialize multimodal model.

        Args:
            vision_backbone: Name of vision backbone ('timesformer', 'videomae', 'i3d').
            num_classes: Number of sign classes.
            keypoint_dim: Dimension of keypoint features.
            hidden_dim: Hidden dimension for fusion.
            fusion_strategy: How to fuse modalities ('late_fusion', 'cross_attention').
            dropout: Dropout probability.
        """
        super().__init__()

        self.vision_backbone_name = vision_backbone
        self.fusion_strategy = fusion_strategy

        # Vision backbone
        if vision_backbone == "timesformer":
            self.vision_backbone = TimesformerModel.from_pretrained(
                "facebook/timesformer-base-finetuned-k400"
            )
            vision_dim = 768
        elif vision_backbone == "videomae":
            self.vision_backbone = VideoMAEModel.from_pretrained(
                "MCG-NJU/videomae-base"
            )
            vision_dim = 768
        else:
            raise ValueError(f"Unknown vision backbone: {vision_backbone}")

        # Keypoint encoder
        self.keypoint_encoder = nn.LSTM(
            input_size=keypoint_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        keypoint_output_dim = hidden_dim * 2

        # Projection layers
        self.vision_projection = nn.Linear(vision_dim, hidden_dim)
        self.keypoint_projection = nn.Linear(keypoint_output_dim, hidden_dim)

        # Fusion module
        if fusion_strategy == "cross_attention":
            from mouna.models.fusion import CrossAttentionFusion
            self.fusion = CrossAttentionFusion(hidden_dim, num_heads=8)
        elif fusion_strategy == "late_fusion":
            from mouna.models.fusion import LateFusion
            self.fusion = LateFusion(hidden_dim * 2, hidden_dim)
        else:
            raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        rgb_frames: torch.Tensor,
        keypoints: torch.Tensor,
        keypoint_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            rgb_frames: RGB video frames (batch, frames, channels, height, width).
            keypoints: Keypoint sequences (batch, seq_len, keypoint_dim).
            keypoint_lengths: Actual sequence lengths.

        Returns:
            Logits (batch, num_classes).
        """
        # Extract vision features
        vision_outputs = self.vision_backbone(rgb_frames)
        vision_features = vision_outputs.last_hidden_state  # (batch, seq_len, dim)
        vision_features = self.vision_projection(vision_features)

        # Extract keypoint features
        if keypoint_lengths is not None:
            keypoints_packed = nn.utils.rnn.pack_padded_sequence(
                keypoints, keypoint_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            keypoint_lstm_out, _ = self.keypoint_encoder(keypoints_packed)
            keypoint_lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                keypoint_lstm_out, batch_first=True
            )
        else:
            keypoint_lstm_out, _ = self.keypoint_encoder(keypoints)

        keypoint_features = self.keypoint_projection(keypoint_lstm_out)

        # Fuse modalities
        fused_features = self.fusion(vision_features, keypoint_features)

        # Classify
        logits = self.classifier(fused_features)

        return logits


class VisionBackbone(nn.Module):
    """Placeholder for custom vision backbone (e.g., I3D)."""

    def __init__(self, backbone_type: str = "i3d", pretrained: bool = True):
        super().__init__()
        # TODO: Implement I3D or other custom backbones
        raise NotImplementedError("Custom vision backbones not yet implemented")
