"""Baseline models using keypoints + RNN (BiLSTM/GRU)."""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class KeypointBiLSTM(nn.Module):
    """BiLSTM model for keypoint sequence classification."""

    def __init__(
        self,
        input_dim: int = 1629,  # Flattened MediaPipe keypoints
        hidden_dim: int = 512,
        num_layers: int = 2,
        num_classes: int = 2000,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        """
        Initialize BiLSTM model.

        Args:
            input_dim: Input feature dimension (flattened keypoints).
            hidden_dim: Hidden dimension of LSTM.
            num_layers: Number of LSTM layers.
            num_classes: Number of sign classes.
            dropout: Dropout probability.
            bidirectional: Whether to use bidirectional LSTM.
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional

        # Input projection (optional, to reduce dimensionality)
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # BiLSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # Output dimension
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # Attention mechanism (optional)
        self.attention = TemporalAttention(lstm_output_dim)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input keypoints of shape (batch, seq_len, input_dim).
            lengths: Actual sequence lengths before padding (batch,).

        Returns:
            Logits of shape (batch, num_classes).
        """
        batch_size = x.size(0)

        # Project input
        x = self.input_projection(x)  # (batch, seq_len, hidden_dim)

        # Pack padded sequence if lengths provided
        if lengths is not None:
            lengths_cpu = lengths.cpu()
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths_cpu, batch_first=True, enforce_sorted=False
            )

        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)

        # Unpack if packed
        if lengths is not None:
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        # Apply attention
        attended = self.attention(lstm_out, lengths)  # (batch, lstm_output_dim)

        # Classify
        logits = self.classifier(attended)  # (batch, num_classes)

        return logits


class KeypointGRU(nn.Module):
    """GRU model for keypoint sequence classification."""

    def __init__(
        self,
        input_dim: int = 1629,
        hidden_dim: int = 512,
        num_layers: int = 2,
        num_classes: int = 2000,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        """
        Initialize GRU model.

        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden dimension of GRU.
            num_layers: Number of GRU layers.
            num_classes: Number of sign classes.
            dropout: Dropout probability.
            bidirectional: Whether to use bidirectional GRU.
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional

        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # GRU layers
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # Output dimension
        gru_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # Attention
        self.attention = TemporalAttention(gru_output_dim)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gru_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input keypoints of shape (batch, seq_len, input_dim).
            lengths: Actual sequence lengths (batch,).

        Returns:
            Logits of shape (batch, num_classes).
        """
        # Project input
        x = self.input_projection(x)

        # Pack if lengths provided
        if lengths is not None:
            lengths_cpu = lengths.cpu()
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths_cpu, batch_first=True, enforce_sorted=False
            )

        # GRU
        gru_out, hidden = self.gru(x)

        # Unpack
        if lengths is not None:
            gru_out, _ = nn.utils.rnn.pad_packed_sequence(gru_out, batch_first=True)

        # Attention
        attended = self.attention(gru_out, lengths)

        # Classify
        logits = self.classifier(attended)

        return logits


class TemporalAttention(nn.Module):
    """Temporal attention mechanism for sequence pooling."""

    def __init__(self, hidden_dim: int):
        """
        Initialize attention layer.

        Args:
            hidden_dim: Dimension of hidden states.
        """
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(
        self, sequence: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply temporal attention.

        Args:
            sequence: (batch, seq_len, hidden_dim)
            lengths: Actual sequence lengths (batch,)

        Returns:
            Attended representation (batch, hidden_dim)
        """
        # Compute attention scores
        attention_scores = self.attention(sequence)  # (batch, seq_len, 1)
        attention_scores = attention_scores.squeeze(-1)  # (batch, seq_len)

        # Mask padding positions if lengths provided
        if lengths is not None:
            max_len = sequence.size(1)
            mask = torch.arange(max_len, device=sequence.device).expand(
                len(lengths), max_len
            ) < lengths.unsqueeze(1)
            attention_scores = attention_scores.masked_fill(~mask, float("-inf"))

        # Apply softmax
        attention_weights = torch.softmax(attention_scores, dim=1)  # (batch, seq_len)

        # Weighted sum
        attended = torch.bmm(
            attention_weights.unsqueeze(1), sequence
        ).squeeze(1)  # (batch, hidden_dim)

        return attended


def create_baseline_model(
    model_type: str = "bilstm",
    input_dim: int = 1629,
    hidden_dim: int = 512,
    num_layers: int = 2,
    num_classes: int = 2000,
    dropout: float = 0.3,
    bidirectional: bool = True,
) -> nn.Module:
    """
    Factory function to create baseline model.

    Args:
        model_type: Type of model ('bilstm' or 'gru').
        input_dim: Input feature dimension.
        hidden_dim: Hidden dimension.
        num_layers: Number of RNN layers.
        num_classes: Number of output classes.
        dropout: Dropout probability.
        bidirectional: Whether to use bidirectional RNN.

    Returns:
        Baseline model instance.
    """
    if model_type.lower() == "bilstm" or model_type.lower() == "lstm":
        return KeypointBiLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
            bidirectional=bidirectional,
        )
    elif model_type.lower() == "gru":
        return KeypointGRU(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
            bidirectional=bidirectional,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
