"""Self-supervised masked-reconstruction pretraining for PoseTransformer.

Simplified SignBERT-style approach: mask contiguous spans of frames, train
the encoder to reconstruct them from context. Deliberately skips SignBERT+'s
hand-model-aware (MANO) machinery — our OpenPose-converted data is 2D with
z=0 and confidence-floored, so there's no real 3D hand geometry to fit a hand
model to. A plain masked autoencoder is the standard first step the fancier
methods build on, not a corner cut.

Masked positions get a learnable mask token substituted at the embedding
level (after input_proj, before pos_enc) rather than zeroing raw input —
zero is already the real encoding for "undetected joint" and "padding" in
this data, so zeroing for masking would be ambiguous with those.
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from mouna.models.transformer import PoseTransformer


def sample_span_mask(
    lengths: torch.Tensor,
    max_len: int,
    mask_ratio: float = 0.4,
    span_min: int = 3,
    span_max: int = 8,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Returns a (B, T) bool mask, True = masked. Never masks positions at or
    beyond each sequence's own length (padding is never "masked" — it's
    already excluded from the loss separately, but keeping it out of the mask
    too avoids wasting mask budget on positions with no signal to predict)."""
    B = lengths.shape[0]
    mask = torch.zeros(B, max_len, dtype=torch.bool)
    for b in range(B):
        L = int(lengths[b].item())
        if L <= 0:
            continue
        target = max(1, int(round(L * mask_ratio)))
        covered = 0
        guard = 0
        while covered < target and guard < 100:
            guard += 1
            span_len = int(torch.randint(span_min, span_max + 1, (1,), generator=generator).item())
            span_len = min(span_len, L)
            start = int(torch.randint(0, L - span_len + 1, (1,), generator=generator).item())
            newly = (~mask[b, start:start + span_len]).sum().item()
            mask[b, start:start + span_len] = True
            covered += newly
    return mask


class MaskedPoseAutoencoder(nn.Module):
    """Wraps a PoseTransformer's encoder (input_proj + pos_enc + encoder) for
    masked-reconstruction pretraining. The wrapped model's classifier head is
    constructed but never trained or used — it exists only so the SAME
    PoseTransformer architecture/constructor is reused unmodified, and is
    explicitly stripped before the checkpoint is saved (see pretrain_transformer.py)."""

    def __init__(
        self,
        input_dim: int = 225,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.3,
        mask_ratio: float = 0.4,
        span_min: int = 3,
        span_max: int = 8,
    ):
        super().__init__()
        # num_classes is irrelevant here — classifier head is never used.
        self.backbone = PoseTransformer(
            input_dim=input_dim, d_model=d_model, nhead=nhead, num_layers=num_layers,
            dim_feedforward=dim_feedforward, dropout=dropout, num_classes=1,
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.mask_token, std=0.02)
        self.decoder_head = nn.Linear(d_model, input_dim)

        self.mask_ratio = mask_ratio
        self.span_min = span_min
        self.span_max = span_max

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            x: (B, T, input_dim) normalized pose features
            lengths: (B,) valid frame counts before padding

        Returns:
            loss: scalar masked-MSE reconstruction loss
            info: dict with 'mask' (B,T) and 'recon' (B,T,input_dim) for inspection
        """
        B, T, _ = x.shape
        mask = sample_span_mask(
            lengths, T, mask_ratio=self.mask_ratio, span_min=self.span_min, span_max=self.span_max,
        ).to(x.device)

        emb = self.backbone.input_proj(x)                     # (B, T, d_model)
        mask_tok = self.mask_token.expand(B, T, -1)
        emb = torch.where(mask.unsqueeze(-1), mask_tok, emb)
        emb = self.backbone.pos_enc(emb)

        pad_mask = torch.arange(T, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
        encoded = self.backbone.encoder(emb, src_key_padding_mask=pad_mask)  # (B, T, d_model)
        recon = self.decoder_head(encoded)                     # (B, T, input_dim)

        valid = ~pad_mask                                      # (B, T)
        target_mask = mask & valid                             # only score masked, non-padded positions
        diff2 = (recon - x) ** 2                                # (B, T, input_dim)
        per_pos = diff2.mean(dim=-1)                            # (B, T)
        denom = target_mask.sum().clamp(min=1)
        loss = (per_pos * target_mask.float()).sum() / denom

        return loss, {"mask": mask, "recon": recon}

    def encoder_state_dict(self) -> dict:
        """Only the transferable encoder weights — explicitly strips the
        (never-trained) classifier head so a fine-tune load can't accidentally
        pull in random classifier weights under a matching-shaped key."""
        full = self.backbone.state_dict()
        return {k: v for k, v in full.items() if not k.startswith("classifier.")}
