#!/usr/bin/env python3
"""Self-supervised masked-reconstruction pretraining for PoseTransformer's
encoder, on the unlabeled WLASL pose pool (scripts/build_pretrain_pool.py).

Mirrors train_transformer.py's conventions (env-var overrides, device
selection, MLflow experiment, seed handling) so pretrain/fine-tune stay
directly comparable. Architecture hyperparameters (D_MODEL, NHEAD,
NUM_LAYERS, DIM_FEEDFORWARD, MAX_SEQ_LEN) are imported from
train_transformer.py rather than duplicated, so the two can't silently drift
apart and break weight loading.
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import mlflow

sys.path.insert(0, "scripts")
from train_transformer import (  # noqa: E402
    D_MODEL, NHEAD, NUM_LAYERS, DIM_FEEDFORWARD, DROPOUT, MAX_SEQ_LEN, KeypointDataset,
)
sys.path.insert(0, "src")
from mouna.models.pretraining import MaskedPoseAutoencoder  # noqa: E402

PARQUET = os.environ.get("MOUNA_PRETRAIN_PARQUET", "data/silver/keypoints/pretrain_pool.parquet")
MODEL_OUT = os.environ.get("MOUNA_PRETRAIN_MODEL_OUT", "data/pretrained_encoder.pth")
RUN_NAME = os.environ.get("MOUNA_PRETRAIN_RUN_NAME", "pose-pretrain-masked-recon")

SEED = int(os.environ.get("MOUNA_PRETRAIN_SEED", 42))
MASK_RATIO = float(os.environ.get("MOUNA_PRETRAIN_MASK_RATIO", 0.4))
SPAN_MIN = int(os.environ.get("MOUNA_PRETRAIN_SPAN_LEN_MIN", 3))
SPAN_MAX = int(os.environ.get("MOUNA_PRETRAIN_SPAN_LEN_MAX", 8))
EPOCHS = int(os.environ.get("MOUNA_PRETRAIN_EPOCHS", 60))
BATCH_SIZE = int(os.environ.get("MOUNA_PRETRAIN_BATCH_SIZE", 32))
LR = float(os.environ.get("MOUNA_PRETRAIN_LR", 5e-4))
PATIENCE = int(os.environ.get("MOUNA_PRETRAIN_PATIENCE", 10))


def run_epoch(model, loader, optimizer, device, train: bool):
    model.train(train)
    total_loss, n = 0.0, 0
    for batch in loader:
        kp = batch["keypoints"].to(device)
        lengths = batch["seq_len"].to(device)

        if train:
            optimizer.zero_grad()
            loss, _ = model(kp, lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        else:
            with torch.no_grad():
                loss, _ = model(kp, lengths)

        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    df = pd.read_parquet(PARQUET)
    df = df[df["success"]].reset_index(drop=True)
    print(f"Pretrain pool: {len(df)} sequences")

    # No gloss labels to stratify on (all rows share the dummy "unlabeled"
    # class) — plain random split just to monitor reconstruction convergence.
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(df))
    n_val = max(1, int(0.05 * len(df)))
    val_df = df.iloc[idx[:n_val]].reset_index(drop=True)
    train_df = df.iloc[idx[n_val:]].reset_index(drop=True)
    print(f"Split — train: {len(train_df)}  val: {len(val_df)}")

    label_to_idx = {"unlabeled": 0}
    train_ds = KeypointDataset(train_df, label_to_idx, max_length=MAX_SEQ_LEN, augment=False)
    val_ds = KeypointDataset(val_df, label_to_idx, max_length=MAX_SEQ_LEN, augment=False)
    input_dim = train_ds.input_dim

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = MaskedPoseAutoencoder(
        input_dim=input_dim, d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT,
        mask_ratio=MASK_RATIO, span_min=SPAN_MIN, span_max=SPAN_MAX,
    ).to(device)
    print(f"Device: {device} | Params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=1, eta_min=1e-5)

    mlflow.set_experiment("mouna/transformer")
    with mlflow.start_run(run_name=RUN_NAME):
        mlflow.log_params({
            "phase": "pretrain", "d_model": D_MODEL, "nhead": NHEAD, "num_layers": NUM_LAYERS,
            "dim_feedforward": DIM_FEEDFORWARD, "dropout": DROPOUT, "batch_size": BATCH_SIZE,
            "lr": LR, "input_dim": input_dim, "train": len(train_df), "val": len(val_df),
            "mask_ratio": MASK_RATIO, "span_min": SPAN_MIN, "span_max": SPAN_MAX,
            "epochs_max": EPOCHS, "patience": PATIENCE, "device": str(device), "seed": SEED,
            "parquet": PARQUET,
        })

        best_val, best_state, no_improve = float("inf"), None, 0
        for epoch in range(EPOCHS):
            tl = run_epoch(model, train_loader, optimizer, device, train=True)
            vl = run_epoch(model, val_loader, optimizer, device, train=False)
            scheduler.step()
            lr_now = scheduler.get_last_lr()[0]
            mlflow.log_metrics({"train_recon_loss": tl, "val_recon_loss": vl, "lr": lr_now}, step=epoch)
            print(f"Epoch {epoch+1:3d}/{EPOCHS} | lr {lr_now:.2e} | train {tl:.4f} | val {vl:.4f}")

            if vl < best_val:
                best_val = vl
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
                print(f"  *** New best val recon loss: {best_val:.4f} ***")
            else:
                no_improve += 1
                if no_improve >= PATIENCE:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        model.load_state_dict(best_state)
        mlflow.log_metrics({"best_val_recon_loss": best_val})

        encoder_state = model.encoder_state_dict()
        torch.save({
            "encoder_state_dict": encoder_state,
            "config": {
                "input_dim": input_dim, "d_model": D_MODEL, "nhead": NHEAD,
                "num_layers": NUM_LAYERS, "dim_feedforward": DIM_FEEDFORWARD, "dropout": DROPOUT,
            },
        }, MODEL_OUT)
        print(f"\nEncoder saved (classifier head stripped) -> {MODEL_OUT}")
        print(f"Keys saved: {sorted(encoder_state.keys())[:5]} ... ({len(encoder_state)} total)")


if __name__ == "__main__":
    main()
