#!/usr/bin/env python3
"""5-fold cross-validated model confidence per clip, for the QA review panel.

Why CV and not just the existing checkpoint: that checkpoint trained on most
of this data, so its "confidence" on a WLASL/MS-ASL clip would be memorization,
not a real signal that the label is correct. Here every clip is scored only
by a fold that never saw it during training -- honest out-of-sample evidence
for every single clip, ASL Citizen included.

Reuses KeypointDataset/train_epoch/eval_epoch from train_transformer.py
directly so this scores against the exact same architecture and recipe
already validated all session -- no second code path to trust separately.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
from train_transformer import (  # noqa: E402
    D_MODEL, NHEAD, NUM_LAYERS, DIM_FEEDFORWARD, DROPOUT, BATCH_SIZE, LR,
    MIXUP_ALPHA, KeypointDataset, train_epoch, eval_epoch,
)
from mouna.models.transformer import PoseTransformer  # noqa: E402

N_FOLDS = 5
NUM_EPOCHS = 150
PATIENCE = 20  # slightly tighter than the benchmark recipe's 30 -- this is a
                # QA diagnostic across 5x the training runs, not the final number
SEED = 42

DATASET = Path("data/silver/keypoints/keypoints_official_split_v2norm_aslc.parquet")
OUT = Path("data/silver/qa/qa_model_confidence.parquet")


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def softmax_eval(model, loader, device):
    model.eval()
    all_probs, all_video_ids = [], []
    with torch.no_grad():
        for batch in loader:
            kp = batch["keypoints"].to(device)
            lengths = batch["seq_len"].to(device)
            logits = model(kp, lengths)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    df = pd.read_parquet(DATASET)
    df = df[df["success"]].reset_index(drop=True)
    df["_video_id_str"] = df["video_id"].astype(str)
    print(f"Dataset: {len(df)} sequences, {df['gloss'].nunique()} classes")

    glosses = sorted(df["gloss"].unique())
    label_to_idx = {g: i for i, g in enumerate(glosses)}
    idx_to_label = {i: g for g, i in label_to_idx.items()}
    num_classes = len(glosses)
    device = get_device()
    print(f"Device: {device}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_assignment = np.zeros(len(df), dtype=int)
    for fold_i, (_, holdout_idx) in enumerate(skf.split(df, df["gloss"])):
        fold_assignment[holdout_idx] = fold_i

    results = []

    for fold_i in range(N_FOLDS):
        holdout_df = df[fold_assignment == fold_i].reset_index(drop=True)
        remaining_df = df[fold_assignment != fold_i].reset_index(drop=True)
        # small internal val split off the remaining 4 folds, purely for early
        # stopping -- the holdout fold itself is never used for that decision
        train_df, val_df = train_test_split(
            remaining_df, test_size=0.10, stratify=remaining_df["gloss"], random_state=SEED
        )
        print(f"\n=== Fold {fold_i+1}/{N_FOLDS} === train:{len(train_df)} val:{len(val_df)} holdout:{len(holdout_df)}")

        train_ds = KeypointDataset(train_df, label_to_idx, augment=True)
        val_ds = KeypointDataset(val_df, label_to_idx, augment=False)
        holdout_ds = KeypointDataset(holdout_df, label_to_idx, augment=False)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        holdout_loader = DataLoader(holdout_ds, batch_size=BATCH_SIZE, shuffle=False)

        model = PoseTransformer(
            input_dim=train_ds.input_dim, d_model=D_MODEL, nhead=NHEAD,
            num_layers=NUM_LAYERS, dim_feedforward=DIM_FEEDFORWARD,
            dropout=DROPOUT, num_classes=num_classes,
        ).to(device)

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=1, eta_min=1e-5)

        best_val_acc, best_state, no_improve = 0.0, None, 0
        for epoch in range(NUM_EPOCHS):
            tl, ta = train_epoch(model, train_loader, criterion, optimizer, device)
            vl, va, _, _ = eval_epoch(model, val_loader, criterion, device)
            scheduler.step()
            if epoch % 10 == 0 or va > best_val_acc:
                print(f"  epoch {epoch+1:3d} | train {ta:.1f}% | val {va:.1f}%")
            if va > best_val_acc:
                best_val_acc = va
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= PATIENCE:
                    print(f"  early stop at epoch {epoch+1}, best val {best_val_acc:.1f}%")
                    break

        model.load_state_dict(best_state)
        probs = softmax_eval(model, holdout_loader, device)

        for i, row in holdout_df.iterrows():
            claimed_idx = label_to_idx[row["gloss"]]
            claimed_prob = float(probs[i, claimed_idx])
            top1_idx = int(probs[i].argmax())
            top3_idx = probs[i].argsort()[-3:][::-1]
            results.append({
                "video_id": row["_video_id_str"],
                "gloss": row["gloss"],
                "fold": fold_i,
                "claimed_prob": claimed_prob,
                "top1_gloss": idx_to_label[top1_idx],
                "top1_prob": float(probs[i, top1_idx]),
                "top1_matches_claimed": top1_idx == claimed_idx,
                "top3": [(idx_to_label[j], float(probs[i, j])) for j in top3_idx],
            })

        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    out_df = pd.DataFrame(results)
    out_df["top3"] = out_df["top3"].apply(lambda x: str(x))  # parquet-friendly
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(OUT, index=False)

    agree = out_df["top1_matches_claimed"].mean()
    print(f"\nDone. {len(out_df)} clips scored out-of-fold.")
    print(f"Model top-1 agrees with claimed label: {100*agree:.1f}%")
    print(f"Mean claimed-label probability: {out_df['claimed_prob'].mean():.3f}")
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
