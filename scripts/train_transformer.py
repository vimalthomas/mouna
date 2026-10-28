#!/usr/bin/env python3
"""Train PoseTransformer on the full local keypoints dataset."""

import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import mlflow
import mlflow.pytorch

# Overridable via env so the same script can train v1 (default) vs v2 (merged,
# OpenPose-recovered) without duplicating the file — isolates the data-only
# comparison requested before layering in the normalization change.
PARQUET = os.environ.get("MOUNA_PARQUET", "data/silver/keypoints/keypoints.parquet")
MODEL_OUT = os.environ.get("MOUNA_MODEL_OUT", "data/best_model_transformer.pth")
RUN_NAME = os.environ.get("MOUNA_RUN_NAME", "pose-transformer-wlasl+msasl")
DATASET_TAG = os.environ.get("MOUNA_DATASET_TAG", "WLASL+MS-ASL")
# Path to an encoder checkpoint from pretrain_transformer.py. When set, the
# classifier starts from self-supervised-pretrained weights instead of random
# init — everything else about the run (data, split, augmentation) is
# unchanged, so this isolates pretrained-vs-scratch as the only variable.
INIT_ENCODER = os.environ.get("MOUNA_INIT_ENCODER", "")

D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 3
DIM_FEEDFORWARD = 256
DROPOUT = 0.3
BATCH_SIZE = 16
LR = 5e-4
NUM_EPOCHS = 200
MAX_SEQ_LEN = 150
PATIENCE = 30
MIXUP_ALPHA = 0.4   # mixup augmentation strength
SEED = int(os.environ.get("MOUNA_SEED", 42))


class KeypointDataset(Dataset):
    def __init__(self, dataframe, label_to_idx, max_length=MAX_SEQ_LEN, augment=False):
        self.data = dataframe.reset_index(drop=True)
        self.label_to_idx = label_to_idx
        self.max_length = max_length
        self.augment = augment
        first_kp = pickle.loads(self.data.at[0, "keypoints_pickle"])
        self.input_dim = first_kp.shape[1]

    def __len__(self):
        return len(self.data)

    def _augment(self, kp):
        # Gaussian noise
        kp = kp + np.random.normal(0, 0.02, kp.shape).astype(np.float32)

        # Temporal scaling ±40% (was ±20%)
        scale = np.random.uniform(0.6, 1.4)
        new_len = max(1, int(len(kp) * scale))
        indices = np.linspace(0, len(kp) - 1, new_len).astype(int)
        kp = kp[indices]

        # Spatial flip (50% chance): mirror x-coords, swap left/right hand
        # Layout: pose(0:99), left_hand(99:162), right_hand(162:225)
        if np.random.random() < 0.5:
            kp = kp.copy()
            for seg_start, n_joints in [(0, 33), (99, 21), (162, 21)]:
                for j in range(n_joints):
                    kp[:, seg_start + j * 3] = 1.0 - kp[:, seg_start + j * 3]
            # swap left and right hand
            tmp = kp[:, 99:162].copy()
            kp[:, 99:162] = kp[:, 162:225]
            kp[:, 162:225] = tmp

        # Joint dropout (50% chance): zero out ~15% of joints per frame
        if np.random.random() < 0.5:
            D = kp.shape[1]
            n_joints = D // 3
            drop_n = max(1, int(n_joints * 0.15))
            drop_idx = np.random.choice(n_joints, drop_n, replace=False)
            kp = kp.copy()
            for j in drop_idx:
                kp[:, j * 3: j * 3 + 3] = 0.0

        return kp

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        kp = pickle.loads(row["keypoints_pickle"])
        if self.augment:
            kp = self._augment(kp)
        seq_len = min(len(kp), self.max_length)
        if len(kp) < self.max_length:
            kp = np.vstack([kp, np.zeros((self.max_length - len(kp), kp.shape[1]))])
        else:
            kp = kp[: self.max_length]
        return {
            "keypoints": torch.tensor(kp, dtype=torch.float32),
            "label":     torch.tensor(self.label_to_idx[row["gloss"]], dtype=torch.long),
            "seq_len":   torch.tensor(seq_len, dtype=torch.long),
        }


def mixup_batch(kp, labels, lengths, alpha):
    """Mixup: blend two random samples and interpolate labels."""
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    idx = torch.randperm(kp.size(0), device=kp.device)
    mixed_kp = lam * kp + (1 - lam) * kp[idx]
    return mixed_kp, labels, labels[idx], lengths, lam


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        kp = batch["keypoints"].to(device)
        labels = batch["label"].to(device)
        lengths = batch["seq_len"].to(device)
        optimizer.zero_grad()

        # Mixup (applied 50% of the time)
        if MIXUP_ALPHA > 0 and np.random.random() < 0.5:
            mixed_kp, labels_a, labels_b, lengths, lam = mixup_batch(kp, labels, lengths, MIXUP_ALPHA)
            logits = model(mixed_kp, lengths)
            loss = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)
        else:
            logits = model(kp, lengths)
            loss = criterion(logits, labels)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item()
        correct += logits.argmax(1).eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), 100.0 * correct / total


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            kp = batch["keypoints"].to(device)
            labels = batch["label"].to(device)
            lengths = batch["seq_len"].to(device)
            logits = model(kp, lengths)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            preds = logits.argmax(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    return total_loss / len(loader), 100.0 * correct / total, all_preds, all_labels


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    df = pd.read_parquet(PARQUET)
    df = df[df["success"]].reset_index(drop=True)
    print(f"Dataset: {len(df)} sequences, {df['gloss'].nunique()} classes")

    glosses = sorted(df["gloss"].unique())
    label_to_idx = {g: i for i, g in enumerate(glosses)}
    idx_to_label = {i: g for g, i in label_to_idx.items()}
    num_classes = len(glosses)

    from sklearn.model_selection import train_test_split

    if "official_split" in df.columns:
        # WLASL's real signer-disjoint benchmark split (from asl100.json), not a
        # random stratified split. Confirmed today: our previous random splits put
        # 91/143 WLASL test videos into the official TRAIN set — numbers from that
        # aren't comparable to any published result and may leak signer identity.
        # Non-WLASL data (no official label) always goes to train.
        train_df = df[df["official_split"] == "train"].reset_index(drop=True)
        val_df = df[df["official_split"] == "val"].reset_index(drop=True)
        test_df = df[df["official_split"] == "test"].reset_index(drop=True)
        print(f"Split (official WLASL benchmark split) — train: {len(train_df)}  val: {len(val_df)}  test: {len(test_df)}")
    elif "source" in df.columns and (df["source"] != df["source"].iloc[0]).any():
        # Mixed-quality dataset (camera MediaPipe + recovered OpenPose instances).
        # Val/test must only ever be drawn from the reliable camera-extracted pool —
        # otherwise a run with more noisy recovered data also gets a noisier test
        # set, and a lower test_acc could reflect harder test examples rather than
        # worse learning. Recovered instances only ever appear in training.
        clean = df[df["source"] == "mediapipe_video"].reset_index(drop=True)
        recovered = df[df["source"] != "mediapipe_video"].reset_index(drop=True)
        clean_train, temp_df = train_test_split(clean, test_size=0.30, stratify=clean["gloss"], random_state=SEED)
        val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["gloss"], random_state=SEED)
        train_df = pd.concat([clean_train, recovered], ignore_index=True)
        print(f"Split (fixed clean eval pool) — train: {len(train_df)} ({len(clean_train)} clean + {len(recovered)} recovered)  "
              f"val: {len(val_df)}  test: {len(test_df)}")
    else:
        train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["gloss"], random_state=SEED)
        val_df, test_df   = train_test_split(temp_df, test_size=0.50, stratify=temp_df["gloss"], random_state=SEED)
        print(f"Split — train: {len(train_df)}  val: {len(val_df)}  test: {len(test_df)}")

    train_ds = KeypointDataset(train_df, label_to_idx, augment=True)
    val_ds   = KeypointDataset(val_df,   label_to_idx, augment=False)
    test_ds  = KeypointDataset(test_df,  label_to_idx, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    input_dim = train_ds.input_dim

    from mouna.models.transformer import PoseTransformer
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = PoseTransformer(
        input_dim=input_dim,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        num_classes=num_classes,
    )

    if INIT_ENCODER:
        ckpt = torch.load(INIT_ENCODER, map_location="cpu", weights_only=False)
        encoder_state = ckpt["encoder_state_dict"] if "encoder_state_dict" in ckpt else ckpt
        result = model.load_state_dict(encoder_state, strict=False)
        print(f"\nLoaded pretrained encoder from {INIT_ENCODER}")
        print(f"  missing_keys (expected: classifier.* only, randomly initialized): {result.missing_keys}")
        print(f"  unexpected_keys (should be empty): {result.unexpected_keys}")
        assert all(k.startswith("classifier.") for k in result.missing_keys), (
            "Pretrained encoder load left non-classifier weights uninitialized — "
            "architecture mismatch between pretrain and fine-tune configs."
        )
        assert not result.unexpected_keys, "Pretrained checkpoint has keys PoseTransformer doesn't recognize."

    model.to(device)
    print(f"\nDevice: {device} | Params: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-4)
    # Warm restarts: restart LR every 50 epochs, decay by 0.5 each restart
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=1, eta_min=1e-5
    )

    mlflow.set_experiment("mouna/transformer")

    with mlflow.start_run(run_name=RUN_NAME):
        mlflow.log_params({
            "model": "transformer", "d_model": D_MODEL, "nhead": NHEAD,
            "num_layers": NUM_LAYERS, "dim_feedforward": DIM_FEEDFORWARD,
            "dropout": DROPOUT, "batch_size": BATCH_SIZE, "lr": LR,
            "num_classes": num_classes, "input_dim": input_dim,
            "train": len(train_df), "val": len(val_df), "test": len(test_df),
            "patience": PATIENCE, "device": str(device), "label_smoothing": 0.1,
            "mixup_alpha": MIXUP_ALPHA, "scheduler": "CosineWarmRestarts",
            "datasets": DATASET_TAG, "parquet": PARQUET,
            "init_encoder": INIT_ENCODER if INIT_ENCODER else "scratch",
        })

        best_val_acc = 0.0
        best_state = None
        no_improve = 0

        for epoch in range(NUM_EPOCHS):
            tl, ta = train_epoch(model, train_loader, criterion, optimizer, device)
            vl, va, _, _ = eval_epoch(model, val_loader, criterion, device)
            scheduler.step()
            lr_now = scheduler.get_last_lr()[0]

            mlflow.log_metrics({"train_loss": tl, "train_acc": ta, "val_loss": vl, "val_acc": va, "lr": lr_now}, step=epoch)
            print(f"Epoch {epoch+1:3d}/{NUM_EPOCHS} | lr {lr_now:.2e} | train {ta:.1f}% loss {tl:.4f} | val {va:.1f}% loss {vl:.4f}")

            if va > best_val_acc:
                best_val_acc = va
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
                print(f"  *** New best val acc: {best_val_acc:.1f}% ***")
            else:
                no_improve += 1
                if no_improve >= PATIENCE:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        model.load_state_dict(best_state)
        _, test_acc, test_preds, test_labels = eval_epoch(model, test_loader, criterion, device)
        print(f"\nBest val acc: {best_val_acc:.1f}%  |  Test acc: {test_acc:.1f}%")

        from collections import defaultdict
        class_correct = defaultdict(int)
        class_total = defaultdict(int)
        for p, l in zip(test_preds, test_labels):
            class_total[idx_to_label[l]] += 1
            if p == l:
                class_correct[idx_to_label[l]] += 1
        print("\nPer-class test accuracy:")
        for g in glosses:
            t = class_total[g]
            c = class_correct[g]
            print(f"  {g:14s}: {c}/{t}  ({100*c/t if t else 0:.0f}%)")

        mlflow.log_metrics({"test_acc": test_acc, "best_val_acc": best_val_acc})
        mlflow.pytorch.log_model(model, "best_model", serialization_format="pickle")

        torch.save({
            "model_state_dict": best_state,
            "label_to_idx": label_to_idx,
            "idx_to_label": idx_to_label,
            "config": {
                "input_dim": input_dim, "d_model": D_MODEL, "nhead": NHEAD,
                "num_layers": NUM_LAYERS, "dim_feedforward": DIM_FEEDFORWARD,
                "dropout": DROPOUT, "num_classes": num_classes,
            },
        }, MODEL_OUT)
        print(f"\nModel saved → {MODEL_OUT}")


if __name__ == "__main__":
    main()
