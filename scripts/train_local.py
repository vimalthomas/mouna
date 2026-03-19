#!/usr/bin/env python3
"""Train BiLSTM on the full local keypoints dataset with train/val/test split."""

import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import mlflow
import mlflow.pytorch

PARQUET = "data/silver/keypoints/keypoints.parquet"
MODEL_OUT = "data/best_model.pth"

HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.5
BATCH_SIZE = 16
LR = 1e-3
NUM_EPOCHS = 100
MAX_SEQ_LEN = 150
PATIENCE = 20
SEED = 42


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
        # Random temporal scaling (speed up/slow down by ±20%)
        scale = np.random.uniform(0.8, 1.2)
        new_len = max(1, int(len(kp) * scale))
        indices = np.linspace(0, len(kp) - 1, new_len).astype(int)
        kp = kp[indices]
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


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        kp = batch["keypoints"].to(device)
        labels = batch["label"].to(device)
        lengths = batch["seq_len"].to(device)
        optimizer.zero_grad()
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
    print(df["gloss"].value_counts().to_string())

    # Label mapping
    glosses = sorted(df["gloss"].unique())
    label_to_idx = {g: i for i, g in enumerate(glosses)}
    idx_to_label = {i: g for g, i in label_to_idx.items()}
    num_classes = len(glosses)

    # Stratified split: 70/15/15
    from sklearn.model_selection import train_test_split
    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["gloss"], random_state=SEED)
    val_df, test_df   = train_test_split(temp_df, test_size=0.50, stratify=temp_df["gloss"], random_state=SEED)
    print(f"\nSplit — train: {len(train_df)}  val: {len(val_df)}  test: {len(test_df)}")

    train_ds = KeypointDataset(train_df, label_to_idx, augment=True)
    val_ds   = KeypointDataset(val_df,   label_to_idx, augment=False)
    test_ds  = KeypointDataset(test_df,  label_to_idx, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    input_dim = train_ds.input_dim

    # Model
    from mouna.models.baseline import create_baseline_model
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")   # Apple Silicon GPU
    else:
        device = torch.device("cpu")
    model = create_baseline_model(
        model_type="bilstm",
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_classes=num_classes,
        dropout=DROPOUT,
        bidirectional=True,
    )
    model.to(device)
    print(f"\nDevice: {device} | Params: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-5)

    mlflow.set_experiment("mouna/local_baseline")

    with mlflow.start_run(run_name="bilstm-wlasl100"):
        mlflow.log_params({
            "model": "bilstm", "hidden_dim": HIDDEN_DIM, "num_layers": NUM_LAYERS,
            "dropout": DROPOUT, "batch_size": BATCH_SIZE, "lr": LR,
            "num_classes": num_classes, "input_dim": input_dim,
            "train": len(train_df), "val": len(val_df), "test": len(test_df),
            "patience": PATIENCE, "device": str(device),
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

        # Test with best model
        model.load_state_dict(best_state)
        _, test_acc, test_preds, test_labels = eval_epoch(model, test_loader, criterion, device)
        print(f"\nBest val acc: {best_val_acc:.1f}%  |  Test acc: {test_acc:.1f}%")

        # Per-class accuracy
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
            print(f"  {g:12s}: {c}/{t}  ({100*c/t if t else 0:.0f}%)")

        mlflow.log_metrics({"test_acc": test_acc, "best_val_acc": best_val_acc})
        mlflow.pytorch.log_model(model, "best_model")

        # Save checkpoint
        torch.save({
            "model_state_dict": best_state,
            "label_to_idx": label_to_idx,
            "idx_to_label": idx_to_label,
            "config": {
                "input_dim": input_dim, "hidden_dim": HIDDEN_DIM,
                "num_layers": NUM_LAYERS, "num_classes": num_classes,
                "dropout": DROPOUT, "bidirectional": True,
            },
        }, MODEL_OUT)
        print(f"\nModel saved → {MODEL_OUT}")


if __name__ == "__main__":
    main()
