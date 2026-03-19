# Databricks notebook source
# MAGIC %md
# MAGIC # Train Baseline Model (BiLSTM)
# MAGIC Train a keypoint-based BiLSTM classifier on the silver layer keypoints.

# COMMAND ----------

import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import mlflow
import mlflow.pytorch
from sklearn.model_selection import StratifiedShuffleSplit

# Inline sys.path — makes the mouna package importable
_nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
_src_path = "/Workspace/" + "/".join(_nb_path.split("/")[1:-3]) + "/src"
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

SILVER_PATH = "abfss://sign-videos-silver@mounastorage2025.dfs.core.windows.net/"

# Hyperparameters — tuned for WLASL-100 scale (~1000-1500 samples, 100 classes)
HIDDEN_DIM    = 256
NUM_LAYERS    = 2
DROPOUT       = 0.5
BIDIRECTIONAL = True
BATCH_SIZE    = 16
LR            = 1e-3
NUM_EPOCHS    = 100
MAX_SEQ_LEN   = 150
TRAIN_FRAC    = 0.70
VAL_FRAC      = 0.15
# TEST_FRAC   = 0.15  (remainder)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Keypoints from Silver Layer

# COMMAND ----------

keypoints_df = spark.read.parquet(f"{SILVER_PATH}keypoints/keypoints.parquet")
keypoints_pd = keypoints_df.filter("success = true").toPandas()

print(f"Total rows:        {keypoints_df.count()}")
print(f"Successful:        {len(keypoints_pd)}")
print(f"Unique glosses:    {keypoints_pd['gloss'].nunique()}")
print(f"\nPer-class counts:")
print(keypoints_pd['gloss'].value_counts().to_string())

if len(keypoints_pd) < 10:
    raise RuntimeError(
        f"Need at least 10 successful sequences; got {len(keypoints_pd)}. "
        "Re-run extract_keypoints task with more data."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build PyTorch Dataset

# COMMAND ----------

class KeypointDataset(Dataset):
    """Pad/truncate keypoint sequences and map glosses to integer labels."""

    def __init__(self, dataframe, label_to_idx, max_length: int = MAX_SEQ_LEN):
        self.data = dataframe.reset_index(drop=True)
        self.max_length = max_length
        self.label_to_idx = label_to_idx
        self.num_classes = len(label_to_idx)
        first_kp = pickle.loads(self.data.at[0, "keypoints_pickle"])
        self.input_dim = first_kp.shape[1] if first_kp.ndim == 2 else len(first_kp[0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        kp = pickle.loads(row["keypoints_pickle"])
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

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stratified 70 / 15 / 15 Split

# COMMAND ----------

unique_glosses = sorted(keypoints_pd["gloss"].unique())
label_to_idx = {g: i for i, g in enumerate(unique_glosses)}
idx_to_label = {i: g for g, i in label_to_idx.items()}

labels = keypoints_pd["gloss"].values

# First split: train vs (val + test)
sss1 = StratifiedShuffleSplit(n_splits=1, test_size=(1 - TRAIN_FRAC), random_state=42)
train_idx, valtest_idx = next(sss1.split(keypoints_pd, labels))

# Second split: val vs test from the remainder
valtest_labels = labels[valtest_idx]
test_frac_of_valtest = 0.5  # 50% of 30% = 15% of total
sss2 = StratifiedShuffleSplit(n_splits=1, test_size=test_frac_of_valtest, random_state=42)
val_rel_idx, test_rel_idx = next(sss2.split(keypoints_pd.iloc[valtest_idx], valtest_labels))
val_idx  = valtest_idx[val_rel_idx]
test_idx = valtest_idx[test_rel_idx]

train_df = keypoints_pd.iloc[train_idx]
val_df   = keypoints_pd.iloc[val_idx]
test_df  = keypoints_pd.iloc[test_idx]

print(f"Train: {len(train_df)}  |  Val: {len(val_df)}  |  Test: {len(test_df)}")
print(f"Classes: {len(unique_glosses)}")

train_ds = KeypointDataset(train_df, label_to_idx)
val_ds   = KeypointDataset(val_df,   label_to_idx)
test_ds  = KeypointDataset(test_df,  label_to_idx)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=False)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Initialize Model

# COMMAND ----------

from mouna.models.baseline import create_baseline_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = create_baseline_model(
    model_type="bilstm",
    input_dim=train_ds.input_dim,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    num_classes=train_ds.num_classes,
    dropout=DROPOUT,
    bidirectional=BIDIRECTIONAL,
)
model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Device: {device}  |  Parameters: {total_params:,}  |  Classes: {train_ds.num_classes}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train with MLflow (cosine LR schedule + early stopping)

# COMMAND ----------

mlflow.set_experiment("/mouna/baseline")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-5)

PATIENCE = 15  # early stopping patience (epochs without improvement)

with mlflow.start_run(run_name="bilstm-wlasl100"):
    mlflow.log_params({
        "model_type": "bilstm", "hidden_dim": HIDDEN_DIM, "num_layers": NUM_LAYERS,
        "dropout": DROPOUT, "bidirectional": BIDIRECTIONAL, "batch_size": BATCH_SIZE,
        "learning_rate": LR, "num_classes": train_ds.num_classes,
        "input_dim": train_ds.input_dim,
        "train_size": len(train_ds), "val_size": len(val_ds), "test_size": len(test_ds),
        "max_seq_len": MAX_SEQ_LEN, "patience": PATIENCE,
    })

    best_val_acc = 0.0
    no_improve   = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for batch in train_loader:
            kp      = batch["keypoints"].to(device)
            labels  = batch["label"].to(device)
            lengths = batch["seq_len"].to(device)
            optimizer.zero_grad()
            logits = model(kp, lengths)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss    += loss.item()
            train_correct += logits.argmax(1).eq(labels).sum().item()
            train_total   += labels.size(0)
        scheduler.step()

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                kp      = batch["keypoints"].to(device)
                labels  = batch["label"].to(device)
                lengths = batch["seq_len"].to(device)
                logits  = model(kp, lengths)
                loss    = criterion(logits, labels)
                val_loss    += loss.item()
                val_correct += logits.argmax(1).eq(labels).sum().item()
                val_total   += labels.size(0)

        avg_tl = train_loss / len(train_loader)
        avg_vl = val_loss   / max(len(val_loader), 1)
        ta     = 100.0 * train_correct / max(train_total, 1)
        va     = 100.0 * val_correct   / max(val_total, 1)
        lr_now = scheduler.get_last_lr()[0]

        mlflow.log_metrics(
            {"train_loss": avg_tl, "train_acc": ta, "val_loss": avg_vl, "val_acc": va, "lr": lr_now},
            step=epoch,
        )
        print(f"Epoch {epoch+1:3d}/{NUM_EPOCHS} | lr {lr_now:.2e} | "
              f"train loss {avg_tl:.4f} acc {ta:.1f}% | val loss {avg_vl:.4f} acc {va:.1f}%")

        if va > best_val_acc:
            best_val_acc = va
            no_improve   = 0
            mlflow.pytorch.log_model(model, "best_model")
            print(f"  *** New best val acc: {best_val_acc:.1f}% ***")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1} (no improvement for {PATIENCE} epochs)")
                break

    # ── Test set evaluation ────────────────────────────────────────────────
    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for batch in test_loader:
            kp      = batch["keypoints"].to(device)
            labels  = batch["label"].to(device)
            lengths = batch["seq_len"].to(device)
            logits  = model(kp, lengths)
            test_correct += logits.argmax(1).eq(labels).sum().item()
            test_total   += labels.size(0)

    test_acc = 100.0 * test_correct / max(test_total, 1)
    mlflow.log_metric("test_acc", test_acc)

    print(f"\nBest val accuracy: {best_val_acc:.1f}%")
    print(f"Test accuracy:     {test_acc:.1f}%")

    # Save label map as artifact
    import json, tempfile, os
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
        json.dump({"label_to_idx": label_to_idx, "idx_to_label": idx_to_label}, tmp)
        tmp_path = tmp.name
    mlflow.log_artifact(tmp_path, "label_map")
    os.unlink(tmp_path)
