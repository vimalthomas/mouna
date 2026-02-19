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

# Inline sys.path — makes the mouna package importable
_nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
_src_path = "/Workspace/" + "/".join(_nb_path.split("/")[1:-3]) + "/src"
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

SILVER_PATH = "abfss://sign-videos-silver@mounastorage2025.dfs.core.windows.net/"

# Hyperparameters
HIDDEN_DIM    = 128
NUM_LAYERS    = 1
DROPOUT       = 0.3
BIDIRECTIONAL = True
BATCH_SIZE    = 2
LR            = 1e-3
NUM_EPOCHS    = 10
MAX_SEQ_LEN   = 150

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Keypoints from Silver Layer

# COMMAND ----------

keypoints_df = spark.read.format("delta").load(f"{SILVER_PATH}keypoints/")
keypoints_pd = keypoints_df.filter("success = true").toPandas()

print(f"Total sequences:  {keypoints_df.count()}")
print(f"Successful:       {len(keypoints_pd)}")
print(f"Unique glosses:   {keypoints_pd['gloss'].nunique()}")

if len(keypoints_pd) < 2:
    raise RuntimeError(
        f"Need at least 2 successful sequences to train; got {len(keypoints_pd)}. "
        "Re-run bronze + silver with a larger sample_size."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build PyTorch Dataset

# COMMAND ----------

class KeypointDataset(Dataset):
    """Pad/truncate keypoint sequences and map glosses to integer labels."""

    def __init__(self, dataframe, max_length: int = MAX_SEQ_LEN):
        self.data = dataframe.reset_index(drop=True)
        self.max_length = max_length
        unique_glosses = sorted(self.data["gloss"].unique())
        self.label_to_idx = {g: i for i, g in enumerate(unique_glosses)}
        self.num_classes = len(unique_glosses)
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


dataset = KeypointDataset(keypoints_pd)
print(f"Dataset size: {len(dataset)}  |  classes: {dataset.num_classes}  |  input_dim: {dataset.input_dim}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train / Val Split

# COMMAND ----------

n_total = len(dataset)
n_train = max(1, int(0.8 * n_total))
n_val   = n_total - n_train

train_ds, val_ds = torch.utils.data.random_split(
    dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=False)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

print(f"Train: {n_train}  |  Val: {n_val}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Initialize Model

# COMMAND ----------

from mouna.models.baseline import create_baseline_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = create_baseline_model(
    model_type="bilstm",
    input_dim=dataset.input_dim,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    num_classes=dataset.num_classes,
    dropout=DROPOUT,
    bidirectional=BIDIRECTIONAL,
)
model.to(device)

print(f"Device: {device}  |  Parameters: {sum(p.numel() for p in model.parameters()):,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train with MLflow

# COMMAND ----------

mlflow.set_experiment("/mouna/baseline")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

with mlflow.start_run(run_name="bilstm-baseline"):
    mlflow.log_params({
        "model_type": "bilstm", "hidden_dim": HIDDEN_DIM, "num_layers": NUM_LAYERS,
        "dropout": DROPOUT, "bidirectional": BIDIRECTIONAL, "batch_size": BATCH_SIZE,
        "learning_rate": LR, "num_classes": dataset.num_classes,
        "input_dim": dataset.input_dim, "train_size": n_train, "val_size": n_val,
    })

    best_val_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for batch in train_loader:
            kp, labels, lengths = batch["keypoints"].to(device), batch["label"].to(device), batch["seq_len"].to(device)
            optimizer.zero_grad()
            logits = model(kp, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss    += loss.item()
            train_correct += logits.argmax(1).eq(labels).sum().item()
            train_total   += labels.size(0)

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                kp, labels, lengths = batch["keypoints"].to(device), batch["label"].to(device), batch["seq_len"].to(device)
                logits = model(kp, lengths)
                loss = criterion(logits, labels)
                val_loss    += loss.item()
                val_correct += logits.argmax(1).eq(labels).sum().item()
                val_total   += labels.size(0)

        avg_tl = train_loss / len(train_loader)
        avg_vl = val_loss   / len(val_loader)
        ta     = 100.0 * train_correct / train_total
        va     = 100.0 * val_correct   / val_total

        mlflow.log_metrics({"train_loss": avg_tl, "train_acc": ta, "val_loss": avg_vl, "val_acc": va}, step=epoch)
        print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | train loss {avg_tl:.4f} acc {ta:.1f}% | val loss {avg_vl:.4f} acc {va:.1f}%")

        if va > best_val_acc:
            best_val_acc = va
            mlflow.pytorch.log_model(model, "best_model")

    print(f"\nBest val accuracy: {best_val_acc:.1f}%")
