# Databricks notebook source
# MAGIC %md
# MAGIC # Train Baseline Model (BiLSTM)
# MAGIC
# MAGIC Train keypoint-based BiLSTM model using mouna package

# COMMAND ----------

# MAGIC %run ./01_setup_environment

# COMMAND ----------

import torch
from torch.utils.data import Dataset, DataLoader
from mouna.models.baseline import KeypointBiLSTM, create_baseline_model
from mouna.utils.config import load_config
import mlflow
import mlflow.pytorch
import pickle
import numpy as np

SILVER_PATH = "abfss://sign-videos-silver@mysignstorage.dfs.core.windows.net/"
GOLD_PATH = "abfss://sign-videos-gold@mysignstorage.dfs.core.windows.net/"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Configuration

# COMMAND ----------

# Load config from the package
config = load_config()

print(f"Model config:")
print(f"  Hidden dim: {config.baseline.hidden_dim}")
print(f"  Num layers: {config.baseline.num_layers}")
print(f"  Batch size: {config.training.batch_size}")
print(f"  Learning rate: {config.training.learning_rate}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Keypoints from Silver Layer

# COMMAND ----------

# Load keypoints
keypoints_df = spark.read.format("delta").load(f"{SILVER_PATH}keypoints/")
keypoints_pd = keypoints_df.toPandas()

print(f"Loaded {len(keypoints_pd)} keypoint sequences")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare Dataset

# COMMAND ----------

class KeypointDataset(Dataset):
    """Dataset for keypoint sequences"""

    def __init__(self, dataframe, max_length=150):
        self.data = dataframe
        self.max_length = max_length

        # Create label mapping
        unique_glosses = sorted(dataframe['gloss'].unique())
        self.label_to_idx = {gloss: idx for idx, gloss in enumerate(unique_glosses)}
        self.num_classes = len(unique_glosses)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Deserialize keypoints
        keypoints = pickle.loads(row['keypoints_pickle'])

        # Pad/truncate
        seq_len = min(len(keypoints), self.max_length)
        if len(keypoints) < self.max_length:
            padding = np.zeros((self.max_length - len(keypoints), keypoints.shape[1]))
            keypoints = np.vstack([keypoints, padding])
        else:
            keypoints = keypoints[:self.max_length]

        # Get label
        label = self.label_to_idx[row['gloss']]

        return {
            'keypoints': torch.tensor(keypoints, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.long),
            'sequence_length': torch.tensor(seq_len, dtype=torch.long)
        }

# Create dataset
dataset = KeypointDataset(keypoints_pd)

# Split train/val
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size)

print(f"Train samples: {len(train_dataset)}")
print(f"Val samples: {len(val_dataset)}")
print(f"Num classes: {dataset.num_classes}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Initialize Model

# COMMAND ----------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = create_baseline_model(
    model_type=config.baseline.model_type,
    input_dim=config.baseline.input_dim,
    hidden_dim=config.baseline.hidden_dim,
    num_layers=config.baseline.num_layers,
    num_classes=dataset.num_classes,
    dropout=config.baseline.dropout,
    bidirectional=config.baseline.bidirectional
)

model.to(device)

print(f"✅ Model initialized on {device}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train with MLflow

# COMMAND ----------

mlflow.set_experiment("/Users/vjosep3@lsu.edu/mouna-baseline")

with mlflow.start_run():
    # Log parameters
    mlflow.log_params({
        "model_type": config.baseline.model_type,
        "hidden_dim": config.baseline.hidden_dim,
        "num_layers": config.baseline.num_layers,
        "batch_size": config.training.batch_size,
        "learning_rate": config.training.learning_rate,
        "num_classes": dataset.num_classes,
    })

    # Training setup
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)

    # Training loop
    num_epochs = 10  # Reduced for demo
    best_val_acc = 0

    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            keypoints = batch['keypoints'].to(device)
            labels = batch['label'].to(device)
            lengths = batch['sequence_length'].to(device)

            optimizer.zero_grad()
            outputs = model(keypoints, lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_acc = 100. * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                keypoints = batch['keypoints'].to(device)
                labels = batch['label'].to(device)
                lengths = batch['sequence_length'].to(device)

                outputs = model(keypoints, lengths)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100. * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        # Log metrics
        mlflow.log_metrics({
            "train_loss": avg_train_loss,
            "train_acc": train_acc,
            "val_loss": avg_val_loss,
            "val_acc": val_acc,
        }, step=epoch)

        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            mlflow.pytorch.log_model(model, "best_model")

    print(f"\n✅ Training complete! Best val accuracy: {best_val_acc:.2f}%")
