# Mouna - Sign Language Recognition System

A sign language recognition system using WLASL dataset, Azure Blob Storage, and Databricks for baseline model development and research.

## Overview

This project implements a baseline sign language recognition model with the following components:

1. **Baseline Model**: Keypoints extraction (MediaPipe) + GRU/BiLSTM
2. **Multimodal Model**: Vision backbone (TimeSformer/VideoMAE) + temporal model
3. **Gemini QA Layer**: Label confidence validation and failure mode explanation
4. **MLflow Tracking**: Comprehensive experiment logging and metrics

### Architecture

```
WLASL Videos → Azure Blob Storage → Databricks
├── Bronze Layer: Raw videos
├── Silver Layer: Preprocessed frames + keypoints
├── Gold Layer: Model-ready features
└── Models:
    ├── Baseline: Keypoints → BiLSTM
    ├── Multimodal: Vision backbone + Temporal
    └── Gemini QA: Validation layer
```

## Project Structure

```
mouna/
├── src/mouna/              # Main package
│   ├── data/               # Data ingestion & preprocessing
│   ├── models/             # Model architectures
│   ├── evaluation/         # Metrics and evaluation
│   ├── gemini/             # Gemini QA integration
│   └── utils/              # Utilities and config
├── notebooks/              # Databricks notebooks
├── configs/                # Configuration files
├── tests/                  # Unit tests
├── .env                    # Environment variables (DO NOT COMMIT)
├── pyproject.toml          # Project dependencies
└── Makefile                # Common tasks
```

## Setup

### Prerequisites

- Python 3.9+
- Azure subscription with Blob Storage
- Databricks workspace (optional for local development)
- CUDA-capable GPU (recommended for training)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd mouna
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
# Install base dependencies
pip install -e .

# Install with development tools
pip install -e ".[dev]"

# Install all dependencies (including notebooks)
pip install -e ".[all]"
```

4. **Configure Azure credentials**

Copy `.env.example` to `.env` and fill in your Azure credentials:
```bash
cp .env.example .env
```

Edit `.env`:
```
AZURE_SUBSCRIPTION_ID=your-subscription-id
AZURE_RESOURCE_GROUP=your-resource-group
AZURE_STORAGE_ACCOUNT=your-storage-account
AZURE_REGION=eastus
AZURE_STORAGE_KEY=your-storage-key

# Gemini API
GEMINI_API_KEY=your-gemini-api-key

# MLflow (optional)
MLFLOW_TRACKING_URI=databricks
DATABRICKS_HOST=your-databricks-host
DATABRICKS_TOKEN=your-databricks-token
```

5. **Verify installation**
```bash
make test
```

## Quick Start

### 1. Download WLASL Dataset

```bash
make download-wlasl
```

### 2. Upload to Azure Blob Storage

```bash
python -m mouna.data.ingestion --upload
```

### 3. Preprocess Videos (Extract Keypoints)

Run in Databricks or locally:
```bash
python -m mouna.data.preprocessing --layer silver
```

### 4. Train Baseline Model

```bash
python -m mouna.models.baseline --config configs/dev.yaml
```

### 5. Evaluate Model

```bash
python -m mouna.evaluation.metrics --model baseline --checkpoint path/to/checkpoint.pth
```

## Usage

### Data Pipeline

**Bronze Layer** (Raw videos):
```python
from mouna.data.ingestion import WLASLDownloader

downloader = WLASLDownloader()
downloader.download_and_upload(
    output_dir="data/raw",
    azure_container="sign-videos-bronze"
)
```

**Silver Layer** (Keypoints extraction):
```python
from mouna.data.preprocessing import KeypointExtractor

extractor = KeypointExtractor()
extractor.process_videos(
    input_container="sign-videos-bronze",
    output_container="sign-videos-silver"
)
```

### Training

**Baseline Model**:
```python
from mouna.models.baseline import KeypointBiLSTM
from mouna.utils.config import load_config

config = load_config("configs/dev.yaml")
model = KeypointBiLSTM(config)
model.train()
```

**Multimodal Model**:
```python
from mouna.models.multimodal import MultimodalSignRecognition

model = MultimodalSignRecognition(config)
model.train()
```

### Evaluation

```python
from mouna.evaluation.metrics import SignRecognitionMetrics

metrics = SignRecognitionMetrics(model)
results = metrics.evaluate(
    test_loader=test_loader,
    metrics=["top1", "top5", "per_signer", "latency"]
)
print(results)
```

### Gemini QA Layer

```python
from mouna.gemini.qa_layer import GeminiQA

qa = GeminiQA(api_key=os.getenv("GEMINI_API_KEY"))
explanation = qa.validate_prediction(
    video_path="path/to/video.mp4",
    predicted_label="hello",
    confidence=0.65,
    top_k_predictions=[("hello", 0.65), ("goodbye", 0.20)]
)
```

## Development

### Running Tests

```bash
make test
```

### Code Formatting

```bash
make format
```

### Linting

```bash
make lint
```

### Type Checking

```bash
make typecheck
```

## Metrics

The system tracks the following metrics:

- **Top-1 Accuracy**: Percentage of correct top predictions
- **Top-k Accuracy**: Percentage where correct label is in top-k predictions (k=5)
- **Latency**: Inference time per video
- **Per-signer Generalization**: Accuracy across different signers (signer-independent evaluation)
- **Confusion Matrix**: Sign confusion patterns
- **Per-class Accuracy**: Performance on individual sign classes

## Configuration

Configuration files are in `configs/`:
- `dev.yaml`: Development configuration
- `prod.yaml`: Production configuration (if needed)
- `schemas/`: Data schemas

Example `dev.yaml`:
```yaml
data:
  azure_container_bronze: "sign-videos-bronze"
  azure_container_silver: "sign-videos-silver"
  azure_container_gold: "sign-videos-gold"
  num_classes: 2000  # WLASL-2000

model:
  baseline:
    input_dim: 1662  # MediaPipe landmarks (543 * 3 coordinates)
    hidden_dim: 512
    num_layers: 2
    dropout: 0.3
    bidirectional: true

  multimodal:
    vision_backbone: "timesformer"
    temporal_model: "transformer"
    fusion_strategy: "cross_attention"

training:
  batch_size: 32
  learning_rate: 0.001
  num_epochs: 100
  early_stopping_patience: 10

gemini:
  model: "gemini-1.5-pro"
  confidence_threshold: 0.7

mlflow:
  experiment_name: "sign-recognition-baseline"
  run_name: "keypoint-bilstm-v1"
```

## Databricks Notebooks

Notebooks in `notebooks/` directory:
1. `01_bronze_ingest.py`: Download WLASL and upload to Azure
2. `02_silver_preprocessing.py`: Extract keypoints and preprocess
3. `03_baseline_training.py`: Train baseline model
4. `04_multimodal_training.py`: Train multimodal model
5. `05_evaluation.py`: Comprehensive evaluation and analysis

## MLflow Tracking

All experiments are logged to MLflow with:
- Hyperparameters
- Metrics (accuracy, loss, latency)
- Model artifacts
- Visualizations (confusion matrices, learning curves)

View experiments:
```bash
mlflow ui
```

## Azure Resources

Required Azure resources:
- **Storage Account**: For blob storage (videos, features)
- **Databricks Workspace**: For distributed processing (optional for local dev)
- **Azure ML Workspace**: For MLflow tracking (optional)

## Contributing

1. Create a feature branch
2. Make changes
3. Run tests and linting
4. Submit pull request

## License

[Your License]

## Citation

If you use this code in your research, please cite:

```bibtex
@software{mouna2024,
  author = {Your Name},
  title = {Mouna: Sign Language Recognition System},
  year = {2024},
  url = {https://github.com/yourusername/mouna}
}
```

## Acknowledgments

- WLASL Dataset: [WLASL](https://dxli94.github.io/WLASL/)
- MediaPipe: Hand and pose landmark detection
- Azure Databricks: Distributed computing platform
