# Testing Guide for Mouna

Complete guide to test your sign language recognition system setup.

## Quick Start

```bash
# 1. Install dependencies
source venv/bin/activate  # Or create: python3 -m venv venv
make install-dev

# 2. Validate setup (no data needed)
make test-setup

# 3. Run end-to-end test with synthetic data (no data needed)
make test-e2e

# 4. Run unit tests
make test
```

---

## Phase 1: Installation Test (2 minutes)

### Step 1: Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 2: Install Package

```bash
# Option A: Development install (recommended)
make install-dev

# Option B: Manual install
pip install -e ".[dev]"
```

**Expected output:**
```
Successfully installed mouna-0.1.0
Installing collected packages: torch, opencv-python, mediapipe, ...
```

### Step 3: Validate Installation

```bash
make test-setup
```

**What it tests:**
- ✓ All dependencies can be imported (PyTorch, OpenCV, MediaPipe, Azure SDK, MLflow, Gemini)
- ✓ CUDA availability (if you have GPU)
- ✓ Mouna package modules import correctly
- ✓ Configuration can be loaded
- ✓ Models can be instantiated
- ✓ Preprocessing works

**Expected output:**
```
Testing imports...
✓ PyTorch 2.x.x
  CUDA available: True/False
✓ OpenCV x.x.x
✓ MediaPipe x.x.x
✓ Azure Storage SDK
✓ MLflow x.x.x
✓ Google Generative AI (Gemini)

Testing mouna package...
✓ Mouna package imported (version 0.1.0)
✓ mouna.utils.config
✓ mouna.data.ingestion
...

🎉 All tests passed! Your setup is ready.
```

---

## Phase 2: End-to-End Test (5 minutes)

Tests the entire pipeline without needing real WLASL data.

```bash
make test-e2e
```

**What it tests:**
1. **Keypoint Extraction**: Creates synthetic video, extracts MediaPipe keypoints
2. **Model Training**: Trains BiLSTM model for 5 epochs on synthetic data
3. **Evaluation**: Tests inference and metrics calculation
4. **Model Save/Load**: Tests checkpointing

**Expected output:**
```
Test 1: Keypoint Extraction
✓ Extracted keypoints from video
  Pose shape: (30, 33, 3)
  Left hand shape: (30, 21, 3)
  ...

Test 2: Model Training (5 epochs)
✓ Model initialized
  Device: cuda
  Parameters: 1,234,567
  Epoch 1/5 - Loss: 1.6094, Acc: 20.00%
  Epoch 2/5 - Loss: 1.4523, Acc: 35.00%
  ...

✓ All end-to-end tests passed!
```

---

## Phase 3: Real Data Test (30 minutes)

### Step 1: Update Environment Variables

Edit [.env](.env):
```bash
# Get your storage key from Azure Portal
AZURE_STORAGE_KEY=<your_actual_key>

# Optional: Get Gemini API key from https://makersuite.google.com/app/apikey
GEMINI_API_KEY=<your_key>
```

### Step 2: Download Sample WLASL Data

```bash
# Download 100 videos for testing (faster)
make download-wlasl

# Or download full dataset (takes longer)
make download-wlasl-full
```

**Expected output:**
```
Downloading WLASL dataset...
Downloading WLASL metadata...
Metadata saved to data/raw/wlasl/wlasl_metadata.json
Downloading videos: 100%|██████| 100/100
Downloaded 100 videos
```

### Step 3: Upload to Azure

```bash
make upload-data
```

**Expected output:**
```
Uploading to Azure Blob Storage...
Found 100 videos to upload
Uploading to Azure: 100%|██████| 100/100
✓ Uploaded 100 videos to sign-videos-bronze
```

### Step 4: Preprocess Videos

```bash
make preprocess-data
```

**Expected output:**
```
Preprocessing videos and extracting keypoints...
Processing videos: 100%|██████| 100/100
✓ Extracted keypoints saved to sign-videos-silver
```

---

## Phase 4: Training Test (1-2 hours)

### Step 1: Train Baseline Model

```bash
# Small test run (modify configs/dev.yaml to limit samples)
make train-baseline
```

**Monitor training:**
- Check terminal for loss/accuracy
- Open MLflow UI: `make mlflow-ui` (http://localhost:5000)

**Expected in MLflow:**
- Experiment: `sign-recognition-dev`
- Metrics: loss, accuracy, top5_accuracy
- Parameters: hidden_dim, num_layers, learning_rate
- Artifacts: model checkpoint

### Step 2: Evaluate Model

```bash
make evaluate
```

**Expected output:**
```
Evaluating model...
✓ Top-1 Accuracy: 45.2%
✓ Top-5 Accuracy: 78.3%
✓ Per-signer mean accuracy: 42.1%
✓ Mean latency: 12.5ms
```

---

## Phase 5: Advanced Testing

### Test Gemini QA Layer

```bash
# Make sure GEMINI_API_KEY is set in .env
make evaluate-with-gemini
```

### Test Multimodal Model

```bash
make train-multimodal
```

### Test with Databricks

1. Set up Databricks workspace
2. Update [.env](.env):
   ```
   DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
   DATABRICKS_TOKEN=your_token
   ```
3. Upload notebooks to Databricks:
   - Upload `notebooks/` folder
   - Run `01_bronze_ingest.py`

---

## Troubleshooting

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'mouna'`

**Solution:**
```bash
pip install -e .
```

### CUDA Issues

**Problem:** `RuntimeError: CUDA out of memory`

**Solution:**
- Reduce batch size in `configs/dev.yaml`
- Use CPU: Set `device: "cpu"` in config

### Azure Connection Failed

**Problem:** `Invalid storage account key`

**Solution:**
1. Go to Azure Portal
2. Storage Account → Access Keys
3. Copy key to `.env`

### MediaPipe Errors

**Problem:** `Segmentation fault` or MediaPipe crashes

**Solution:**
```bash
# Reinstall MediaPipe
pip uninstall mediapipe
pip install mediapipe --no-cache-dir
```

### Environment Variables Not Loading

**Problem:** Config fails to load Azure credentials

**Solution:**
```bash
# Check .env file exists
ls -la .env

# Verify it's being loaded
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print(os.getenv('AZURE_STORAGE_KEY'))"
```

---

## Test Checklist

Use this checklist to verify your setup:

- [ ] Virtual environment created and activated
- [ ] Dependencies installed (`make install-dev`)
- [ ] Setup validation passed (`make test-setup`)
- [ ] End-to-end test passed (`make test-e2e`)
- [ ] Unit tests passed (`make test`)
- [ ] Azure credentials configured in `.env`
- [ ] Sample WLASL data downloaded
- [ ] Data uploaded to Azure
- [ ] Videos preprocessed (keypoints extracted)
- [ ] Baseline model training works
- [ ] MLflow tracking UI accessible
- [ ] Model evaluation works
- [ ] Model checkpoints saved correctly

---

## Next Steps After Testing

Once all tests pass:

1. **Full Training**: Train on complete WLASL dataset
2. **Hyperparameter Tuning**: Experiment with different configs
3. **Multimodal Model**: Add vision backbone
4. **Gemini Integration**: Enable QA validation
5. **Publication Analysis**: Use metrics for paper

---

## Performance Benchmarks

Expected performance on test setup:

| Test | Expected Time | Expected Result |
|------|--------------|-----------------|
| `make test-setup` | 30 seconds | All imports pass |
| `make test-e2e` | 2-3 minutes | Training converges |
| `make download-wlasl` | 10-20 minutes | 100 videos |
| `make preprocess-data` | 15-30 minutes | 100 videos processed |
| `make train-baseline` | 1-2 hours | ~40-60% top-1 accuracy |

GPU vs CPU:
- GPU (RTX 3090): ~2-3x faster training
- CPU: Slower but functional for testing

---

## Getting Help

If tests fail:

1. Check error messages in terminal
2. Review [TESTING.md](TESTING.md) troubleshooting section
3. Check logs in `logs/` directory
4. Verify environment variables in `.env`
5. Try with fresh virtual environment

For CUDA issues, ensure:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
