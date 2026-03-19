#!/usr/bin/env bash
# ── Local pipeline: validate → upload bronze → (Databricks handles the rest) ──
set -e
cd "$(dirname "$0")/.."

# Load env vars from .env
set -a; source .env; set +a

echo "=========================================="
echo "Step 1/3: Validate downloaded videos"
echo "=========================================="
python3 scripts/validate_videos.py

echo ""
echo "=========================================="
echo "Step 2/3: Upload validated videos to Azure bronze"
echo "=========================================="
python3 scripts/upload_bronze.py

echo ""
echo "=========================================="
echo "Step 3/3: Deploy & run Databricks pipeline"
echo "=========================================="
echo "Deploying bundle..."
databricks bundle deploy --profile mouna

echo ""
echo "Running pipeline (extract_keypoints → train_model)..."
echo "Provide your Azure storage key when prompted, or set azure_storage_key variable in databricks.yml."
databricks bundle run mouna_pipeline --profile mouna \
  -var="azure_storage_key=${AZURE_STORAGE_KEY}"

echo ""
echo "Done! Check MLflow at your Databricks workspace for results."
