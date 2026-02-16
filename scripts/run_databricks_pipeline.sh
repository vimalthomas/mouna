#!/bin/bash
# Run Mouna pipeline on Databricks

set -e

# Load from .env file or environment
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

DATABRICKS_HOST="${DATABRICKS_HOST:-https://adb-7405614728635717.17.azuredatabricks.net}"
DATABRICKS_TOKEN="${DATABRICKS_TOKEN}"

if [ -z "$DATABRICKS_TOKEN" ]; then
    echo "Error: DATABRICKS_TOKEN not set"
    echo "Please set it in .env file or export it as environment variable"
    exit 1
fi

# Job IDs
DOWNLOAD_JOB_ID="1042809300717714"
PREPROCESS_JOB_ID="932672777567925"
TRAIN_JOB_ID="950196652779229"

echo "🚀 Mouna Databricks Pipeline"
echo "=============================="
echo ""

# Function to trigger job and wait
trigger_job() {
    local job_id=$1
    local job_name=$2

    echo "📌 Starting: $job_name"
    echo "   Job ID: $job_id"

    # Trigger job
    run_response=$(curl -s -X POST "${DATABRICKS_HOST}/api/2.1/jobs/run-now" \
        -H "Authorization: Bearer ${DATABRICKS_TOKEN}" \
        -H "Content-Type: application/json" \
        -d "{\"job_id\": $job_id}")

    run_id=$(echo $run_response | python3 -c "import sys, json; print(json.load(sys.stdin).get('run_id', ''))")

    if [ -z "$run_id" ]; then
        echo "❌ Failed to start job"
        echo "   Response: $run_response"
        return 1
    fi

    echo "   Run ID: $run_id"
    echo "   URL: ${DATABRICKS_HOST}/#job/${job_id}/run/${run_id}"
    echo ""

    # Monitor job
    echo "   ⏳ Waiting for job to complete..."
    while true; do
        status_response=$(curl -s -X GET "${DATABRICKS_HOST}/api/2.1/jobs/runs/get?run_id=${run_id}" \
            -H "Authorization: Bearer ${DATABRICKS_TOKEN}")

        state=$(echo $status_response | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('state', {}).get('life_cycle_state', ''))")
        result=$(echo $status_response | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('state', {}).get('result_state', ''))")

        if [ "$state" == "TERMINATED" ]; then
            if [ "$result" == "SUCCESS" ]; then
                echo "   ✅ Job completed successfully!"
                return 0
            else
                echo "   ❌ Job failed with result: $result"
                return 1
            fi
        fi

        echo -n "."
        sleep 10
    done
}

# Step 1: Download sample videos
echo "Step 1: Download Sample Videos (10 videos)"
echo "─────────────────────────────────────────────"
trigger_job $DOWNLOAD_JOB_ID "Download Sample Videos"
if [ $? -ne 0 ]; then
    echo "Pipeline stopped due to error"
    exit 1
fi
echo ""

# Step 2: Extract keypoints
echo "Step 2: Extract Keypoints"
echo "─────────────────────────────────────────────"
trigger_job $PREPROCESS_JOB_ID "Extract Keypoints"
if [ $? -ne 0 ]; then
    echo "Pipeline stopped due to error"
    exit 1
fi
echo ""

# Step 3: Train baseline model
echo "Step 3: Train Baseline Model"
echo "─────────────────────────────────────────────"
trigger_job $TRAIN_JOB_ID "Train Baseline Model"
if [ $? -ne 0 ]; then
    echo "Pipeline stopped due to error"
    exit 1
fi
echo ""

echo "✨ Pipeline Complete!"
echo "=============================="
echo ""
echo "Check results in Databricks:"
echo "  Workspace: ${DATABRICKS_HOST}/#workspace/Users/vjosep3@lsu.edu/mouna"
echo "  Jobs: ${DATABRICKS_HOST}/#job/list"
echo "  MLflow: ${DATABRICKS_HOST}/#mlflow"
