#!/bin/bash
# v5canonmotion (palm orientation + velocity, re-tested on the now-validated
# canonicalized dataset) -- seeds 1,2,3,4. Seed 42 already ran as the
# directional check (67.1%); v4canon's 5-seed baseline already exists from
# the canonicalization sweep. Reuses both rather than re-running.
set -e
cd ~/mouna/mouna
source venv/bin/activate

RESULTS=/tmp/mouna_v5canonmotion_results.log
echo "SEED=42 VARIANT=v5canonmotion TEST_ACC=67.1" > "$RESULTS"

for SEED in 1 2 3 4; do
  echo "=== SEED $SEED : v5canonmotion ==="
  MOUNA_SEED=$SEED \
    PYTHONPATH=src MLFLOW_ALLOW_FILE_STORE=true \
    MOUNA_DISABLE_FLIP_AUG=1 \
    MOUNA_PARQUET="data/silver/keypoints/keypoints_official_split_v5canonmotion.parquet" \
    MOUNA_MODEL_OUT="data/sweep/v5canonmotion_seed${SEED}.pth" \
    MOUNA_RUN_NAME="pose-transformer-v5canonmotion-seed${SEED}" \
    MOUNA_DATASET_TAG="v4canon-vs-v5canonmotion-sweep" \
    python3 -u scripts/train_transformer.py > "/tmp/mouna_v5test_seed${SEED}.log" 2>&1
  acc=$(grep -oE 'Test acc: [0-9.]+%' "/tmp/mouna_v5test_seed${SEED}.log" | tail -1 | grep -oE '[0-9.]+')
  echo "SEED=$SEED VARIANT=v5canonmotion TEST_ACC=$acc" >> "$RESULTS"
  echo "  v5canonmotion seed $SEED -> $acc%"
done

echo "V5CANONMOTION SWEEP COMPLETE"
cat "$RESULTS"
