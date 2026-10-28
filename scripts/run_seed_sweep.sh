#!/bin/bash
# Multi-seed v1 vs v2 comparison: each seed varies both weight init and the
# train/val/test split simultaneously (same SEED drives both), but v1 and v2
# still share an identical test set per seed since v2's clean-eval-pool split
# draws from the same underlying content as v1's full df. This gives N paired
# samples of "does v2 beat v1" instead of relying on one noisy run.
set -e
cd ~/mouna/mouna
source venv/bin/activate

RESULTS=/tmp/mouna_seed_sweep_results.log
> "$RESULTS"

for SEED in 1 2 3 4; do
  echo "=== SEED $SEED : v1 ==="
  MOUNA_SEED=$SEED \
    PYTHONPATH=src MLFLOW_ALLOW_FILE_STORE=true \
    MOUNA_PARQUET="data/silver/keypoints/keypoints.parquet" \
    MOUNA_MODEL_OUT="data/sweep/v1_seed${SEED}.pth" \
    MOUNA_RUN_NAME="pose-transformer-v1-seed${SEED}" \
    MOUNA_DATASET_TAG="WLASL+MS-ASL-seedsweep" \
    python3 scripts/train_transformer.py > "/tmp/mouna_sweep_v1_seed${SEED}.log" 2>&1
  acc=$(grep -oE 'Test acc: [0-9.]+%' "/tmp/mouna_sweep_v1_seed${SEED}.log" | tail -1 | grep -oE '[0-9.]+')
  echo "SEED=$SEED VARIANT=v1 TEST_ACC=$acc" >> "$RESULTS"
  echo "  v1 seed $SEED -> $acc%"

  echo "=== SEED $SEED : v2 ==="
  MOUNA_SEED=$SEED \
    PYTHONPATH=src MLFLOW_ALLOW_FILE_STORE=true \
    MOUNA_PARQUET="data/silver/keypoints/keypoints_v2_merged.parquet" \
    MOUNA_MODEL_OUT="data/sweep/v2_seed${SEED}.pth" \
    MOUNA_RUN_NAME="pose-transformer-v2-seed${SEED}" \
    MOUNA_DATASET_TAG="WLASL+MS-ASL+OpenPose-seedsweep" \
    python3 scripts/train_transformer.py > "/tmp/mouna_sweep_v2_seed${SEED}.log" 2>&1
  acc=$(grep -oE 'Test acc: [0-9.]+%' "/tmp/mouna_sweep_v2_seed${SEED}.log" | tail -1 | grep -oE '[0-9.]+')
  echo "SEED=$SEED VARIANT=v2 TEST_ACC=$acc" >> "$RESULTS"
  echo "  v2 seed $SEED -> $acc%"
done

echo "SWEEP COMPLETE"
cat "$RESULTS"
