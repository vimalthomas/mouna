#!/bin/bash
# Multi-seed comparison: current z-score normalization vs. shoulder-width +
# wrist-relative normalization, on IDENTICAL raw keypoints — isolates the
# normalization scheme as the only variable, same discipline as the v1/v2
# data-source sweep.
set -e
cd ~/mouna/mouna
source venv/bin/activate

RESULTS=/tmp/mouna_norm_sweep_results.log
> "$RESULTS"

for SEED in 42 1 2 3 4; do
  echo "=== SEED $SEED : zscore ==="
  MOUNA_SEED=$SEED \
    PYTHONPATH=src MLFLOW_ALLOW_FILE_STORE=true \
    MOUNA_PARQUET="data/silver/keypoints/keypoints_norm_zscore.parquet" \
    MOUNA_MODEL_OUT="data/sweep/norm_zscore_seed${SEED}.pth" \
    MOUNA_RUN_NAME="pose-transformer-norm-zscore-seed${SEED}" \
    MOUNA_DATASET_TAG="norm-zscore-sweep" \
    python3 scripts/train_transformer.py > "/tmp/mouna_normsweep_zscore_seed${SEED}.log" 2>&1
  acc=$(grep -oE 'Test acc: [0-9.]+%' "/tmp/mouna_normsweep_zscore_seed${SEED}.log" | tail -1 | grep -oE '[0-9.]+')
  echo "SEED=$SEED VARIANT=zscore TEST_ACC=$acc" >> "$RESULTS"
  echo "  zscore seed $SEED -> $acc%"

  echo "=== SEED $SEED : shoulder ==="
  MOUNA_SEED=$SEED \
    PYTHONPATH=src MLFLOW_ALLOW_FILE_STORE=true \
    MOUNA_PARQUET="data/silver/keypoints/keypoints_norm_shoulder.parquet" \
    MOUNA_MODEL_OUT="data/sweep/norm_shoulder_seed${SEED}.pth" \
    MOUNA_RUN_NAME="pose-transformer-norm-shoulder-seed${SEED}" \
    MOUNA_DATASET_TAG="norm-shoulder-sweep" \
    python3 scripts/train_transformer.py > "/tmp/mouna_normsweep_shoulder_seed${SEED}.log" 2>&1
  acc=$(grep -oE 'Test acc: [0-9.]+%' "/tmp/mouna_normsweep_shoulder_seed${SEED}.log" | tail -1 | grep -oE '[0-9.]+')
  echo "SEED=$SEED VARIANT=shoulder TEST_ACC=$acc" >> "$RESULTS"
  echo "  shoulder seed $SEED -> $acc%"
done

echo "NORM SWEEP COMPLETE"
cat "$RESULTS"
