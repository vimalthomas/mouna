#!/bin/bash
# v1 (WLASL + old unverified MS-ASL, 1820 seq) vs v4 (WLASL + newly-sourced
# official MS-ASL, 1947 seq) — same multi-seed paired discipline as every
# other comparison today.
set -e
cd ~/mouna/mouna
source venv/bin/activate

RESULTS=/tmp/mouna_v4_sweep_results.log
> "$RESULTS"

for SEED in 42 1 2 3 4; do
  echo "=== SEED $SEED : v1 ==="
  MOUNA_SEED=$SEED \
    PYTHONPATH=src MLFLOW_ALLOW_FILE_STORE=true \
    MOUNA_PARQUET="data/silver/keypoints/keypoints.parquet" \
    MOUNA_MODEL_OUT="data/sweep/v4cmp_v1_seed${SEED}.pth" \
    MOUNA_RUN_NAME="pose-transformer-v1-v4sweep-seed${SEED}" \
    MOUNA_DATASET_TAG="v1-vs-v4-sweep" \
    python3 scripts/train_transformer.py > "/tmp/mouna_v4sweep_v1_seed${SEED}.log" 2>&1
  acc=$(grep -oE 'Test acc: [0-9.]+%' "/tmp/mouna_v4sweep_v1_seed${SEED}.log" | tail -1 | grep -oE '[0-9.]+')
  echo "SEED=$SEED VARIANT=v1 TEST_ACC=$acc" >> "$RESULTS"
  echo "  v1 seed $SEED -> $acc%"

  echo "=== SEED $SEED : v4 ==="
  MOUNA_SEED=$SEED \
    PYTHONPATH=src MLFLOW_ALLOW_FILE_STORE=true \
    MOUNA_PARQUET="data/silver/keypoints/keypoints_v4_msasl_official.parquet" \
    MOUNA_MODEL_OUT="data/sweep/v4cmp_v4_seed${SEED}.pth" \
    MOUNA_RUN_NAME="pose-transformer-v4-v4sweep-seed${SEED}" \
    MOUNA_DATASET_TAG="v1-vs-v4-sweep" \
    python3 scripts/train_transformer.py > "/tmp/mouna_v4sweep_v4_seed${SEED}.log" 2>&1
  acc=$(grep -oE 'Test acc: [0-9.]+%' "/tmp/mouna_v4sweep_v4_seed${SEED}.log" | tail -1 | grep -oE '[0-9.]+')
  echo "SEED=$SEED VARIANT=v4 TEST_ACC=$acc" >> "$RESULTS"
  echo "  v4 seed $SEED -> $acc%"
done

echo "V4 SWEEP COMPLETE"
cat "$RESULTS"
