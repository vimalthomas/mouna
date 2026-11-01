#!/bin/bash
# v2norm (per-part normalization only) vs v3motion (+ orientation/velocity
# features), 5 seeds each — same paired-test discipline as every other
# comparison today.
set -e
cd ~/mouna/mouna
source venv/bin/activate

RESULTS=/tmp/mouna_v2v3_sweep_results.log
> "$RESULTS"

for SEED in 42 1 2 3 4; do
  echo "=== SEED $SEED : v2norm ==="
  MOUNA_SEED=$SEED \
    PYTHONPATH=src MLFLOW_ALLOW_FILE_STORE=true \
    MOUNA_PARQUET="data/silver/keypoints/keypoints_official_split_v2norm.parquet" \
    MOUNA_MODEL_OUT="data/sweep/v2norm_seed${SEED}.pth" \
    MOUNA_RUN_NAME="pose-transformer-v2norm-v2v3sweep-seed${SEED}" \
    MOUNA_DATASET_TAG="v2norm-vs-v3motion-sweep" \
    python3 -u scripts/train_transformer.py > "/tmp/mouna_v2v3sweep_v2norm_seed${SEED}.log" 2>&1
  acc=$(grep -oE 'Test acc: [0-9.]+%' "/tmp/mouna_v2v3sweep_v2norm_seed${SEED}.log" | tail -1 | grep -oE '[0-9.]+')
  echo "SEED=$SEED VARIANT=v2norm TEST_ACC=$acc" >> "$RESULTS"
  echo "  v2norm seed $SEED -> $acc%"

  echo "=== SEED $SEED : v3motion ==="
  MOUNA_SEED=$SEED \
    PYTHONPATH=src MLFLOW_ALLOW_FILE_STORE=true \
    MOUNA_PARQUET="data/silver/keypoints/keypoints_official_split_v3motion.parquet" \
    MOUNA_MODEL_OUT="data/sweep/v3motion_seed${SEED}.pth" \
    MOUNA_RUN_NAME="pose-transformer-v3motion-v2v3sweep-seed${SEED}" \
    MOUNA_DATASET_TAG="v2norm-vs-v3motion-sweep" \
    python3 -u scripts/train_transformer.py > "/tmp/mouna_v2v3sweep_v3motion_seed${SEED}.log" 2>&1
  acc=$(grep -oE 'Test acc: [0-9.]+%' "/tmp/mouna_v2v3sweep_v3motion_seed${SEED}.log" | tail -1 | grep -oE '[0-9.]+')
  echo "SEED=$SEED VARIANT=v3motion TEST_ACC=$acc" >> "$RESULTS"
  echo "  v3motion seed $SEED -> $acc%"
done

echo "V2V3 SWEEP COMPLETE"
cat "$RESULTS"
