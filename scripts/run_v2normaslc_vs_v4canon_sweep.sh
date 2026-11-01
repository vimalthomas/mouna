#!/bin/bash
# v2norm+ASL-Citizen (baseline) vs v4canon (same data, handedness-canonicalized
# at preprocessing time, flip augmentation disabled since it's now redundant
# and was buggy anyway) -- 5 seeds each, same paired-test discipline as every
# other comparison this project has made.
set -e
cd ~/mouna/mouna
source venv/bin/activate

RESULTS=/tmp/mouna_v2normaslc_v4canon_results.log
> "$RESULTS"

for SEED in 42 1 2 3 4; do
  echo "=== SEED $SEED : v2norm_aslc ==="
  MOUNA_SEED=$SEED \
    PYTHONPATH=src MLFLOW_ALLOW_FILE_STORE=true \
    MOUNA_PARQUET="data/silver/keypoints/keypoints_official_split_v2norm_aslc.parquet" \
    MOUNA_MODEL_OUT="data/sweep/v2normaslc_seed${SEED}.pth" \
    MOUNA_RUN_NAME="pose-transformer-v2normaslc-canonsweep-seed${SEED}" \
    MOUNA_DATASET_TAG="v2normaslc-vs-v4canon-sweep" \
    python3 -u scripts/train_transformer.py > "/tmp/mouna_canonsweep_v2normaslc_seed${SEED}.log" 2>&1
  acc=$(grep -oE 'Test acc: [0-9.]+%' "/tmp/mouna_canonsweep_v2normaslc_seed${SEED}.log" | tail -1 | grep -oE '[0-9.]+')
  echo "SEED=$SEED VARIANT=v2normaslc TEST_ACC=$acc" >> "$RESULTS"
  echo "  v2norm_aslc seed $SEED -> $acc%"

  echo "=== SEED $SEED : v4canon ==="
  MOUNA_SEED=$SEED \
    PYTHONPATH=src MLFLOW_ALLOW_FILE_STORE=true \
    MOUNA_DISABLE_FLIP_AUG=1 \
    MOUNA_PARQUET="data/silver/keypoints/keypoints_official_split_v4canon.parquet" \
    MOUNA_MODEL_OUT="data/sweep/v4canon_seed${SEED}.pth" \
    MOUNA_RUN_NAME="pose-transformer-v4canon-canonsweep-seed${SEED}" \
    MOUNA_DATASET_TAG="v2normaslc-vs-v4canon-sweep" \
    python3 -u scripts/train_transformer.py > "/tmp/mouna_canonsweep_v4canon_seed${SEED}.log" 2>&1
  acc=$(grep -oE 'Test acc: [0-9.]+%' "/tmp/mouna_canonsweep_v4canon_seed${SEED}.log" | tail -1 | grep -oE '[0-9.]+')
  echo "SEED=$SEED VARIANT=v4canon TEST_ACC=$acc" >> "$RESULTS"
  echo "  v4canon seed $SEED -> $acc%"
done

echo "V2NORMASLC_VS_V4CANON SWEEP COMPLETE"
cat "$RESULTS"
