#!/bin/bash
# Multi-seed comparison: fine-tuning FROM the self-supervised-pretrained
# encoder vs. from-scratch (random init), on the IDENTICAL labeled dataset
# and splits per seed — isolates pretrained-vs-scratch as the only variable,
# same discipline as the earlier v1/v2 data-source and z-score/shoulder-width
# normalization sweeps.
set -e
cd ~/mouna/mouna
source venv/bin/activate

RESULTS=/tmp/mouna_pretrain_sweep_results.log
> "$RESULTS"

PRETRAIN_CKPT="data/pretrained_encoder.pth"

for SEED in 42 1 2 3 4; do
  echo "=== SEED $SEED : scratch ==="
  MOUNA_SEED=$SEED \
    PYTHONPATH=src MLFLOW_ALLOW_FILE_STORE=true \
    MOUNA_PARQUET="data/silver/keypoints/keypoints.parquet" \
    MOUNA_MODEL_OUT="data/sweep/pretrain_cmp_scratch_seed${SEED}.pth" \
    MOUNA_RUN_NAME="pose-transformer-scratch-pretrainsweep-seed${SEED}" \
    MOUNA_DATASET_TAG="pretrain-vs-scratch-sweep" \
    python3 scripts/train_transformer.py > "/tmp/mouna_pretrainsweep_scratch_seed${SEED}.log" 2>&1
  acc=$(grep -oE 'Test acc: [0-9.]+%' "/tmp/mouna_pretrainsweep_scratch_seed${SEED}.log" | tail -1 | grep -oE '[0-9.]+')
  echo "SEED=$SEED VARIANT=scratch TEST_ACC=$acc" >> "$RESULTS"
  echo "  scratch seed $SEED -> $acc%"

  echo "=== SEED $SEED : pretrained ==="
  MOUNA_SEED=$SEED \
    PYTHONPATH=src MLFLOW_ALLOW_FILE_STORE=true \
    MOUNA_PARQUET="data/silver/keypoints/keypoints.parquet" \
    MOUNA_INIT_ENCODER="$PRETRAIN_CKPT" \
    MOUNA_MODEL_OUT="data/sweep/pretrain_cmp_pretrained_seed${SEED}.pth" \
    MOUNA_RUN_NAME="pose-transformer-pretrained-pretrainsweep-seed${SEED}" \
    MOUNA_DATASET_TAG="pretrain-vs-scratch-sweep" \
    python3 scripts/train_transformer.py > "/tmp/mouna_pretrainsweep_pretrained_seed${SEED}.log" 2>&1
  acc=$(grep -oE 'Test acc: [0-9.]+%' "/tmp/mouna_pretrainsweep_pretrained_seed${SEED}.log" | tail -1 | grep -oE '[0-9.]+')
  echo "SEED=$SEED VARIANT=pretrained TEST_ACC=$acc" >> "$RESULTS"
  echo "  pretrained seed $SEED -> $acc%"
done

echo "PRETRAIN SWEEP COMPLETE"
cat "$RESULTS"
