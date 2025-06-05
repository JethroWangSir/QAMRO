#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

TRAIN_CSV="../audiomos2025_track2/audiomos2025-track2-train_list_filtered.csv"
DEV_CSV="../audiomos2025_track2/audiomos2025-track2-dev_list_filtered.csv"
EXPNAME="../ckpt/l_pa_pamr_7.0_inverse_120"
EPOCHS=1000
TRAIN_BATCH_SIZE=120
EVAL_BATCH_SIZE=8
LR=1e-5
FREEZE_ENCODER=1
RANKING_LOSS_WEIGHT=1.0
MARGIN_SCALE=0.2
PREFERENCE_FACTOR=7.0

python mos_track2.py \
  --train_csv  "$TRAIN_CSV" \
  --dev_csv    "$DEV_CSV"   \
  --exp_name   "$EXPNAME"       \
  --epochs     $EPOCHS      \
  --train_batch_size $TRAIN_BATCH_SIZE       \
  --eval_batch_size  $EVAL_BATCH_SIZE       \
  --lr         $LR          \
  --freeze_encoder  $FREEZE_ENCODER \
  --ranking_loss_weight  $RANKING_LOSS_WEIGHT \
  --margin_scale  $MARGIN_SCALE \
  --preference_factor  $PREFERENCE_FACTOR \
