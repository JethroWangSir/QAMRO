#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

EXP=l_pa_pamr_7.0_inverse_120
INPUT_CSV="../audiomos2025_track2/audiomos2025-track2-dev_list_filtered.csv"
OUTPUT_CSV="../prediction/"$EXP"/predictions.csv"
METRICS_CSV="../prediction/"$EXP"/metrics.csv"
CKPT="../ckpt/"$EXP"/best_model_36.pt"

mkdir -p "$(dirname "$OUTPUT_CSV")"
mkdir -p "$(dirname "$METRICS_CSV")"

python predict.py \
  "$INPUT_CSV" \
  "$OUTPUT_CSV" \
  "$METRICS_CSV" \
  --ckpt "$CKPT" \
