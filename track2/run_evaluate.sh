#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

EXP=full
OUTPUT_CSV="../evaluation/"$EXP"/answer.txt"
CKPT="../ckpt/"$EXP"/last_model.pt"

mkdir -p "$(dirname "$OUTPUT_CSV")"

python evaluate.py \
  --eval_list /share/nas169/jethrowang/AudioMOS/track2/audiomos2025-track2-eval-phase/DATA/sets/eval_list.txt \
  --wav_dir /share/nas169/jethrowang/AudioMOS/track2/audiomos2025-track2-eval-phase/DATA/wav \
  --output_csv $OUTPUT_CSV \
  --ckpt $CKPT \
  --batch_size 1 \

