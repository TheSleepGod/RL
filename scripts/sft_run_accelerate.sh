#!/usr/bin/env bash
export PYTHONPATH=$PWD:$PYTHONPATH
set -e

CFG=${1:-configs/sft_qwen2_7b.yaml}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

accelerate launch --config_file configs/accelerate.yaml \
  scripts/train_sft_lora.py \
  --config "$CFG" \
  --set train.num_train_epochs=2 train.learning_rate=2e-4
