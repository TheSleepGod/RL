#!/usr/bin/env bash
export PYTHONPATH=$PWD:$PYTHONPATH
set -e

CFG=${1:-configs/ppo_qwen2_7b.yaml}

echo "======================================================"
echo "Starting PPO training with config: $CFG"
echo "Extra arguments: $@"
echo "======================================================"

accelerate launch --config_file configs/accelerate.yaml \
  rl/train_ppo.py \
  --config "$CFG" \
  "$@"
