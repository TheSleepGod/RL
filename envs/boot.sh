#!/usr/bin/env bash
export PYTHONPATH=$PWD:$PYTHONPATH
set -e

python3 -m pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0

python3 -m pip install "transformers>=4.44,<4.45" "datasets>=2.19,<2.20" \
  "accelerate>=0.30,<0.31" "trl>=0.8,<0.9" "peft>=0.11,<0.12" \
  "wandb" "einops"
