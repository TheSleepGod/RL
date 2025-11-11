#!/usr/bin/env bash
export PYTHONPATH=$PWD:$PYTHONPATH
set -e

python3 -m pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0

python3 -m pip install -r requirement.txt

