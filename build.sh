#!/usr/bin/env bash
# Exit on error
set -o errexit

pip install -r requirements.txt
python train_model.py
python test_model.py
