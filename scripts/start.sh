#!/bin/bash

SCRIPT_DIR=$(dirname "$0")
source "$SCRIPT_DIR/mamba_env.sh"
pip install -r "$SCRIPT_DIR/../requirements.txt"
python "$SCRIPT_DIR/../autoscaler/main.py"


