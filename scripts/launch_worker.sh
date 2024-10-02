#!/bin/bash

HEAD_NODE_IP=$(hostname -I | awk '{print $1}')
SCRIPT_DIR=$(dirname "$0")

sbatch --export=HEAD_NODE_IP=${HEAD_NODE_IP},SCRIPT_DIR=${SCRIPT_DIR} "$SCRIPT_DIR/worker.sh"
