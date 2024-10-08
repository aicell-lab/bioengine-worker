#!/bin/bash

HEAD_NODE_IP=$(hostname -I | awk '{print $1}')
SCRIPT_DIR=$(dirname "$0")

source "$SCRIPT_DIR/mamba_env.sh"
pip install -r "$SCRIPT_DIR/../requirements.txt"

ray stop
ray start --head --node-ip-address=${HEAD_NODE_IP} --port=6379 --num-cpus=0 --num-gpus=0

cleanup() {
    echo "Stopping autoscaler..."
    ray stop
    kill $MAIN_PID
    wait $MAIN_PID 
    exit 0
}
trap cleanup SIGINT SIGTERM

python "$SCRIPT_DIR/../autoscaler/main.py" "$HEAD_NODE_IP"
MAIN_PID=$!
wait $MAIN_PID


