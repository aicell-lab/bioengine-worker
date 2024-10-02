#!/bin/bash

HEAD_NODE_IP=$(hostname -I | awk '{print $1}')
SCRIPT_DIR=$(dirname "$0")

source "$SCRIPT_DIR/mamba_env.sh"
source "$SCRIPT_DIR/mamba_packages.sh"
source "$SCRIPT_DIR/token.sh"

ray stop
ray start --head --node-ip-address=${HEAD_NODE_IP} --port=6379 --num-cpus=0 --num-gpus=0

cleanup() {
    echo "Stopping all child processes..."
    ray stop
    kill $MAIN_PID $SERVICE_PID
    wait $MAIN_PID $SERVICE_PID 
    exit 0
}
trap cleanup SIGINT SIGTERM


python "$SCRIPT_DIR/../autoscaler/main.py" &
AUTOSCALER_PID=$!
python "$SCRIPT_DIR/../hypha/service.py" &
SERVICE_PID=$!

wait $MAIN_PID $SERVICE_PID

