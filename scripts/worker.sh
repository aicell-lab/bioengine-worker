#!/bin/bash
#SBATCH --job-name=ray_worker
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=00:10:00 		# Tweak time if necessary
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.err

# $HEAD_NODE_IP exported from launch_worker.sh
# $SCRIPT_DIR exported from launch_worker.sh
RAY_PORT=6379
source "$SCRIPT_DIR/mamba_env.sh"

# Same number of cpus and gpus as specified for SBATCH
ray start --address=${HEAD_NODE_IP}:${RAY_PORT} --num-cpus=8 --num-gpus=1 --block
