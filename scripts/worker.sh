#!/bin/bash
#SBATCH --job-name=ray_worker
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=00:00:20 		# Tweak time if necessary
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --output=$HOME/logs/%x_%j.out   # Ensure logs directory exists before running this
#SBATCH --error=$HOME/logs/%x_%j.err

# IP of Ray head node
HEAD_NODE_IP=10.81.254.11   # The IP may be changed by the autoscaler script via REGEX find-replace
RAY_PORT=6379
ENV_NAME="ray_env"

module load Mambaforge/23.3.1-1-hpc1-bdist 
module load buildenv-gcccuda/12.1.1-gcc12.3.0
mamba activate ${ENV_NAME}

# Same number of cpus and gpus as specified for SBATCH
ray start --address=${HEAD_NODE_IP}:${RAY_PORT} --num-cpus=8 --num-gpus=1 --block
