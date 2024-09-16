#!/bin/bash
#SBATCH --job-name=ray_worker
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=00:00:20 		# Tweak time if necessary
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --output=$HOME/logs/%x_%j.out
#SBATCH --error=$HOME/logs/%x_%j.err

# IP of Ray head node
head_node_ip=10.81.254.11
ray_port=6379

# "ray[all]" via pip requires cpp dependencies 
module load buildenv-gcccuda/12.1.1-gcc12.3.0

# Make sure to create a virtual environment with packages from "requirements.txt"
source $HOME/myenv/bin/activate

# Same number of cpus and gpus as specified for SBATCH
ray start --address=${head_node_ip}:${ray_port} --num-cpus=8 --num-gpus=1 --block
