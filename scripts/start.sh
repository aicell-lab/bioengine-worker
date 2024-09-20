#!/bin/bash

module load Mambaforge/23.3.1-1-hpc1-bdist 
module load buildenv-gcccuda/12.1.1-gcc12.3.0

ENV_NAME="ray_env"
HEAD_NODE_IP=$(hostname -I | awk '{print $1}')

conda info --envs | grep -q "^${ENV_NAME} " || mamba create --name ${ENV_NAME} python=3.9
mamba activate ${ENV_NAME}
pip install "ray[all]"
pip install hypha-rpc

ray stop
ray start --head --node-ip-address=${HEAD_NODE_IP} --port=6379 --num-cpus=0 --num-gpus=0



