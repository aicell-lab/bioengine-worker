#!/bin/bash

ENV_NAME="ray_env"

module load Mambaforge/23.3.1-1-hpc1-bdist 
module load buildenv-gcccuda/12.1.1-gcc12.3.0

conda info --envs | grep -q "^${ENV_NAME} " || mamba create --name ${ENV_NAME} python=3.9
mamba activate ${ENV_NAME}


