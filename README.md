# autoscaler
Autoscaler for Ray on SLURM as a separate module.

## About
This script has hardcoded values as its main purpose is to be run on the Berzelius compute cluster as a component of a larger Ray system. It assumes that there is already a running Ray head node process running on the same machine.

## SBatch Script
[worker.sh](scripts/worker.sh) contains a sbatch script that is launched on worker nodes. When it's run on a node it connects it to the head node via a hardcoded IP address. On Berzelius there are 3 different login nodes with different addresses; `10.81.254.10` (l0), `10.81.254.11` (l1), `10.81.254.12` (l2). Depending on what login node the Ray head is started on the address may need to be changed.

## How-to Run
SSH into Berzelius and start two [tmux](https://github.com/tmux/tmux/wiki) sessions. Launch a Ray head node server in the first session. Once the Ray server is up run [main.py](autoscaler/main.py) in the second session.

## Virtual Environment (Python)
Ray and other [dependecies](requirements.txt) are required to run the autoscaler and Ray workers. [worker.sh](scripts/worker.sh) starts a virtual environment with a hardcoded path before connecting to the head node.
