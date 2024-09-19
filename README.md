# autoscaler

Autoscaler for Ray on SLURM as a separate module.

## About

This script has hardcoded values as its main purpose is to be run on the Berzelius compute cluster as a component of a larger Ray system. It assumes that there is already a running Ray head node process running on the same machine.

### SBatch Script
[worker.sh](scripts/worker.sh) contains a sbatch script that is launched on worker nodes. When it's run on a node it connects it to the head node via a hardcoded IP address. On Berzelius there are 3 different login nodes with different addresses; `10.81.254.10` (l0), `10.81.254.11` (l1), `10.81.254.12` (l2). Depending on what login node the Ray head is started on the address may need to be changed.

