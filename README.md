# autoscaler
Autoscaler for Ray on SLURM. Has automated scripts for deployment of Ray cluster and exposing Hypha services. <br>

The [start.sh](scripts/start.sh) script starts a Ray head node and python [script](autoscaler/main.py). It's used to expose hypha-[services](autoscaler/hypha/service.py) running on a HPC. Upon start it will prompt for token verification via link.

## How-to Run
SSH into [Berzelius](https://www.nsc.liu.se/systems/berzelius/), copy-paste this repo, and start a [tmux](https://github.com/tmux/tmux/wiki) session. Run [start.sh](scripts/start.sh). 

