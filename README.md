# autoscaler
Autoscaler for Ray on SLURM. Also has automated scripts for deployment of Ray cluster and exposing Hypha services.

## How-to Run
SSH into Berzelius, copy-paste this repo, and start a [tmux](https://github.com/tmux/tmux/wiki) session. Run [start.sh](scripts/start.sh). <br><br>
The start.sh script starts the Ray head node server as well as two parallel python processes. The python processes are the [autoscaler](autoscaler/main.py) for the Ray server and [services](hypha/service.py) exposed via Hypha. When the Hypha script runs it will prompt for verification via link. Make sure to click on that link; it can be easy to miss.

