# autoscaler
Autoscaler for Ray on SLURM. Has automated scripts for deployment of Ray cluster and exposing Hypha services. The autoscaler dynamically adjusts resources based on workload, helping to optimize the use of computational resources.<br>

The [start.sh](scripts/start.sh) script starts a Ray head node and python [script](autoscaler/main.py). It's used to expose [hypha-services](autoscaler/hypha/service.py) running on a HPC with [SLURM](https://slurm.schedmd.com/documentation.html). Upon start it may prompt for token verification via link.

## How-to Run

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/aicell-lab/autoscaler
   cd autoscaler
   ```

2. Run [start.sh](scripts/start.sh). 


## Build Environment and Dependencies
This project is designed to run on a **Debian-based Linux** distribution and is intended for use in high-performance computing (HPC) environments. Users should consider their specific HPC configurations, which may involve loading modules and adjusting settings according to their cluster's requirements.

Most dependencies and environment configurations are managed through the [mamba_env.sh](scripts/mamba_env.sh) script. It is recommended to consult your HPC's documentation for guidance on setting up the environment correctly, including any necessary software installations and module management practices.
