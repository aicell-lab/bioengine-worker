# autoscaler
Autoscaler for Ray on SLURM. It has automated scripts for deployment of a Ray cluster and the exposure of Hypha services. The autoscaler dynamically adjusts resources based on workload, helping to optimize the use of computational resources.<br>

The [start.sh](scripts/start.sh) script starts a Ray head node and python [script](autoscaler/main.py). It's used to expose [hypha-services](autoscaler/hypha/service.py) running on an HPC with [SLURM](https://slurm.schedmd.com/documentation.html). Upon start it may prompt for token verification via link.

## How-to Run

1. Clone this repository to your local machine:
   ```bash
   git clone git@github.com:aicell-lab/autoscaler.git
   cd autoscaler
   ```

2. Run [start.sh](scripts/start.sh). 


## Build Environment and Dependencies
This project is designed to run on a **Debian-based Linux** distribution and is intended for use in high-performance computing (HPC) environments. Users should consider their specific HPC configurations, which may involve loading modules and adjusting settings according to their cluster's requirements.

Most dependencies and environment configurations are managed through the [mamba_env.sh](scripts/mamba_env.sh) script. It is recommended to consult your HPC's documentation for guidance on setting up the environment correctly, including any necessary software installations and module management practices.

## Workflow

### Initialization
- Start by running [start.sh](scripts/start.sh) in a SLURM environment.
- Hypha service(s) are registered and a ray head node is spawned without any worker nodes attached.
- Upon receiving requests via Hypha, code is submitted as a ray [task](https://docs.ray.io/en/latest/ray-core/tasks.html).
- The head node is checked continuously for [status updates](https://docs.ray.io/en/latest/ray-observability/user-guides/cli-sdk.html#state-api-overview-ref) regarding tasks and workers. 

### Upscaling
- If there are more pending tasks than worker nodes, more worker nodes are spawned via SLURM jobs (uptime: 10 minutes per job). 
- The number of workers is limited by the `MAX_WORKER` config setting.
- As soon as there is an idle worker node connected Ray handles task assignment automatically.
- Hypha retrieves the Ray task result and returns it.

### Downscaling
- If there are no pending tasks then the number of worker nodes is reduced by terminating active SLURM jobs acting as worker nodes.
- `MIN_WORKER` currently limits how many worker nodes are terminated.