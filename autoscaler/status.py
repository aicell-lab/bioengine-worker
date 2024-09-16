import subprocess
import logging

logger = logging.getLogger(__name__)

## SLURM STATUS
def get_slurm_jobs_by_state(state: str) -> int:
    try:
        result = subprocess.run(
            ["squeue", "-u", "$USER", f"--state={state}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        num_jobs = len(result.stdout.decode().strip().split("\n")) - 1
        logger.info(f"Number of SLURM {state.lower()} jobs: {num_jobs}")
        return num_jobs
    except subprocess.CalledProcessError as e:
        logger.error(f"Error getting SLURM {state.lower()} jobs: {e}")
        return 0
    except FileNotFoundError:
        logger.error("The 'squeue' command was not found. Ensure SLURM is installed and accessible.")
        return 0
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return 0

def get_num_slurm_pending_jobs() -> int:
    return get_slurm_jobs_by_state("PENDING")
def get_num_slurm_running_jobs() -> int:
    return get_slurm_jobs_by_state("RUNNING")
def get_num_slurm_jobs() -> int:
    return get_num_slurm_pending_jobs() + get_num_slurm_running_jobs()


## RAY CLUSTER STATUS
def get_ray_status() -> str:
    try:
        result = subprocess.run(
            ["ray", "status"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        return result.stdout.decode()
    except subprocess.CalledProcessError as e:
        logger.error(f"Error getting Ray status: {e}")
        return ""
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return ""
def get_num_ray_pending_jobs() -> int:
    status_output = get_ray_status()
    if "Pending:" in status_output:
        pending_section = status_output.split("Pending:")[1].split("\n")[0].strip()
        if pending_section == "(no pending nodes)":
            return 0
        else:
            return len(pending_section.split("\n"))
    return 0
def get_num_ray_running_jobs() -> int:
    status_output = get_ray_status()
    if "Healthy:" in status_output:
        healthy_section = status_output.split("Healthy:")[1].split("\n")[0].strip()
        num_active_workers = len(healthy_section.split("\n"))
        return max(0,num_active_workers - 1)  # Subtract 1 to exclude login head-node
    return 0

def get_num_ray_jobs() -> int:
    return get_num_ray_pending_jobs() + get_num_ray_running_jobs()

