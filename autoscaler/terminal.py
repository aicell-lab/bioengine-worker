import subprocess
import logging
from typing import List
from config import Config

def run_command(args: List[str]) -> str:
    try:
        result = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        return result.stdout.decode().strip()
    except subprocess.CalledProcessError as e:
        logging.error(f"Process error: {e.stderr}, args: {args}")
    except FileNotFoundError:
        logging.error("The 'squeue' command was not found. Ensure SLURM is installed and accessible.")
    except Exception as e:
        logging.error(f"Unkown error: {e}")

def _get_worker_launch_args() -> List[str]:
    #sbatch --export=HEAD_NODE_IP=${HEAD_NODE_IP},SCRIPT_DIR=${SCRIPT_DIR} "$SCRIPT_DIR/worker.sh"
    env_vars = f"--export=HEAD_NODE_IP={Config.Head.ip},SCRIPT_DIR={Config.Shell.script_directory_path}"
    return ["sbatch", env_vars, Config.Shell.worker_script_path]

def launch_worker_node() -> int:
    try:
        output = run_command(args=_get_worker_launch_args())
        if output is None:
            logging.error("Failed to launch worker script!")
            return -1
        else:
            job_id = int(output.split()[-1])
            logging.info(f"Job submitted with ID: {job_id}")
            return job_id
    except subprocess.CalledProcessError as e:
        logging.error(f"Error submitting SLURM job: {e.stderr}")
    return -1