import subprocess
import logging
from config import Config

logger = logging.getLogger(__name__)

def _submit_slurm_job(script_path: str) -> int:
    try:
        result = subprocess.run(
            ["sbatch", script_path],
            capture_output=True,
            text=True,
            check=True
        )

        output = result.stdout.strip()
        job_id = int(output.split()[-1])
        logger.info(f"Job submitted successfully with Job ID: {job_id}")
        return job_id
    except subprocess.CalledProcessError as e:
        logger.error(f"Error submitting SLURM job: {e.stderr}")
        return -1

def launch_worker_node():
    _submit_slurm_job(Config.WORKER_SCRIPT_PATH)