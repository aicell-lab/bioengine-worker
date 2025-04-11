import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

from bioengine_worker.utils.logger import create_logger


class SlurmActor:
    def __init__(
        self,
        job_name: str,
        logs_dir: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.job_name = job_name

        # Directory for SLURM logs
        if logs_dir:
            self.logs_dir = Path(logs_dir).resolve()
        else:
            self.logs_dir = Path(__file__).resolve().parent.parent / "slurm_logs"

        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Set up logging
        self.logger = logger or create_logger("SlurmActor")

    def create_sbatch_script(
        self,
        command: str,
        gpus: int = 1,
        cpus_per_task: int = 8,
        mem_per_cpu: int = 8,
        time: str = "4:00:00",
        further_slurm_args: Optional[Dict[str, str]] = None,
    ) -> str:
        try:

            further_slurm_args = (
                "\n".join([f"#SBATCH {arg}" for arg in further_slurm_args if arg])
                if further_slurm_args
                else ""
            )

            # Create a temporary batch script
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".sh", delete=False
            ) as batch_file:
                batch_script = f"""
                #!/bin/bash
                #SBATCH --job-name={self.job_name}
                #SBATCH --ntasks=1
                #SBATCH --nodes=1
                #SBATCH --gpus={gpus}
                #SBATCH --cpus-per-task={cpus_per_task}
                #SBATCH --mem-per-cpu={mem_per_cpu}G
                #SBATCH --time={time}
                #SBATCH --output={self.logs_dir}/%x_%j.out
                #SBATCH --error={self.logs_dir}/%x_%j.err
                {further_slurm_args}

                # Print some diagnostic information
                echo "Host: $(hostname)"
                echo "Date: $(date)"
                echo "GPUs: {gpus}, CPUs: {cpus_per_task}"
                echo "GPU info: $(nvidia-smi -L)"
                echo "Job ID: $SLURM_JOB_ID"
                echo "Running command: {command}"
                echo ""
                echo "========================================"
                echo ""

                if [ -z "$SLURM_JOB_ID" ]; then
                    echo "SLURM_JOB_ID is not set. This script may not be running in a SLURM job."
                    exit 1
                fi

                {command}

                # Print completion status
                echo "Job completed with status $?"
                """
                for line in batch_script.split("\n"):
                    # Remove leading whitespace and write non-empty lines
                    line = line.strip()
                    if line:
                        batch_file.write(line + "\n")

                return batch_file.name
        except Exception as e:
            self.logger.error(f"Failed to create sbatch script: {e}")
            raise e

    def submit_job(self, sbatch_script: str, delete_script: bool = False) -> str:
        try:
            # Submit the job with sbatch
            result = subprocess.run(
                ["sbatch", sbatch_script], capture_output=True, text=True, check=True
            )
            self.logger.info(result.stdout.strip())

            # Clean up the temporary file
            if delete_script:
                try:
                    os.remove(sbatch_script)
                except Exception:
                    self.logger.warning(
                        f"Failed to remove temporary script: {sbatch_script}"
                    )

            # Parse job ID from Slurm output (usually "Submitted batch job 12345")
            if result.stdout and "Submitted batch job" in result.stdout:
                job_id = result.stdout.strip().split()[-1]
            else:
                raise RuntimeError(f"Failed to get job ID - stdout: {result.stdout}")

            return job_id
        except Exception as e:
            self.logger.error(f"Failed to submit job: {e}")
            raise e

    def get_jobs(self) -> Dict[str, Dict[str, str]]:
        """Query SLURM for status of all jobs.

        Returns:
            Dict mapping job_id to dict containing name, state, runtime, time_limit
        """
        try:
            # Run squeue command to get all jobs for the current user
            # Format: JobID, Name, State, Time, TimeLimit
            result = subprocess.run(
                ["squeue", "-u", os.environ["USER"], "-o", "%i %j %T %M %L"],
                capture_output=True,
                text=True,
                check=True,
            )

            # Parse the output
            lines = result.stdout.strip().split("\n")

            # Skip the header line if present
            if len(lines) > 0 and not lines[0][0].isdigit():
                lines = lines[1:]

            jobs = {}
            for line in lines:
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 5:
                        job_id, job_name, state, runtime, time_limit = (
                            parts[0],
                            parts[1],
                            parts[2],
                            parts[3],
                            parts[4],
                        )

                        # Only include Ray worker jobs
                        if job_name == self.job_name:
                            jobs[job_id] = {
                                "name": job_name,
                                "state": state,
                                "runtime": runtime,
                                "time_limit": time_limit,
                            }

            return jobs
        except Exception as e:
            self.logger.error(f"Failed to get jobs: {e}")
            raise e

    def cancel_jobs(
        self, job_ids: Optional[List[str]] = None, grace_period: Optional[int] = 30
    ) -> List[str]:
        """Cancel running jobs

        Args:
            job_ids: Specific jobs to cancel. Cancels all if None.
            grace_period: Seconds to wait until job is cancelled.

        Returns:
            List of job IDs that were cancelled.
        """
        # First get all ray worker jobs
        jobs_to_cancel = list(self.get_jobs().keys())

        if not jobs_to_cancel:
            self.logger.info("No jobs found to cancel")
            return []

        try:
            # Filter jobs if specific job_ids were provided
            if job_ids:
                jobs_to_cancel = [
                    str(job_id) for job_id in job_ids if str(job_id) in jobs_to_cancel
                ]
                if not jobs_to_cancel:
                    self.logger.warning(f"No matching jobs found to cancel: {job_ids}")
                    raise ValueError(
                        f"No matching jobs found to cancel. Provided job IDs: {job_ids} - Running jobs: {jobs_to_cancel}"
                    )

            # Cancel the jobs
            self.logger.info(
                f"Cancelling {len(jobs_to_cancel)} job(s): {jobs_to_cancel}"
            )
            subprocess.run(
                ["scancel", *jobs_to_cancel], capture_output=True, check=True
            )

            # Check if jobs were successfully cancelled
            start_time = time.time()
            while time.time() - start_time < grace_period:
                time.sleep(1)
                running_jobs = self.get_jobs()
                if not any(job_id in running_jobs for job_id in jobs_to_cancel):
                    self.logger.info(
                        f"Successfully cancelled {len(jobs_to_cancel)} job(s)"
                    )
                    return jobs_to_cancel

            cancelled_jobs = [
                job_id for job_id in jobs_to_cancel if job_id not in running_jobs
            ]
            self.logger.warning(
                f"Failed to cancel all jobs within {grace_period} seconds. Cancelled jobs: {cancelled_jobs}"
            )
            return cancelled_jobs
        except Exception as e:
            self.logger.error(f"Failed to cancel jobs: {e}")
            raise e


if __name__ == "__main__":
    # Example usage
    actor = SlurmActor(job_name="ray_worker")
    sbatch_script = actor.create_sbatch_script(
        command="sleep 30", gpus=1, cpus_per_task=1, mem_per_cpu=1, time="00:10:00"
    )
    job_id = actor.submit_job(sbatch_script, delete_script=False)
    job_id = actor.submit_job(sbatch_script, delete_script=False)
    job_id = actor.submit_job(sbatch_script, delete_script=True)

    running_jobs = []
    while len(running_jobs) < 3:
        time.sleep(1)
        running_jobs = actor.get_jobs()

    actor.cancel_jobs(grace_period=10)
