import asyncio
import logging
import os
import subprocess
import tempfile
import time
from typing import Dict, List, Optional

from bioengine_worker.utils import create_logger


class SlurmManager:
    def __init__(
        self,
        job_name: str,
        slurm_log_dir: str,
        log_file: Optional[str] = None,
        debug: bool = False,
    ):
        # Set up logging
        self.logger = create_logger(
            name="SlurmManager",
            level=logging.DEBUG if debug else logging.INFO,
            log_file=log_file,
        )

        # Set SLURM job name and log directory
        self.job_name = job_name
        self.slurm_log_dir = slurm_log_dir

    async def create_sbatch_script(
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

            def create_temp_script():
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
                    #SBATCH --chdir=/home/{os.environ['USER']}
                    #SBATCH --output={self.slurm_log_dir}/%x_%j.out
                    #SBATCH --error={self.slurm_log_dir}/%x_%j.err
                    {further_slurm_args}

                    # Print some diagnostic information
                    echo "Host: $(hostname)"
                    echo "Date: $(date)"
                    echo "GPUs: {gpus}, CPUs: {cpus_per_task}"
                    echo "GPU info: $(nvidia-smi -L)"
                    echo "Job ID: $SLURM_JOB_ID"
                    echo "Working directory: $(pwd)"
                    echo "Running command: {command}"
                    echo ""
                    echo "========================================"
                    echo ""

                    if [ -z "$SLURM_JOB_ID" ]; then
                        echo "SLURM_JOB_ID is not set. This script may not be running in a SLURM job."
                        exit 1
                    fi

                    # Reset bound in paths in case of submission from a container
                    APPTAINER_BIND=
                    SINGULARITY_BIND=

                    {command}

                    # Print completion status
                    echo "Job completed with status $?"
                    """
                    for line in batch_script.split("\n"):
                        # Remove leading whitespace and write non-empty lines
                        line = line.strip()
                        if line:
                            batch_file.write(line + "\n")

                    self.logger.debug(
                        f"Created sbatch script '{batch_file.name}':\n{batch_script}"
                    )

                    return batch_file.name

            return await asyncio.to_thread(create_temp_script)
        except Exception as e:
            self.logger.error(f"Failed to create sbatch script: {e}")
            raise e

    async def submit_job(self, sbatch_script: str, delete_script: bool = False) -> str:
        try:
            # Submit the job with sbatch
            proc = await asyncio.create_subprocess_exec(
                "sbatch", sbatch_script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            
            if proc.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                raise subprocess.CalledProcessError(proc.returncode, "sbatch", stderr=error_msg)
            
            output = stdout.decode().strip()
            self.logger.info(output)

            # Clean up the temporary file
            if delete_script:
                try:
                    await asyncio.to_thread(os.remove, sbatch_script)
                except Exception:
                    self.logger.warning(
                        f"Failed to remove temporary script: {sbatch_script}"
                    )

            # Parse job ID from SlurmManager output (usually "Submitted batch job 12345")
            if output and "Submitted batch job" in output:
                job_id = output.split()[-1]
            else:
                raise RuntimeError(f"Failed to get job ID - stdout: {output}")

            return job_id
        except Exception as e:
            self.logger.error(f"Failed to submit job: {e}")
            raise e

    async def get_jobs(self) -> Dict[str, Dict[str, str]]:
        """Query SLURM for status of all jobs.

        Returns:
            Dict mapping job_id to dict containing name, state, runtime, time_limit
        """
        try:
            # Run squeue command to get all jobs for the current user
            # Format: JobID, Name, State, Time, TimeLimit
            proc = await asyncio.create_subprocess_exec(
                "squeue", "-u", os.environ["USER"], "-o", "%i %j %T %M %L",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            
            if proc.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                raise subprocess.CalledProcessError(proc.returncode, "squeue", stderr=error_msg)

            # Parse the output
            lines = stdout.decode().strip().split("\n")

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

    async def cancel_jobs(
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
        jobs_to_cancel = list((await self.get_jobs()).keys())

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
            proc = await asyncio.create_subprocess_exec(
                "scancel", *jobs_to_cancel,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            
            if proc.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                raise subprocess.CalledProcessError(proc.returncode, "scancel", stderr=error_msg)

            # Check if jobs were successfully cancelled
            start_time = time.time()
            while time.time() - start_time < grace_period:
                await asyncio.sleep(1)
                running_jobs = await self.get_jobs()
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
    import asyncio
    
    async def main():
        # Example usage
        actor = SlurmManager(job_name="ray_worker", slurm_log_dir="/tmp")
        sbatch_script = await actor.create_sbatch_script(
            command="sleep 30", gpus=1, cpus_per_task=1, mem_per_cpu=1, time="00:10:00"
        )
        job_id = await actor.submit_job(sbatch_script, delete_script=False)
        job_id = await actor.submit_job(sbatch_script, delete_script=False)
        job_id = await actor.submit_job(sbatch_script, delete_script=True)

        running_jobs = []
        while len(running_jobs) < 3:
            await asyncio.sleep(1)
            running_jobs = await actor.get_jobs()

        await actor.cancel_jobs(grace_period=10)

    asyncio.run(main())
