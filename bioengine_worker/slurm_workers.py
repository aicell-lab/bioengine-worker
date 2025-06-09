import asyncio
import logging
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Literal, Optional

import ray
from ray.util.state import get_node
from ray import serve

from bioengine_worker import __version__
from bioengine_worker.ray_autoscaler import RayAutoscaler
from bioengine_worker.utils import create_logger, format_time, stream_logging_format


class SlurmWorkers:
    """
    A class to manage Slurm workers for Bioengine.
    This class is responsible for handling the Slurm cluster configuration and worker management.
    """

    def __init__(
        self,
        worker_cache_dir: str,  # Cache directory mounted to the container when starting a worker
        worker_data_dir: str = None,  # Data directory mounted to the container when starting a worker
        image: str = f"ghcr.io/aicell-lab/bioengine-worker:{__version__}",  # BioEngine image tag or path to the image file
        default_num_gpus: int = 1,
        default_num_cpus: int = 8,
        default_mem_per_cpu: int = 16,
        default_time_limit: str = "4:00:00",
        further_slurm_args: List[str] = None,
        log_file: Optional[str] = None,
        debug: bool = False,
    ):
        """
        Initialize the SlurmWorkers with the given configuration.

        :param config: Configuration dictionary for the Slurm workers.
        """
        # Set up logging
        self.logger = create_logger(
            name="SlurmManager",
            level=logging.DEBUG if debug else logging.INFO,
            log_file=log_file,
        )

        # Set SLURM job name and log directory
        self.worker_cache_dir = worker_cache_dir
        self.worker_data_dir = worker_data_dir
        self.image = image
        self.job_name = "ray_worker"
        self.default_num_gpus = default_num_gpus
        self.default_num_cpus = default_num_cpus
        self.default_mem_per_cpu = default_mem_per_cpu
        self.default_time_limit = default_time_limit
        self.further_slurm_args = further_slurm_args or []
        self.slurm_log_dir = Path(self.worker_cache_dir) / "slurm_logs"

        if self.worker_cache_dir is None:
            raise ValueError(
                "Mountable worker cache directory must be set in 'SLURM' mode"
            )

    async def _check_slurm_available(self) -> bool:
        """Check if SLURM is available on the system.

        Returns:
            True if SLURM is available, False otherwise
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                "sinfo", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            await proc.communicate()
            if proc.returncode == 0:
                self.logger.info("SLURM is available")
                return True
            else:
                self.logger.info("SLURM is not available")
                return False
        except FileNotFoundError:
            self.logger.info("SLURM is not available")
            return False

    def _get_job_id(self, node_resource: Dict) -> Optional[str]:
        # Extract worker ID from resources
        job_id = None
        for resource in node_resource.keys():
            if resource.startswith("node:__internal_job_id"):
                job_id = resource.split("_")[-3]
                break
        return job_id

    async def _create_sbatch_script(
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
                # TODO: set APPTAINER_CACHEDIR to worker_cache_dir
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

    async def _submit_job(self, sbatch_script: str, delete_script: bool = False) -> str:
        try:
            # Submit the job with sbatch
            proc = await asyncio.create_subprocess_exec(
                "sbatch",
                sbatch_script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                raise subprocess.CalledProcessError(
                    proc.returncode, "sbatch", stderr=error_msg
                )

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

    async def _get_jobs(self) -> Dict[str, Dict[str, str]]:
        """Query SLURM for status of all jobs.

        Returns:
            Dict mapping job_id to dict containing name, state, runtime, time_limit
        """
        try:
            # Run squeue command to get all jobs for the current user
            # Format: JobID, Name, State, Time, TimeLimit
            proc = await asyncio.create_subprocess_exec(
                "squeue",
                "-u",
                os.environ["USER"],
                "-o",
                "%i %j %T %M %L",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                raise subprocess.CalledProcessError(
                    proc.returncode, "squeue", stderr=error_msg
                )

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

    async def _cancel_jobs(
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
                "scancel",
                *jobs_to_cancel,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                raise subprocess.CalledProcessError(
                    proc.returncode, "scancel", stderr=error_msg
                )

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

    # add_worker
    async def start_worker(
        self,
        num_gpus: int = 1,
        num_cpus: int = 8,
        mem_per_cpu: int = 16,
        time_limit: str = "6:00:00",
    ) -> None:
        """Start a new Slurm worker as Ray worker node.

        Creates temporary batch script to launch containerized Ray worker
        with configured resources.

        Args:
            num_gpus: GPUs allocated per worker
            num_cpus: CPUs allocated per worker
            mem_per_cpu: Memory (GB) allocated per CPU
            time_limit: SLURM job time limit (HH:MM:SS)
        """
        try:
            # Check if Ray cluster is running
            if not ray.is_initialized():
                raise RuntimeError("Ray cluster is not running")

            # Define the apptainer command with the Ray worker command
            apptainer_cmd = (
                "apptainer run "
                "--nv "
                "--cleanenv "
                '--env="SLURM_JOB_ID"="$SLURM_JOB_ID" '
                "--pwd /app "
                f"--bind {self.worker_cache_dir}:/tmp/bioengine"
            )
            self.logger.info(
                f"Binding cache directory '{self.worker_cache_dir}' to container directory '/tmp/bioengine'"
            )

            # Add data directory binding if specified
            if self.worker_data_dir:
                self.logger.info(
                    f"Binding data directory '{self.worker_data_dir}' to container directory '/data'"
                )
                apptainer_cmd += f" --bind {self.worker_data_dir}:/data"

            # Add the container image to the command
            apptainer_cmd += f" {self.image}"

            # Define the Ray worker command that will run inside the container and add it to the command
            ray_worker_cmd = (
                "ray start "
                f"--address={self.head_node_address} "
                f"--num-cpus={num_cpus} "
                f"--num-gpus={num_gpus} "
                "--resources='{\\\"node:__internal_job_id_${SLURM_JOB_ID}__\\\": 1}' "
                "--block"
            )
            # ray start --address='10.81.254.11:6379' --num-cpus=8 --num-gpus=1 --resources='{"node:__internal_job_id_${SLURM_JOB_ID}__": 1}' --block
            apptainer_cmd += f" {ray_worker_cmd}"

            # Create sbatch script using SlurmManager
            sbatch_script = await self.slurm_manager.create_sbatch_script(
                command=apptainer_cmd,
                gpus=num_gpus,
                cpus_per_task=num_cpus,
                mem_per_cpu=mem_per_cpu,
                time=time_limit,
                further_slurm_args=self.job_config.get("further_slurm_args"),
            )

            # Submit the job
            job_id = await self.slurm_manager.submit_job(
                sbatch_script, delete_script=True
            )

            if job_id:
                self.logger.info(
                    f"Worker job submitted successfully. Worker & Job ID: {job_id}, Resources: {num_gpus} GPU(s), "
                    f"{num_cpus} CPU(s), {mem_per_cpu}G mem/CPU, {time_limit} time limit"
                )

                return job_id  # equivalent to worker ID
            else:
                raise RuntimeError("Failed to submit worker job")

        except Exception as e:
            self.logger.error(f"Error adding worker: {e}")
            raise e

    # def _get_worker_status(self, worker_id: str) -> str:
    #     # TODO: needed?
    #     # Check if worker node exists in cluster
    #     for node in list_nodes():
    #         node_worker_id = self._node_resource_to_worker_id(node["resources_total"])
    #         if node_worker_id == worker_id:
    #             node_id = node["NodeID"]
    #             node_ip = node["NodeManagerAddress"]

    #             if node["Alive"]:
    #                 self.logger.debug(
    #                     f"Ray worker '{worker_id}' on machine '{node_ip}' with node ID '{node_id}' is currently running"
    #                 )
    #                 return "alive"
    #             else:
    #                 self.logger.debug(
    #                     f"Ray worker '{worker_id}' on machine '{node_ip}' with node ID '{node_id}' is already stopped"
    #                 )
    #                 return "dead"

    #     # If not found in either alive or dead nodes
    #     self.logger.debug(f"Ray worker '{worker_id}' not found in cluster")
    #     return "not_found"

    # remove_worker
    async def stop_worker(self, node_id: str, grace_period: int = 30) -> None:
        """Shut down an existing Slurm worker node from the Ray cluster.


        Returns:
            True if worker was successfully shut down
        """
        try:
            # Check if Ray cluster is running
            if not ray.is_initialized():
                raise RuntimeError("Ray cluster is not running")

            # Check worker status
            worker_status = get_node(node_id)

            if worker_status == "not_found":
                raise ValueError(f"Node ID '{worker_id}' not found in cluster status")

            self.logger.info(
                f"Removing worker '{worker_id}' (status='{worker_status}') from cluster status..."
            )

            if worker_status == "alive":
                self.logger.info(f"Stopping all processes on worker '{worker_id}'...")

                @ray.remote(resources={f"node:__internal_job_id_{worker_id}__": 0.01})
                def stop_worker():
                    """Stops the worker node by executing 'ray stop'."""
                    import subprocess

                    subprocess.Popen(
                        ["ray", "stop"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )

                # This should only effect a single worker node if the worker runs in a container
                obj_ref = stop_worker.remote()

                try:
                    await asyncio.to_thread(ray.get, obj_ref, timeout=15)
                except ray.exceptions.GetTimeoutError:
                    self.logger.error(
                        f"Failed to send shutdown command to worker '{worker_id}' within 15 seconds"
                    )
                    raise

                # Wait for worker node to disappear from cluster status
                start_time = time.time()
                while time.time() - start_time < grace_period:
                    await asyncio.sleep(3)
                    status = self._get_worker_status(worker_id)
                    if status != "alive":
                        break

            cancelled_jobs = await self.slurm_manager.cancel_jobs([worker_id])
            if cancelled_jobs == [worker_id]:
                self.logger.info(f"Successfully removed worker '{worker_id}'")
            else:
                raise RuntimeError(f"Failed to cancel job '{worker_id}'")

        except Exception as e:
            self.logger.error(f"Error shutting down worker {worker_id}: {e}")
            raise e


if __name__ == "__main__":
    import asyncio

    async def main():
        # Example usage
        actor = SlurmWorkers(job_name="ray_worker", slurm_log_dir="/tmp")
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
