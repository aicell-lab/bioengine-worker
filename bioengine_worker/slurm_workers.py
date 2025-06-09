import asyncio
import logging
import os
import subprocess
import tempfile
import time
from typing import Dict, List, Optional

import ray
from ray.util.state import get_node, list_nodes

from bioengine_worker import __version__
from bioengine_worker.utils import create_logger


class SlurmWorkers:
    """
    A class to manage Slurm workers for Bioengine.

    This class is responsible for handling the Slurm cluster configuration and worker management.
    It provides functionality to start, stop, and monitor Ray worker nodes running as Slurm jobs
    within containerized environments using Apptainer/Singularity.
    """

    def __init__(
        self,
        head_node_address: str,
        worker_cache_dir: str,
        image: str = f"docker://ghcr.io/aicell-lab/bioengine-worker:{__version__}",
        worker_data_dir: Optional[str] = None,
        default_num_gpus: int = 1,
        default_num_cpus: int = 8,
        default_mem_per_cpu: int = 16,
        default_time_limit: str = "4:00:00",
        further_slurm_args: Optional[List[str]] = None,
        log_file: Optional[str] = None,
        debug: bool = False,
    ):
        """
        Initialize the SlurmWorkers with the given configuration.

        Args:
            head_node_address: Address of the Ray head node to connect workers to
            worker_cache_dir: Cache directory mounted to the container when starting a worker
            image: BioEngine remote docker image or path to the image file
            worker_data_dir: Optional data directory mounted to the container when starting a worker
            default_num_gpus: Default number of GPUs to allocate per worker
            default_num_cpus: Default number of CPUs to allocate per worker
            default_mem_per_cpu: Default memory (GB) to allocate per CPU
            default_time_limit: Default SLURM job time limit (HH:MM:SS format)
            further_slurm_args: Additional SLURM arguments to include in job submissions
            log_file: Optional log file to write logs to
            debug: Enable debug logging

        Raises:
            ValueError: If worker_cache_dir is None
        """
        # Set up logging
        self.logger = create_logger(
            name="SlurmManager",
            level=logging.DEBUG if debug else logging.INFO,
            log_file=log_file,
        )

        # Set SLURM job name and log directory
        self.head_node_address = head_node_address
        self.image = image
        self.job_name = "ray_worker"
        self.worker_cache_dir = str(worker_cache_dir)
        self.worker_data_dir = str(worker_data_dir) if worker_data_dir else None
        self.default_num_gpus = default_num_gpus
        self.default_num_cpus = default_num_cpus
        self.default_mem_per_cpu = default_mem_per_cpu
        self.default_time_limit = default_time_limit
        self.further_slurm_args = further_slurm_args if further_slurm_args else []

        if self.worker_cache_dir is None:
            raise ValueError(
                "Mountable worker cache directory must be set in 'SLURM' mode"
            )

    def _create_sbatch_script(
        self,
        num_gpus: int,
        num_cpus: int,
        mem_per_cpu: int,
        time_limit: str,
        further_slurm_args: List[str],
    ) -> str:
        """
        Create a SLURM batch script for launching a Ray worker in a container.

        Args:
            num_gpus: Number of GPUs to allocate
            num_cpus: Number of CPUs to allocate
            mem_per_cpu: Memory (GB) to allocate per CPU
            time_limit: SLURM job time limit (HH:MM:SS format)
            further_slurm_args: Additional SLURM arguments to include

        Returns:
            Path to the created temporary batch script file

        Raises:
            Exception: If script creation fails
        """
        try:
            # Define the apptainer command with the Ray worker command
            apptainer_cmd = (
                "apptainer exec "
                "--nv "
                "--cleanenv "
                "--env=SLURM_JOB_ID='${SLURM_JOB_ID}' "
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

            # Define the Ray worker command that will run inside the container and add it to the command
            ray_worker_cmd = (
                "ray start "
                f"--address={self.head_node_address} "
                f"--num-cpus={num_cpus} "
                f"--num-gpus={num_gpus} "
                "--resources='{\"node:slurm_job_id='${SLURM_JOB_ID}'\": 1}' "
                "--block"
            )
            # Example: ray start --address='10.81.254.11:6379' --num-cpus=8 --num-gpus=1 --resources='{"node:slurm_job_id=${SLURM_JOB_ID}": 1}' --block

            # Add the container image and Ray worker command to the apptainer command
            apptainer_cmd += f" {self.image} {ray_worker_cmd}"

            # Define further SLURM arguments if provided
            further_slurm_args = (
                "\n".join([f"#SBATCH {arg}" for arg in further_slurm_args if arg])
                if further_slurm_args
                else ""
            )

            # Define content of batch script with SLURM directives
            batch_script = f"""#!/bin/bash
            #SBATCH --job-name={self.job_name}
            #SBATCH --ntasks=1
            #SBATCH --nodes=1
            #SBATCH --gpus={num_gpus}
            #SBATCH --cpus-per-task={num_cpus}
            #SBATCH --mem-per-cpu={mem_per_cpu}G
            #SBATCH --time={time_limit}
            #SBATCH --chdir={self.worker_cache_dir}
            #SBATCH --output={self.worker_cache_dir}/slurm_logs/%x_%j.out
            #SBATCH --error={self.worker_cache_dir}/slurm_logs/%x_%j.err
            {further_slurm_args}

            # Print some diagnostic information
            echo "Host: $(hostname)"
            echo "Date: $(date)"
            echo "GPUs: {num_gpus}, CPUs: {num_cpus}"
            echo "GPU info: $(nvidia-smi -L)"
            echo "Job ID: $SLURM_JOB_ID"
            echo "Working directory: $(pwd)"
            echo "Running command: {apptainer_cmd}"
            echo ""
            echo "========================================"
            echo ""

            if [ -z "$SLURM_JOB_ID" ]; then
                echo "SLURM_JOB_ID is not set. This script may not be running in a SLURM job."
                exit 1
            fi

            # Create the log directory if it doesn't exist
            mkdir -p {self.worker_cache_dir}/slurm_logs

            # Reset bound in paths in case of submission from a container
            APPTAINER_BIND=
            SINGULARITY_BIND=

            # Set the cache directory for Apptainer
            export APPTAINER_CACHEDIR="{self.worker_cache_dir}/images"
            export SINGULARITY_CACHEDIR="{self.worker_cache_dir}/images"

            {apptainer_cmd}

            # Print completion status
            echo "Job completed with status $?"
            """

            # Create a temporary batch script
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".sh", delete=False
            ) as batch_file:
                for line in batch_script.split("\n"):
                    # Remove leading whitespace
                    line = line.strip()
                    batch_file.write(line + "\n")

                self.logger.debug(
                    f"Created sbatch script '{batch_file.name}':\n{batch_script}"
                )

            return batch_file.name
        except Exception as e:
            self.logger.error(f"Failed to create sbatch script: {e}")
            raise e

    async def _submit_job(self, sbatch_script: str, delete_script: bool = False) -> str:
        """
        Submit a SLURM job using the provided batch script.

        Args:
            sbatch_script: Path to the batch script file to submit
            delete_script: Whether to delete the script file after submission

        Returns:
            The SLURM job ID of the submitted job

        Raises:
            subprocess.CalledProcessError: If sbatch command fails
            RuntimeError: If job ID cannot be parsed from output
            Exception: If job submission fails for any other reason
        """
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

    async def get_jobs(self) -> Dict[str, Dict[str, str]]:
        """
        Query SLURM for status of all BioEngine worker jobs.

        Returns:
            Dict mapping job_id to dict containing name, state, runtime, time_limit

        Raises:
            subprocess.CalledProcessError: If squeue command fails
            Exception: If job status retrieval fails for any other reason
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

                        # Only include BioEngine worker jobs
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

    def get_job_id_from_resource(self, node_resource: Dict) -> Optional[str]:
        """
        Extract the SLURM job ID from a Ray node's resource dictionary.

        Args:
            node_resource: Dictionary of node resources from Ray

        Returns:
            The SLURM job ID if found, None otherwise
        """
        # Extract worker ID from resources
        job_id = None
        for resource in node_resource.keys():
            if resource.startswith("node:slurm_job_id="):
                job_id = resource.split("=")[1]
                break
        return job_id

    async def cancel_jobs(
        self, job_ids: Optional[List[str]] = None, grace_period: Optional[int] = 60
    ) -> List[str]:
        """
        Cancel running SLURM jobs.

        Args:
            job_ids: Specific job IDs to cancel. If None, cancels all BioEngine worker jobs
            grace_period: Seconds to wait until job is cancelled

        Returns:
            List of job IDs that were cancelled

        Raises:
            ValueError: If no matching jobs found to cancel when job_ids are specified
            subprocess.CalledProcessError: If scancel command fails
            Exception: If job cancellation fails for any other reason
        """
        # First get all ray worker jobs
        jobs = await self.get_jobs()
        jobs_to_cancel = list(jobs.keys())

        if not jobs_to_cancel:
            self.logger.info("No running BioEngine worker jobs found")
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

    async def start_worker(
        self,
        num_gpus: Optional[int] = None,
        num_cpus: Optional[int] = None,
        mem_per_cpu: Optional[int] = None,
        time_limit: Optional[str] = None,
        further_slurm_args: Optional[List[str]] = None,
    ) -> str:
        """
        Start a new Slurm worker as Ray worker node.

        Creates temporary batch script to launch containerized Ray worker
        with configured resources.

        Args:
            num_gpus: Number of GPUs to allocate per worker. Uses default if None
            num_cpus: Number of CPUs to allocate per worker. Uses default if None
            mem_per_cpu: Memory (GB) to allocate per CPU. Uses default if None
            time_limit: SLURM job time limit (HH:MM:SS format). Uses default if None
            further_slurm_args: Additional SLURM arguments. Uses default if None

        Returns:
            The Ray node ID of the started worker

        Raises:
            RuntimeError: If Ray is not initialized or worker fails to start
            Exception: If worker creation fails for any other reason
        """
        try:
            submitted_job_id = None

            # Set default values if not provided
            num_gpus = num_gpus if num_gpus is not None else self.default_num_gpus
            num_cpus = num_cpus if num_cpus is not None else self.default_num_cpus
            mem_per_cpu = (
                mem_per_cpu if mem_per_cpu is not None else self.default_mem_per_cpu
            )
            time_limit = (
                time_limit if time_limit is not None else self.default_time_limit
            )
            further_slurm_args = (
                further_slurm_args
                if further_slurm_args is not None
                else self.further_slurm_args
            )

            # Check if Ray cluster is running
            if not ray.is_initialized():
                raise RuntimeError("Ray is not initialized. Call start() first.")

            # Create sbatch script using SlurmManager
            sbatch_script = await asyncio.to_thread(
                self._create_sbatch_script,
                num_gpus=num_gpus,
                num_cpus=num_cpus,
                mem_per_cpu=mem_per_cpu,
                time_limit=time_limit,
                further_slurm_args=further_slurm_args,
            )

            # Submit the job
            submitted_job_id = await self._submit_job(sbatch_script, delete_script=True)
            self.logger.info(
                f"Worker job submitted successfully. Worker & Job ID: {submitted_job_id}, Resources: {num_gpus} GPU(s), "
                f"{num_cpus} CPU(s), {mem_per_cpu}G mem/CPU, {time_limit} time limit"
            )

            # Wait for the worker to be added to the Ray cluster
            node_is_pending = True
            while node_is_pending:
                await asyncio.sleep(5)

                # Check if the job is still running
                jobs = await self.get_jobs()
                if submitted_job_id not in jobs:
                    raise RuntimeError(
                        f"Job {submitted_job_id} is no longer running. Worker may not have started successfully."
                    )

                # Check if the worker node has already been added to the Ray cluster
                all_nodes = await asyncio.to_thread(
                    list_nodes, address=self.head_node_address
                )
                for node in all_nodes:
                    if node.is_head_node:
                        continue
                    worker_job_id = self.get_job_id_from_resource(node.resources_total)
                    if worker_job_id == submitted_job_id:
                        node_id = node.node_id
                        self.logger.info(
                            f"Worker node with ID '{worker_job_id}' is now running in the Ray cluster"
                        )
                        node_is_pending = False

            return node_id

        except Exception as e:
            self.logger.error(f"Error starting worker: {e}")
            if submitted_job_id:
                self.cancel_jobs([submitted_job_id])
            raise e

    async def stop_worker(self, node_id: str, grace_period: int = 60) -> None:
        """
        Shut down an existing Slurm worker node from the Ray cluster.

        Gracefully stops the worker by sending a stop command to the node,
        then cancels the corresponding SLURM job.

        Args:
            node_id: The Ray node ID of the worker to stop
            grace_period: Seconds to wait for graceful shutdown before forcing

        Raises:
            RuntimeError: If Ray is not initialized, worker not found, or shutdown fails
            Exception: If worker shutdown fails for any other reason
        """
        try:
            # Check if Ray cluster is running
            if not ray.is_initialized():
                raise RuntimeError("Ray is not initialized. Call start() first.")

            # Check worker status
            worker_status = await asyncio.to_thread(
                get_node, id=node_id, address=self.head_node_address
            )
            if worker_status is None:
                raise RuntimeError(f"Worker '{node_id}' not found in cluster")

            job_id = self.get_job_id_from_resource(worker_status.resources_total)

            self.logger.info(
                f"Removing worker '{node_id}' (status='{worker_status}' | job_id='{job_id}') from cluster status..."
            )

            if worker_status.state == "ALIVE":
                self.logger.info(f"Stopping all processes on worker '{node_id}'...")

                @ray.remote(resources={f"node:slurm_job_id={job_id}": 0.01})
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
                    await asyncio.to_thread(ray.get, obj_ref, timeout=grace_period)
                except ray.exceptions.GetTimeoutError:
                    self.logger.error(
                        f"Failed to send shutdown command to worker '{node_id}' within {grace_period} seconds. Cancelling job..."
                    )
                    raise

                # Wait for worker node to be removed from the cluster
                start_time = time.time()
                while time.time() - start_time < grace_period:
                    await asyncio.sleep(3)
                    worker_status = await asyncio.to_thread(
                        get_node, id=node_id, address=self.head_node_address
                    )
                    if worker_status is None or worker_status.state != "ALIVE":
                        break

            cancelled_jobs = await self.cancel_jobs([job_id], grace_period=grace_period)
            if cancelled_jobs == [job_id]:
                self.logger.info(f"Successfully removed worker '{node_id}'")
            else:
                raise RuntimeError(
                    f"Failed to cancel job '{job_id}' from worker '{node_id}'"
                )

        except Exception as e:
            self.logger.error(f"Error shutting down worker {node_id}: {e}")
            raise e
