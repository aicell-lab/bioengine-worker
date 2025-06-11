import asyncio
import logging
import os
import subprocess
import tempfile
import time
from typing import Dict, List, Optional, Set

import ray
from ray.util.state import get_node, list_nodes, summarize_tasks

from bioengine_worker import __version__
from bioengine_worker.utils import create_logger


class SlurmWorkers:
    """
    Manages SLURM workers for BioEngine with intelligent autoscaling capabilities.

    This class provides comprehensive management of Ray worker nodes running as SLURM jobs
    within containerized environments using Apptainer/Singularity. It includes intelligent
    autoscaling that automatically adjusts the number of workers based on pending tasks,
    resource utilization, and configurable policies.

    The autoscaling system continuously monitors the Ray cluster state and makes scaling decisions based on:
    - Number of pending tasks requiring node assignment
    - Current worker node utilization (CPU/GPU usage patterns)
    - Configurable cooldown periods to prevent oscillations
    - Min/max worker limits for cost control
    - Historical idle time patterns for efficient scale-down decisions

    Key Features:
    - Automatic worker scaling with intelligent demand prediction
    - Containerized worker execution with Apptainer/Singularity support
    - Graceful worker shutdown with proper job lifecycle management
    - Comprehensive error handling with retry mechanisms
    - Customizable SLURM job parameters and resource allocation
    - Real-time monitoring and logging for operational visibility

    Attributes:
        ray_cluster: Reference to the Ray cluster manager instance
        image (str): Container image path for worker nodes
        job_name (str): SLURM job name prefix for worker identification
        worker_cache_dir (str): Cache directory mounted to worker containers
        worker_data_dir (str): Optional data directory mounted to worker containers
        default_num_gpus (int): Default GPU allocation per worker
        default_num_cpus (int): Default CPU allocation per worker
        default_mem_per_cpu (int): Default memory (GB) per CPU
        default_time_limit (str): Default SLURM job time limit
        further_slurm_args (List[str]): Additional SLURM arguments
        min_workers (int): Minimum number of workers to maintain
        max_workers (int): Maximum number of workers allowed
        check_interval (int): Interval in seconds between scaling checks
        scale_down_threshold (int): Idle time in seconds before scaling down
        scale_up_cooldown (int): Cooldown period in seconds between scale-ups
        scale_down_cooldown (int): Cooldown period in seconds between scale-downs
        is_running (bool): Whether the autoscaling monitoring loop is active
        monitoring_task (asyncio.Task): Background monitoring task handle
        last_scaling_decision (float): Timestamp of last scaling decision
        last_scale_up_time (float): Timestamp of last scale-up action
        last_scale_down_time (float): Timestamp of last scale-down action
    """

    def __init__(
        self,
        ray_cluster,
        # Slurm job configuration parameters
        worker_cache_dir: str,
        image: str = f"ghcr.io/aicell-lab/bioengine-worker:{__version__}",
        worker_data_dir: Optional[str] = None,
        default_num_gpus: int = 1,
        default_num_cpus: int = 8,
        default_mem_per_cpu: int = 16,
        default_time_limit: str = "4:00:00",
        further_slurm_args: Optional[List[str]] = None,
        # Autoscaling configuration parameters
        min_workers: int = 0,
        max_workers: int = 4,
        check_interval_seconds: int = 60,  # Check scaling once per minute
        scale_down_threshold_seconds: int = 300,  # 5 minutes of idleness before scale down
        scale_up_cooldown_seconds: int = 30,  # 30 seconds between scale ups
        scale_down_cooldown_seconds: int = 60,  # 1 minute between scale downs
        grace_period: int = 60,  # Grace period for job cancellation
        # Logger
        log_file: Optional[str] = None,
        debug: bool = False,
    ):
        """
        Initialize the SlurmWorkers with the given configuration.

        Args:
            ray_cluster: Ray cluster manager instance
            worker_cache_dir: Cache directory mounted to the container when starting a worker
            image: BioEngine remote docker image or path to the image file
            worker_data_dir: Optional data directory mounted to the container when starting a worker
            default_num_gpus: Default number of GPUs to allocate per worker
            default_num_cpus: Default number of CPUs to allocate per worker
            default_mem_per_cpu: Default memory (GB) to allocate per CPU
            default_time_limit: Default SLURM job time limit (HH:MM:SS format)
            further_slurm_args: Additional SLURM arguments to include in job submissions
            min_workers: Minimum number of workers to maintain
            max_workers: Maximum number of workers allowed
            check_interval_seconds: Interval in seconds between scaling checks
            scale_down_threshold_seconds: Idle time in seconds before scaling down
            scale_up_cooldown_seconds: Cooldown period in seconds between scale-ups
            scale_down_cooldown_seconds: Cooldown period in seconds between scale-downs
            grace_period: Grace period in seconds for job cancellation
            log_file: Optional log file to write logs to
            debug: Enable debug logging

        Raises:
            ValueError: If worker_cache_dir is None
        """
        # Set up logging
        self.logger = create_logger(
            name="SlurmWorkers",
            level=logging.DEBUG if debug else logging.INFO,
            log_file=log_file,
        )

        if worker_cache_dir is None:
            raise ValueError(
                "Mountable worker cache directory must be set in 'SLURM' mode"
            )

        self.ray_cluster = ray_cluster

        # SLURM job configuration parameters
        self.image = (
            f"docker://{image}"
            if not image.startswith("docker://") and not image.endswith(".sif")
            else image
        )
        self.job_name = "ray_worker"
        self.worker_cache_dir = str(worker_cache_dir)
        self.worker_data_dir = str(worker_data_dir) if worker_data_dir else None
        self.default_num_gpus = default_num_gpus
        self.default_num_cpus = default_num_cpus
        self.default_mem_per_cpu = default_mem_per_cpu
        self.default_time_limit = default_time_limit
        self.further_slurm_args = further_slurm_args if further_slurm_args else []

        # Autoscaling configuration parameters
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.check_interval = check_interval_seconds
        self.scale_down_threshold = scale_down_threshold_seconds
        self.scale_up_cooldown = scale_up_cooldown_seconds
        self.scale_down_cooldown = scale_down_cooldown_seconds
        self.grace_period = grace_period

        # Initialize state variables
        self.last_scaling_decision = 0
        self.last_scale_up_time = 0
        self.last_scale_down_time = 0

        # Background task
        self.monitoring_task = None
        self.is_running = False

    async def _get_num_pending_tasks(self) -> int:
        """
        Get the number of pending tasks in the Ray cluster.

        Queries the Ray cluster to count tasks that are waiting for node assignment.
        These tasks indicate demand for additional worker resources.

        Returns:
            Number of tasks with PENDING_NODE_ASSIGNMENT state

        Raises:
            Exception: If task summary retrieval fails due to Ray connection issues
        """
        summary = await asyncio.to_thread(
            summarize_tasks, address=self.ray_cluster.head_node_address
        )
        num_pending_tasks = sum(
            task["state_counts"].get("PENDING_NODE_ASSIGNMENT", 0)
            for task in summary["cluster"]["summary"].values()
        )
        return num_pending_tasks

    async def _get_num_worker_jobs(self) -> int:
        """
        Get number of SLURM workers in the cluster (configuring, pending or running).

        Queries SLURM for all BioEngine worker jobs regardless of their current state.
        This includes jobs that are pending in the queue, configuring, or actively running.

        Returns:
            Total count of BioEngine worker jobs in all states

        Raises:
            Exception: If job query fails due to SLURM connection issues
        """
        return len(await self._get_jobs())

    async def _try_scale_up(self, n_worker_jobs: int, num_pending_tasks: int) -> None:
        """
        Attempt to scale up the cluster by adding a new worker.

        Checks if scaling up is allowed based on cooldown periods and worker limits.
        If conditions are met, submits a new SLURM job with default resource allocation.

        Args:
            n_worker_jobs: Current number of worker jobs in SLURM
            num_pending_tasks: Number of pending tasks requiring resources

        Returns:
            None

        Raises:
            Exception: If worker creation fails
        """
        # SCALE UP LOGIC
        can_scale_up = (
            # Cooldown period has passed
            (time.time() - self.last_scale_up_time) > self.scale_up_cooldown
            # Not at max worker limit
            and n_worker_jobs < self.max_workers
        )
        if not can_scale_up:
            return

        # TODO: Check if any pending task needs more resources than default
        # for task in pending_tasks:
        #     task_req = task.get("required_resources", {}) or {}
        #     if task_req.get("GPU", 0) > num_gpus:
        #         num_gpus = max(num_gpus, int(task_req.get("GPU", 1)))
        #     if task_req.get("CPU", 0) > num_cpus:
        #         num_cpus = max(num_cpus, int(task_req.get("CPU", 4)))

        num_gpus = None or self.default_num_gpus
        num_cpus = None or self.default_num_cpus
        mem_per_cpu = None or self.default_mem_per_cpu
        time_limit = None or self.default_time_limit
        further_slurm_args = None or self.further_slurm_args

        self.logger.info(
            f"Scaling up with {num_gpus} GPU(s) and {num_cpus} CPU(s) due to {num_pending_tasks} pending task(s)"
        )
        await self.add(
            num_gpus=num_gpus,
            num_cpus=num_cpus,
            mem_per_cpu=mem_per_cpu,
            time_limit=time_limit,
            further_slurm_args=further_slurm_args,
        )
        self.last_scale_up_time = time.time()

    async def _check_is_idle(self, node_info: Dict) -> bool:
        """
        Check if a worker node is idle based on its GPU and CPU utilization.

        A node is considered idle if it has no active CPU or GPU usage.

        Args:
            node_info: Dictionary containing node resource information with keys
                      'total_cpu', 'available_cpu', 'total_gpu', 'available_gpu'

        Returns:
            True if the node is completely idle (no used CPUs or GPUs), False otherwise
        """
        used_cpus = node_info["total_cpu"] - node_info["available_cpu"]
        # GPU nodes
        if node_info["total_gpu"] > 0:
            used_gpus = node_info["total_gpu"] - node_info["available_gpu"]
        else:
            used_gpus = 0

        # Node is idle if no GPUs or CPUs are used
        return used_gpus == 0 and used_cpus == 0

    async def _find_idle_nodes(self, running_nodes: List) -> Set[str]:
        """
        Find idle worker nodes based on GPU and CPU utilization metrics.

        Iterates through all running nodes and identifies those that are completely idle
        (no GPU or CPU usage).

        Args:
            running_nodes: List of node information dictionaries from Ray cluster

        Returns:
            Set of node_ids that are currently idle
        """
        idle_nodes = set()
        for node_info in running_nodes:
            if await self._check_is_idle(node_info):
                # Node is completely idle
                self.logger.debug(
                    f"Node {node_info['node_id']} is completely idle (no used GPUs or CPUs)"
                )
                idle_nodes.add(node_info["node_id"])

        return idle_nodes

    async def _try_scale_down(self, n_worker_jobs) -> None:
        """
        Attempt to scale down the cluster by removing idle workers.

        Checks if scaling down is allowed based on cooldown periods and worker limits.
        If conditions are met, identifies the longest idle node and removes it.

        Args:
            n_worker_jobs: Current number of worker jobs in SLURM

        Returns:
            None

        Raises:
            Exception: If worker removal fails
        """
        # SCALE DOWN LOGIC
        current_time = time.time()
        can_scale_down = (
            # Cooldown period has passed
            (current_time - self.last_scale_down_time) > self.scale_down_cooldown
            # Not at min worker limit
            and n_worker_jobs > self.min_workers
        )
        if not can_scale_down:
            return

        # Get the longest idle node
        longest_idle_nodes = None
        idle_time = 0
        for i, (timepoint, nodes_status) in enumerate(
            reversed(self.ray_cluster.worker_nodes_history.items())
        ):
            running_nodes = nodes_status.get("RUNNING", [])
            idle_nodes = await self._find_idle_nodes(running_nodes)

            # If no idle nodes found, we can stop checking further
            if not idle_nodes:
                break

            # If this is the first iteration, initialize longest_idle_nodes
            if i == 0:
                longest_idle_nodes = idle_nodes
            else:
                # Otherwise, find the intersection of longest idle nodes and current idle nodes
                overlap = longest_idle_nodes & idle_nodes
                if len(overlap) == 0:
                    # No overlap means no common idle nodes, we can stop checking further
                    break

                longest_idle_nodes = overlap

            # Calculate idle time for the current timepoint
            idle_time = current_time - timepoint

            if len(longest_idle_nodes) == 1:
                # If we have only one idle node, we can stop checking further
                break

        if longest_idle_nodes and idle_time > self.scale_down_threshold:
            # Get the first longest idle node (if there are multiple nodes with the same idle time)
            for idle_node_id in list(longest_idle_nodes):
                # Check worker status
                worker_status = await asyncio.to_thread(
                    get_node,
                    id=idle_node_id,
                    address=self.ray_cluster.head_node_address,
                )
                # Check if worker is still idle
                # TODO: Check if node is still idle
                if worker_status:
                    self.logger.info(
                        f"Scaling down worker '{idle_node_id}' due to inactivity for {idle_time:.1f}s"
                    )
                    await self._close(idle_node_id)
                    self.last_scale_down_time = current_time

    async def _make_scaling_decisions(self) -> None:
        """
        Make scaling decisions based on resource utilization and pending tasks.

        Evaluates the current cluster state including pending tasks and worker jobs,
        then decides whether to scale up (if there are pending tasks) or scale down
        (if workers are idle). Updates the last scaling decision timestamp.

        Raises:
            Exception: If Ray is not initialized or scaling decision fails
        """
        current_time = time.time()
        # Check if Ray is still initialized
        if not ray.is_initialized():
            self.logger.warning("Ray is not initialized, skipping scaling decision")
            self.last_scaling_decision = current_time
            return

        num_pending_tasks = await self._get_num_pending_tasks()
        n_worker_jobs = await self._get_num_worker_jobs()

        # If at least one task needs resources, check scale up, otherwise check scale down
        if num_pending_tasks > 0:
            await self._try_scale_up(n_worker_jobs, num_pending_tasks)
        else:
            await self._try_scale_down(n_worker_jobs)
        self.last_scaling_decision = current_time

    async def _monitoring_loop(self, max_consecutive_errors: int = 3) -> None:
        """
        Main monitoring loop for autoscaling.

        Continuously monitors the cluster state and makes scaling decisions at regular
        intervals. Handles errors gracefully and stops after consecutive failures.

        Args:
            max_consecutive_errors: Maximum number of consecutive errors before stopping

        Raises:
            Exception: If max consecutive errors is reached or monitoring fails
        """
        consecutive_errors = 0

        while self.is_running:
            try:
                await asyncio.sleep(1)
                if time.time() - self.last_scaling_decision < self.check_interval:
                    continue

                # Check if a scaling decision is needed
                self.logger.debug("Starting autoscaling monitoring iteration")
                await self._make_scaling_decisions()

                consecutive_errors = 0  # Reset error counter on success

            except asyncio.CancelledError:
                self.logger.info("Autoscaling monitoring loop cancelled")
                break

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                consecutive_errors += 1

                if consecutive_errors >= max_consecutive_errors:
                    self.logger.error(
                        f"Stopping monitoring loop after {consecutive_errors} consecutive errors"
                    )
                    raise e

            # Update last scaling decision time
            self.last_scaling_decision = time.time()

        # Ensure autoscaling stops if we break from the loop
        if self.is_running:
            asyncio.create_task(self.stop())

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

        Generates a temporary batch script with SLURM directives and Apptainer commands
        to launch a containerized Ray worker with specified resource allocation.

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
                f"--address={self.ray_cluster.head_node_address} "
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

        Executes the sbatch command to submit a job to the SLURM scheduler and parses
        the job ID from the output.

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

    async def _get_jobs(self) -> Dict[str, Dict[str, str]]:
        """
        Query SLURM for status of all BioEngine worker jobs.

        Executes squeue command to retrieve information about all jobs for the current user,
        filters for BioEngine worker jobs, and returns their status information.

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

    async def _get_job_ids(self) -> List[str]:
        """
        Get a list of all running BioEngine worker job IDs.

        Returns:
            List of job IDs for all running BioEngine worker jobs

        Raises:
            Exception: If job retrieval fails for any reason
        """
        jobs = await self._get_jobs()
        return list(jobs.keys())

    async def _get_job_id_from_resource(self, node_resource: Dict) -> Optional[str]:
        """
        Extract the SLURM job ID from a Ray node's resource dictionary.

        Searches through a Ray node's resource dictionary to find and extract
        the SLURM job ID that was assigned when the worker was created.

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

    async def _cancel_jobs(self, job_ids: Optional[List[str]] = None) -> List[str]:
        """
        Cancel running SLURM jobs.

        Executes the scancel command to cancel SLURM jobs and waits for them
        to be properly terminated within the grace period.

        Args:
            job_ids: Specific job IDs to cancel. If None, cancels all BioEngine worker jobs

        Returns:
            List of job IDs that were cancelled

        Raises:
            ValueError: If no matching jobs found to cancel when job_ids are specified
            subprocess.CalledProcessError: If scancel command fails
            Exception: If job cancellation fails for any other reason
        """
        # First get all ray worker jobs
        jobs_to_cancel = await self._get_job_ids()

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
            while time.time() - start_time < self.grace_period:
                await asyncio.sleep(1)
                running_jobs = await self._get_job_ids()
                if not any(job_id in running_jobs for job_id in jobs_to_cancel):
                    self.logger.info(
                        f"Successfully cancelled {len(jobs_to_cancel)} job(s)"
                    )
                    return jobs_to_cancel

            cancelled_jobs = [
                job_id for job_id in jobs_to_cancel if job_id not in running_jobs
            ]
            self.logger.warning(
                f"Failed to cancel all jobs within {self.grace_period} seconds. Cancelled jobs: {cancelled_jobs}"
            )
            return cancelled_jobs
        except Exception as e:
            self.logger.error(f"Failed to cancel jobs: {e}")
            raise e

    async def add(
        self,
        num_gpus: Optional[int] = None,
        num_cpus: Optional[int] = None,
        mem_per_cpu: Optional[int] = None,
        time_limit: Optional[str] = None,
        further_slurm_args: Optional[List[str]] = None,
    ) -> str:
        """
        Start a new SLURM worker as a Ray worker node.

        Creates a temporary batch script to launch a containerized Ray worker
        with the specified resources. Submits the job to SLURM and waits for
        the worker to join the Ray cluster.

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
                job_ids = await self._get_job_ids()
                if submitted_job_id not in job_ids:
                    raise RuntimeError(
                        f"Job {submitted_job_id} is no longer running. Worker may not have started successfully."
                    )

                # Check if the worker node has already been added to the Ray cluster
                all_nodes = await asyncio.to_thread(
                    list_nodes,
                    address=self.ray_cluster.head_node_address,
                    filters=[("state", "=", "ALIVE"), ("is_head_node", "=", False)],
                )
                for node in all_nodes:
                    worker_job_id = await self._get_job_id_from_resource(
                        node.resources_total
                    )
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
                self._cancel_jobs([submitted_job_id])
            raise e

    async def _close(self, node_id: str) -> None:
        """
        Shut down an existing SLURM worker node from the Ray cluster.

        Gracefully stops the worker by sending a stop command to the node,
        then cancels the corresponding SLURM job.

        Args:
            node_id: The Ray node ID of the worker to stop

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
                get_node, id=node_id, address=self.ray_cluster.head_node_address
            )
            if worker_status is None:
                raise RuntimeError(f"Worker '{node_id}' not found in cluster")

            job_id = await self._get_job_id_from_resource(worker_status.resources_total)

            self.logger.info(
                f"Removing worker '{node_id}' (status='{worker_status.state}' | job_id='{job_id}') from cluster status..."
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
                    await asyncio.to_thread(ray.get, obj_ref, timeout=self.grace_period)
                except ray.exceptions.GetTimeoutError:
                    self.logger.error(
                        f"Failed to send shutdown command to worker '{node_id}' within {self.grace_period} seconds. Cancelling job..."
                    )
                    raise

                # Wait for worker node to be removed from the cluster
                start_time = time.time()
                while time.time() - start_time < self.grace_period:
                    await asyncio.sleep(3)
                    worker_status = await asyncio.to_thread(
                        get_node, id=node_id, address=self.ray_cluster.head_node_address
                    )
                    if worker_status is None or worker_status.state != "ALIVE":
                        break

            cancelled_jobs = await self._cancel_jobs([job_id])
            if cancelled_jobs == [job_id]:
                self.logger.info(f"Successfully removed worker '{node_id}'")
            else:
                raise RuntimeError(
                    f"Failed to cancel job '{job_id}' from worker '{node_id}'"
                )

        except Exception as e:
            self.logger.error(f"Error shutting down worker {node_id}: {e}")
            raise e

    async def _close_all(self) -> None:
        """
        Shut down all SLURM worker nodes from the Ray cluster.

        Gracefully stops each worker by sending a stop command to the node,
        then cancels the corresponding SLURM job.

        Raises:
            RuntimeError: If Ray is not initialized or shutdown fails
            Exception: If worker shutdown fails for any other reason
        """
        try:
            # Check if Ray cluster is running
            if not ray.is_initialized():
                raise RuntimeError("Ray is not initialized. Call start() first.")

            # Get all worker nodes
            worker_nodes = await asyncio.to_thread(
                list_nodes,
                address=self.ray_cluster.head_node_address,
                filters=[("state", "=", "ALIVE"), ("is_head_node", "=", False)],
            )

            if not worker_nodes:
                self.logger.info("No active worker nodes found")
                return

            self.logger.info(f"Found {len(worker_nodes)} active worker nodes")

            # Stop each worker node
            for node in worker_nodes:
                await self._close(node.node_id)

            # Make sure no jobs are left running
            remaining_jobs = await self._get_job_ids()
            if remaining_jobs:
                self.logger.warning(
                    f"Some jobs are still running after shutdown: {remaining_jobs}. Cancelling them..."
                )
                cancelled_jobs = await self._cancel_jobs(remaining_jobs)
                if cancelled_jobs:
                    self.logger.warning(
                        f"Cancelled jobs after shutdown: {cancelled_jobs}"
                    )
                else:
                    self.logger.info("All jobs cancelled successfully")

        except Exception as e:
            self.logger.error(f"Error shutting down all workers: {e}")
            raise e

    async def start(self):
        """
        Start the autoscaling monitoring task.

        Initializes and starts the background monitoring loop that handles
        automatic scaling decisions. Requires Ray to be initialized.

        Raises:
            RuntimeError: If Ray cluster is not running
            Exception: If autoscaling startup fails
        """
        try:
            if self.is_running:
                self.logger.warning("Autoscaling is already running")
                return

            if not ray.is_initialized():
                raise RuntimeError(
                    "Ray cluster is not running. Please start the Ray cluster first."
                )

            self.is_running = True

            # Create monitoring task
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())

        except Exception as e:
            self.logger.error(f"Failed to start autoscaling: {e}")
            self.is_running = False
            self.monitoring_task = None
            raise e

    async def stop(self):
        """
        Stop the autoscaling monitoring task.

        Gracefully stops the background monitoring loop and cancels any ongoing
        monitoring tasks. Ensures proper cleanup of autoscaling resources.

        Raises:
            Exception: If error occurs during shutdown process
        """
        if not self.is_running:
            self.logger.warning("Autoscaling is not running")
            return

        self.logger.info("Stopping Ray autoscaling")
        self.is_running = False

        try:
            # Cancel monitoring task with timeout
            if self.monitoring_task:
                self.monitoring_task.cancel()
                try:
                    await asyncio.wait_for(self.monitoring_task, timeout=5.0)
                except asyncio.TimeoutError:
                    self.logger.warning("Timeout waiting for monitoring task to cancel")
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    self.logger.error(f"Error while stopping monitoring task: {e}")
                finally:
                    self.monitoring_task = None

            # Close all worker nodes
            await self._close_all()

        except Exception as e:
            self.logger.error(f"Error during autoscaling shutdown: {e}")
            # Ensure monitoring_task is cleared even if there's an error
            self.monitoring_task = None
            raise e

    async def notify(self, delay_s: int = 3) -> None:
        """
        Notify the autoscaling system of a change in cluster state.

        This method is called when the cluster state changes, such as when a new
        task is submitted or a node is added/removed. It triggers a scaling decision
        after a specified delay by adjusting the last scaling decision timestamp.

        Args:
            delay_s: Delay in seconds before triggering scaling decision
        """
        self.logger.info("Notifying autoscaling system of cluster state change")
        self.last_scaling_decision = time.time() - self.check_interval + delay_s
