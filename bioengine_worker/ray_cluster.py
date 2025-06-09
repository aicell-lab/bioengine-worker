import asyncio
import logging
import os
import re
import socket
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import ray
from ray import serve
from ray._private.state import available_resources_per_node  # DeveloperAPI
from ray.util.state import list_nodes

from bioengine_worker import __version__
from bioengine_worker.ray_autoscaler import RayAutoscaler
from bioengine_worker.slurm_workers import SlurmWorkers
from bioengine_worker.utils import create_logger, format_time, stream_logging_format


class RayCluster:
    """Manages Ray cluster lifecycle across different deployment environments.

    This class provides a unified interface for managing Ray clusters in various environments:
    - SLURM-managed HPC systems with automatic worker scaling
    - Single-machine deployments for local development
    - Connection to existing Ray clusters

    Features:
    - Dynamic port allocation to avoid conflicts
    - Container-based worker deployment via SLURM
    - Automatic scaling based on resource utilization
    - Ray Serve integration for model serving
    - Comprehensive cluster monitoring and status reporting

    The class handles the complete lifecycle from cluster initialization through
    worker management to graceful shutdown, with robust error handling and
    logging throughout.
    """

    def __init__(
        self,
        mode: Literal["slurm", "single-machine"] = "slurm",
        # Ray Head Node Configuration parameters
        head_node_address: Optional[str] = None,
        head_node_port: int = 6379,
        node_manager_port: int = 6700,
        object_manager_port: int = 6701,
        redis_shard_port: int = 6702,
        serve_port: int = 8100,
        dashboard_port: int = 8269,
        client_server_port: int = 10001,
        redis_password: Optional[str] = None,
        ray_temp_dir: str = "/tmp/bioengine/ray",
        head_num_cpus: int = 0,
        head_num_gpus: int = 0,
        ray_connection_address: str = "auto",  # 'auto' or 'ip:port' format
        force_clean_up: bool = True,
        # Cluster Monitoring parameters
        status_interval_seconds: int = 15,
        max_status_history_length: int = 100,
        # SLURM Worker Configuration parameters
        image: str = f"ghcr.io/aicell-lab/bioengine-worker:{__version__}",  # BioEngine image tag or path to the image file
        worker_cache_dir: Optional[
            str
        ] = None,  # Cache directory mounted to the container when starting a worker (required for SLURM mode)
        worker_data_dir: Optional[
            str
        ] = None,  # Data directory mounted to the container when starting a worker
        default_num_gpus: int = 1,
        default_num_cpus: int = 8,
        default_mem_per_cpu: int = 16,
        default_time_limit: str = "4:00:00",
        further_slurm_args: Optional[List[str]] = None,
        # Autoscaling configuration parameters
        min_workers: int = 0,
        max_workers: int = 4,
        metrics_interval_seconds: int = 60,  # Higher value to reduce monitoring overhead
        gpu_idle_threshold: float = 0.05,
        cpu_idle_threshold: float = 0.1,
        scale_down_threshold_seconds: int = 300,  # 5 minutes of idleness before scale down
        scale_up_cooldown_seconds: int = 120,  # 2 minutes between scale ups
        scale_down_cooldown_seconds: int = 600,  # 10 minutes between scale downs
        node_grace_period_seconds: int = 600,  # 10 minutes grace period for new nodes
        # Logger configuration
        log_file: Optional[str] = None,
        debug: bool = False,
    ):
        """Initialize cluster manager with networking and resource configurations.

        Ray ports configuration: https://docs.ray.io/en/latest/ray-core/configure.html#ports-configurations
        SLURM networking caveats: https://github.com/ray-project/ray/blob/1000ae9671967994f7bfdf7b1e1399223ad4fc61/doc/source/cluster/vms/user-guides/community/slurm.rst#id22

        Args:
            mode: Mode of operation ('slurm', 'single-machine', or 'connect').
            head_node_address: IP address for head node. Uses first system IP if None.
            head_node_port: Port for Ray head node and GCS server. Default 6379.
            node_manager_port: Base port for Ray node manager services. Default 6700.
            object_manager_port: Port for object manager service. Default 6701.
            redis_shard_port: Port for Redis sharding. Default 6702.
            serve_port: Port for Ray Serve HTTP server. Default 8100.
            dashboard_port: Port for Ray dashboard. Default 8269.
            client_server_port: Base port for Ray client services. Default 10001.
            redis_password: Password for Redis server. Generated randomly if None.
            ray_temp_dir: Temporary directory for Ray. Default '/tmp/bioengine/ray'.
            head_num_cpus: Number of CPUs for head node (single-machine mode). Default 0.
            head_num_gpus: Number of GPUs for head node (single-machine mode). Default 0.
            ray_connection_address: Address to connect to existing cluster ('auto' or 'ip:port').
            force_clean_up: Force cleanup of previous Ray cluster on start. Default True.
            image: Container image for workers (SLURM mode). Default bioengine-worker.
            worker_cache_dir: Cache directory mounted to worker containers (SLURM mode).
            worker_data_dir: Data directory mounted to worker containers (SLURM mode).
            default_num_gpus: Default GPU count per worker. Default 1.
            default_num_cpus: Default CPU count per worker. Default 8.
            default_mem_per_cpu: Default memory per CPU in GB. Default 16.
            default_time_limit: Default SLURM job time limit. Default '4:00:00'.
            further_slurm_args: Additional SLURM arguments for job submission.
            min_workers: Minimum number of workers for autoscaling. Default 0.
            max_workers: Maximum number of workers for autoscaling. Default 4.
            metrics_interval_seconds: Interval for resource monitoring. Default 60.
            gpu_idle_threshold: GPU idle threshold for scaling decisions. Default 0.05.
            cpu_idle_threshold: CPU idle threshold for scaling decisions. Default 0.1.
            scale_down_threshold_seconds: Idle time before scaling down. Default 300.
            scale_up_cooldown_seconds: Cooldown between scale-up operations. Default 120.
            scale_down_cooldown_seconds: Cooldown between scale-down operations. Default 600.
            node_grace_period_seconds: Grace period for new nodes. Default 600.
            log_file: File path for logging output. Uses console if None.
            debug: Enable debug-level logging. Default False.

        Raises:
            ValueError: If mode is invalid or configuration is inconsistent.
            RuntimeError: If SLURM is required but not available.
            FileNotFoundError: If Ray executable is not found.
        """
        # Set up logging
        self.logger = create_logger(
            name="RayCluster",
            level=logging.DEBUG if debug else logging.INFO,
            log_file=log_file,
        )

        # Find and store Ray executable path
        self.ray_exec_path = self._find_ray_executable()

        # Check if mode is valid
        self.mode = mode
        if self.mode == "slurm":
            self._check_slurm_available()
        elif self.mode == "connect":
            if ray_connection_address != "auto":
                try:
                    head_node_address, port = ray_connection_address.split(":")
                    if head_node_address.startswith("ray://"):
                        client_server_port = int(port.strip())
                    else:
                        head_node_port = int(port.strip())
                except ValueError:
                    raise ValueError(
                        "Invalid ray_connection_address format. Use 'ip:port' format."
                    )
        elif self.mode != "single-machine":
            raise ValueError(
                f"Invalid mode '{self.mode}'. Supported modes are 'slurm', 'single-machine' and 'connect'."
            )

        # Check number of CPUs and GPUs
        if self.mode == "slurm":
            if head_num_cpus > 0:
                self.logger.warning(
                    "Ignoring 'head_num_cpus' setting in 'SLURM' mode - will be set to 0"
                )
                head_num_cpus = 0
            if head_num_gpus > 0:
                self.logger.warning(
                    "Ignoring 'head_num_gpus' setting in 'SLURM' mode - will be set to 0"
                )
                head_num_gpus = 0
        elif head_num_cpus <= 0 and head_num_gpus <= 0:
            raise ValueError(
                "When SLURM is not available, either head_num_cpus or head_num_gpus must be greater than 0"
            )

        self.ray_cluster_config = {
            "head_node_address": head_node_address or self._find_internal_ip(),
            "head_node_port": head_node_port,  # GCS server port
            "node_manager_port": node_manager_port,
            "object_manager_port": object_manager_port,
            "redis_shard_port": redis_shard_port,
            "serve_port": serve_port,
            "dashboard_port": dashboard_port,
            "client_server_port": client_server_port,
            "redis_password": redis_password or os.urandom(16).hex(),
            "ray_temp_dir": ray_temp_dir,
            "head_num_cpus": head_num_cpus,
            "head_num_gpus": head_num_gpus,
            "force_clean_up": force_clean_up,
        }

        if self.mode == "slurm":
            # Initialize SlurmWorkers
            # TODO: update
            self.slurm_workers = SlurmWorkers(
                worker_cache_dir=worker_cache_dir,
                worker_data_dir=worker_data_dir,
                image=image,
                head_node_address=self.head_node_address,
                default_num_gpus=default_num_gpus,
                default_num_cpus=default_num_cpus,
                default_mem_per_cpu=default_mem_per_cpu,
                default_time_limit=default_time_limit,
                further_slurm_args=further_slurm_args,
                log_file=log_file,
                debug=debug,
            )

            # Initialize RayAutoscaler
            # TODO: update
            self.autoscaler = RayAutoscaler(
                ray_cluster=self,
                default_num_gpus=default_num_gpus,
                default_num_cpus=default_num_cpus,
                default_mem_per_cpu=default_mem_per_cpu,
                default_time_limit=default_time_limit,
                min_workers=min_workers,
                max_workers=max_workers,
                metrics_interval_seconds=metrics_interval_seconds,
                gpu_idle_threshold=gpu_idle_threshold,
                cpu_idle_threshold=cpu_idle_threshold,
                scale_down_threshold_seconds=scale_down_threshold_seconds,
                scale_up_cooldown_seconds=scale_up_cooldown_seconds,
                scale_down_cooldown_seconds=scale_down_cooldown_seconds,
                node_grace_period_seconds=node_grace_period_seconds,
                log_file=log_file,
                debug=debug,
            )
        else:
            self.slurm_workers = None
            self.autoscaler = None

        self.is_running = False
        self.ray_start_time = None
        self.status_interval_seconds = status_interval_seconds
        self.max_status_history_length = max_status_history_length
        self.worker_nodes_history = OrderedDict()

    @property
    def head_node_address(self) -> str:
        """Get the full address of the Ray head node including port.

        Returns the head node address with the appropriate port based on the
        address format. Uses client server port for ray:// addresses and
        GCS server port for IP addresses.

        Returns:
            str: Complete head node address in format 'ip:port' or 'ray://ip:port'
        """
        head_node_address = str(self.ray_cluster_config["head_node_address"])
        if head_node_address.startswith("ray://"):
            # Choose client server port for head node address
            port = self.ray_cluster_config["client_server_port"]
        else:
            # Choose GCS server port for head node address
            port = self.ray_cluster_config["head_node_port"]
        return f"{head_node_address}:{port}"

    @property
    def status(self) -> Dict[str, Union[str, dict]]:
        """Get current cluster status.

        Returns a dictionary containing the head node address, start time,
        uptime, and the most recent worker nodes status from the monitoring history.

        Returns:
            Dict containing:
                - head_address: Head node IP address
                - start_time_s: Start time as Unix timestamp
                - start_time: Formatted start time string
                - uptime: Human-readable uptime string
                - worker_nodes: Most recent worker nodes status grouped by state
        """
        formatted_time = format_time(self.ray_start_time)
        last_status = (
            next(reversed(self.worker_nodes_history.values()))
            if self.worker_nodes_history
            else None
        )
        status = {
            "head_address": self.ray_cluster_config["head_node_address"],
            "start_time_s": self.ray_start_time,
            "start_time": formatted_time["start_time"],
            "uptime": formatted_time["uptime"],
            "worker_nodes": last_status,
        }
        return status

    def _find_ray_executable(self) -> str:
        """Find the Ray executable path.

        Searches for the Ray executable in the Python environment's bin directory.

        Returns:
            str: Path to the Ray executable

        Raises:
            FileNotFoundError: If Ray executable is not found
        """
        ray_path = Path(sys.executable).parent / "ray"
        if not ray_path.exists():
            raise FileNotFoundError("Ray executable not found")
        self.logger.debug(f"Ray executable found at: {ray_path}")
        return str(ray_path)

    def _check_slurm_available(self) -> None:
        """Check if SLURM is available on the system.

        Raises:
            RuntimeError: If SLURM is not available
        """
        try:
            subprocess.run(["sinfo"], capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            self.logger.error(
                "SLURM is not available. Please ensure you are running this on a SLURM-managed HPC system."
            )
            raise RuntimeError("SLURM is not available") from e

    def _find_internal_ip(self) -> str:
        """Find the internal IP address of the system.

        Returns:
            str: The internal IP address
        """
        result = subprocess.run(["hostname", "-I"], capture_output=True, text=True)
        return result.stdout.strip().split()[0]  # Take the first IP

    async def _find_available_port(self, port: int, step: int = 1) -> int:
        """Find next available port starting from given port number.

        Args:
            port: Starting port number to check
            step: Increment between port numbers to check

        Returns:
            First available port number
        """

        def check_port(port_num):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                return s.connect_ex(("localhost", port_num)) != 0

        available = False
        out_port = port
        while not available:
            available = await asyncio.to_thread(check_port, out_port)
            if not available:
                out_port += step

        if out_port != port:
            self.logger.warning(
                f"Port {port} is not available. Using {out_port} instead."
            )
        return out_port

    async def _set_cluster_ports(self) -> None:
        """Update cluster configuration with available ports.

        Checks for port availability and updates the ray_cluster_config
        dictionary with the next available ports for all Ray services.
        Also sets worker port ranges based on the client server port.
        """
        # Update ports to available ones
        self.ray_cluster_config["head_node_port"] = await self._find_available_port(
            self.ray_cluster_config["head_node_port"], step=1
        )
        self.ray_cluster_config["node_manager_port"] = await self._find_available_port(
            self.ray_cluster_config["node_manager_port"], step=100
        )
        self.ray_cluster_config["object_manager_port"] = (
            await self._find_available_port(
                self.ray_cluster_config["object_manager_port"], step=100
            )
        )
        self.ray_cluster_config["redis_shard_port"] = await self._find_available_port(
            self.ray_cluster_config["redis_shard_port"], step=100
        )
        self.ray_cluster_config["serve_port"] = await self._find_available_port(
            self.ray_cluster_config["serve_port"], step=1
        )
        self.ray_cluster_config["dashboard_port"] = await self._find_available_port(
            self.ray_cluster_config["dashboard_port"], step=1
        )
        self.ray_cluster_config["client_server_port"] = await self._find_available_port(
            self.ray_cluster_config["client_server_port"], step=10000
        )
        self.ray_cluster_config["min_worker_port"] = (
            self.ray_cluster_config["client_server_port"] + 1
        )
        self.ray_cluster_config["max_worker_port"] = (
            self.ray_cluster_config["client_server_port"] + 9998
        )

    async def _start_cluster(self) -> None:
        """Start Ray cluster head node with configured ports and resources.

        Initializes the Ray cluster head node with all configured settings,
        starts Ray Serve for model serving, and optionally starts the
        autoscaler for SLURM environments. Handles port allocation,
        directory setup, and symlink management for containerized environments.

        Raises:
            RuntimeError: If Ray is already initialized.
            subprocess.CalledProcessError: If Ray startup command fails.
            Exception: For other initialization errors.
        """
        if ray.is_initialized():
            raise RuntimeError(
                "Ray is already initialized. Please stop the existing Ray instance before starting the worker."
            )

        force_clean_up = self.ray_cluster_config["force_clean_up"]
        if force_clean_up:
            self.logger.info("Forcing Ray cleanup...")
            await self._shutdown()
        try:
            self.logger.info("Starting Ray cluster...")

            # Check and set cluster ports
            await self._set_cluster_ports()

            # Make sure the temporary directory exists (triggers better error message than Ray)
            ray_temp_dir = Path(self.ray_cluster_config["ray_temp_dir"])
            await asyncio.to_thread(ray_temp_dir.mkdir, parents=True, exist_ok=True)

            # Start ray as the head node with the specified parameters
            args = [
                "start",
                "--head",
                f"--num-cpus={self.ray_cluster_config['head_num_cpus']}",
                f"--num-gpus={self.ray_cluster_config['head_num_gpus']}",
                f"--node-ip-address={self.ray_cluster_config['head_node_address']}",
                f"--port={self.ray_cluster_config['head_node_port']}",
                f"--node-manager-port={self.ray_cluster_config['node_manager_port']}",
                f"--object-manager-port={self.ray_cluster_config['object_manager_port']}",
                f"--redis-shard-ports={self.ray_cluster_config['redis_shard_port']}",
                f"--ray-client-server-port={self.ray_cluster_config['client_server_port']}",
                f"--min-worker-port={self.ray_cluster_config['min_worker_port']}",
                f"--max-worker-port={self.ray_cluster_config['max_worker_port']}",
                "--include-dashboard=True",
                f"--dashboard-port={self.ray_cluster_config['dashboard_port']}",
                f"--redis-password={self.ray_cluster_config['redis_password']}",
                f"--temp-dir={self.ray_cluster_config['ray_temp_dir']}",
            ]
            proc = await asyncio.create_subprocess_exec(
                self.ray_exec_path,
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                raise subprocess.CalledProcessError(
                    proc.returncode, f"ray start", stderr=error_msg
                )

            self.logger.debug(
                f"Ray start command output:\n----------\n{stdout.decode()}"
            )

            # Verify the cluster is running on the correct IP and port
            await self._connect_to_cluster()

            # Start Ray Serve
            await asyncio.to_thread(
                serve.start,
                http_options={
                    "host": "0.0.0.0",
                    "port": self.ray_cluster_config["serve_port"],
                },
            )

            # Log the start time and head node address
            self.ray_start_time = time.time()
            formatted_time = format_time(self.ray_start_time)
            start_time = formatted_time["start_time"]
            ray_address = self.ray_cluster_config["head_node_address"]
            self.logger.info(
                f"Ray cluster with Ray Serve started on '{ray_address}' - Start time: {start_time}"
            )

            # Change '<ray_temp_dir>/session_latest' symlink to use relative path instead of absolute (container) path
            # (needed when starting ray in container)
            symlink_path = (
                Path(self.ray_cluster_config["ray_temp_dir"]) / "session_latest"
            )

            def update_symlink():
                if symlink_path.is_symlink():
                    # Get the target of the symlink
                    symlink_target = symlink_path.readlink()
                    relative_symlink_target = Path(symlink_target.name)
                    self.logger.debug(
                        f"Changing symlink target from '{symlink_target}' to '{relative_symlink_target}'"
                    )
                    symlink_path.unlink()
                    symlink_path.symlink_to(relative_symlink_target)
                else:
                    self.logger.error(f"Symlink '{symlink_path}' does not exist")
                    raise FileNotFoundError(f"Symlink '{symlink_path}' does not exist")

            await asyncio.to_thread(update_symlink)

            # If running on a HPC system, use the RayAutoscaler to manage the Ray cluster
            if self.mode == "slurm":
                await self.autoscaler.start()

        except subprocess.CalledProcessError as e:
            self.logger.error(
                f"Ray start command failed with error code {e.returncode}:\n{e.stderr}"
            )
            if ray.is_initialized():
                await self._shutdown()
            raise e
        except Exception as e:
            self.logger.error(f"Error initializing Ray: {e}")
            if ray.is_initialized():
                await self._shutdown()
            raise e

    async def _connect_to_cluster(self) -> ray.client_builder.ClientContext:
        """Connect to the Ray cluster using the configured head node address.

        Establishes a connection to an existing Ray cluster using the head node
        address. This method is used both for connecting to external clusters
        and for verifying connections after starting a new cluster.

        Returns:
            ray.client_builder.ClientContext: Ray client context for the connected cluster

        Raises:
            RuntimeError: If Ray is already initialized.
            Exception: If connection to the Ray cluster fails.
        """
        try:
            if ray.is_initialized():
                raise RuntimeError(
                    "Ray is already initialized. Please stop the existing Ray client before connecting to the cluster."
                )
            context = await asyncio.to_thread(
                ray.init,
                address=self.head_node_address,
                logging_format=stream_logging_format,
            )
        except Exception as e:
            self.logger.error(f"Failed to connect to existing Ray cluster: {e}")
            raise e
        return context

    async def _get_nodes_status(self) -> Dict[str, Union[str, dict]]:
        """Get current cluster state including head node and worker information.

        Returns a detailed status report including information about all worker nodes 
        including their resources and SLURM job IDs if applicable. This method
        excludes the head node unless running in single-machine mode.

        Returns:
            Dict containing worker nodes grouped by state (e.g., 'ALIVE', 'DEAD').
            Each node entry includes:
                - Node ID: Unique identifier for the node
                - Node IP: IP address of the worker node  
                - SLURM Job ID: Job ID if running in SLURM mode, 'N/A' otherwise
                - Total/Available GPU: GPU resource information
                - Total/Available CPU: CPU resource information
                - Total/Available Memory: Memory resource information

        Raises:
            RuntimeError: If Ray cluster is not initialized.
            Exception: For other cluster status retrieval errors.
        """
        try:
            # Get the status and resources of all nodes (run in thread to avoid blocking)
            all_nodes = await asyncio.to_thread(list_nodes)
            available_resources = await asyncio.to_thread(available_resources_per_node)

            # In SLURM mode, get all running jobs
            if self.mode == "slurm":
                # TODO: update
                running_jobs = await self.slurm_workers.get_running_jobs()

            nodes_status = {}
            for node in all_nodes:
                # Skip the head node if it is not a worker node
                if self.mode != "single-machine" and node.is_head_node:
                    continue

                if self.mode == "slurm":
                    if not node.resources_total:
                        self.logger.warning(
                            f"Encountered worker node without node resources and state '{node.state}'"
                        )
                        job_id = None
                    else:
                        # Get SLURM job ID from resources
                        job_id = self.slurm_workers._get_job_id(node.resources_total)

                        # Skip nodes if job is not running anymore
                        if job_id not in running_jobs:
                            self.logger.warning(
                                f"Skipping worker node '{node.node_id}' with already cancelled job ID '{job_id}'"
                            )
                            continue
                else:
                    job_id = "N/A"  # Not applicable

                # Get available node resources
                available_node_resources = available_resources.get(node.node_id, {})

                node_info = {
                    "Node ID": node.node_id,
                    "Node IP": node.node_ip,
                    "SLURM Job ID": job_id,
                    "Total GPU": node.resources_total.get("GPU", 0),
                    "Available GPU": available_node_resources.get("GPU", 0),
                    "Total CPU": node.resources_total.get("CPU", 0),
                    "Available CPU": available_node_resources.get("CPU", 0),
                    "Total Memory": node.resources_total.get("memory", 0),
                    "Available Memory": available_node_resources.get("memory", 0),
                }
                nodes_status.setdefault(node.state, []).append(node_info)

            return nodes_status

        except Exception as e:
            self.logger.error(f"Error checking ray cluster: {e}")
            raise e

    async def _shutdown(self, grace_period: int = 30) -> None:
        """Stop Ray cluster and all worker nodes.

        Performs a graceful shutdown of the Ray cluster including stopping
        the autoscaler (if running in SLURM mode), disconnecting from the
        cluster, stopping Ray Serve, and canceling any remaining SLURM jobs.

        Args:
            grace_period: Seconds to wait for graceful shutdown

        Raises:
            OSError: If Ray executable is not reachable.
            subprocess.CalledProcessError: If Ray stop command fails.
            Exception: For other shutdown errors.
        """
        try:
            if self.mode == "slurm":
                await self.autoscaler.stop()

            # Disconnect from Ray cluster if it was initialized
            if ray.is_initialized():

                # Stop Ray Serve if it was initialized
                try:
                    await asyncio.to_thread(serve.context._get_global_client)
                    self.logger.info("Shutting down Ray Serve...")
                    await asyncio.to_thread(serve.shutdown)
                except serve.exceptions.RayServeException:
                    # Ray Serve was not initialized, ignore the error
                    pass

                # Disconnect current client from Ray cluster
                self.logger.info("Disconnecting from Ray cluster...")
                await asyncio.to_thread(ray.shutdown)

            # Shutdown the Ray cluster head node
            self.logger.info("Shutting down Ray cluster...")
            proc = await asyncio.create_subprocess_exec(
                self.ray_exec_path,
                "stop",
                f"--grace-period={grace_period}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                raise subprocess.CalledProcessError(
                    proc.returncode, "ray stop", stderr=error_msg
                )

            output = stdout.decode()
            if re.search(r"Stopped all \d+ Ray processes\.", output):
                self.logger.info("All Ray processes stopped successfully")
            elif "Did not find any active Ray processes." in output:
                self.logger.info("No active Ray processes found")
            else:
                message = re.search(
                    r"Stopped only (\d+) out of (\d+) Ray processes within the grace period (\d+) seconds\.",
                    output,
                )
                if message:
                    self.logger.warning(
                        f"Some Ray processes could not be stopped: {message.group(0)}"
                    )
                else:
                    self.logger.warning(
                        f"Unknown message during Ray shutdown:\n----------\n{output}"
                    )

            # Clean up any remaining worker jobs
            await asyncio.sleep(5)  # Wait for Ray to fully shut down
            if self.mode == "slurm":
                # TODO: update
                await self.slurm_workers.cancel_jobs(grace_period=grace_period)

            self.logger.info("Ray cluster shut down complete")

            self.ray_start_time = None

        except OSError as e:
            if e.errno == 107 and os.environ.get("APPTAINER_BIND", None) is not None:
                self.logger.warning(
                    "Ray executable is not reachable. This may be due to the container's overlay filesystem being torn down."
                )
                return
            else:
                self.logger.error(f"Error shutting down Ray cluster: {e}")
                raise e

        except Exception as e:
            self.logger.error(f"Error shutting down Ray cluster: {e}")
            raise e

    async def _monitoring_loop(self) -> None:
        """Continuously monitor cluster status and update worker nodes history.

        This loop runs while the cluster is active, periodically collecting
        cluster status information and updating the history. It handles
        connection errors by attempting to reconnect to the Ray cluster.

        The monitoring loop:
        1. Collects cluster status every status_interval_seconds
        2. Updates worker_nodes_history with timestamped entries
        3. Maintains history size within max_status_history_length
        4. Handles Ray connection errors with automatic reconnection
        5. Gracefully handles task cancellation during shutdown

        Raises:
            Exception: If an unrecoverable error occurs during monitoring.
        """
        self.logger.debug("Starting monitoring loop")
        while self.is_running:
            try:
                # Get the current status of the cluster
                nodes_status = await self._get_nodes_status()
                self.worker_nodes_history[time.time()] = nodes_status
                if len(self.worker_nodes_history) > self.max_status_history_length:
                    self.worker_nodes_history.popitem(last=False)
                await asyncio.sleep(self.status_interval_seconds)
            except asyncio.CancelledError:
                # Handle task cancellation gracefully
                self.logger.debug("Monitoring loop cancelled")
                break
            except Exception as e:
                # Handle specific Ray connection errors
                error_message = str(e)
                if (
                    isinstance(e, ConnectionError)
                    and "Could not find any running Ray instance" in error_message
                ) or (
                    isinstance(e, ray.exceptions.RaySystemError)
                    and "Ray has not been started yet" in error_message
                ):
                    self.logger.warning(f"Ray client disconnected: {e}")
                    # Make sure to shutdown and reconnect
                    if ray.is_initialized():
                        await asyncio.to_thread(ray.shutdown)
                    self.logger.info("Reconnecting to Ray cluster...")
                    try:
                        await self._connect_to_cluster()
                    except Exception as reconnect_error:
                        self.logger.error(
                            f"Failed to reconnect to Ray cluster: {reconnect_error}"
                        )
                        self.is_running = False
                        break
                else:
                    self.logger.error(f"Error in monitoring loop: {e}")
                    # Don't raise the exception to avoid crashing the monitoring loop
                    # Instead, wait a bit and continue monitoring
                    await asyncio.sleep(self.status_interval_seconds)
        
        self.logger.debug("Monitoring loop stopped")

    async def start(self) -> None:
        """Start the Ray cluster based on the configured mode.

        Depending on the mode, this method will either start a new Ray cluster
        head node, connect to an existing cluster, or perform no action if already
        connected. After starting the cluster, it launches a background monitoring
        task to continuously track cluster status.

        Raises:
            RuntimeError: If the cluster is already running.
            Exception: For other startup errors.
        """
        if self.is_running:
            raise RuntimeError("Ray cluster is already running")

        self.is_running = True
        self.monitoring_task = None

        try:
            if self.mode == "connect":
                await self._connect_to_cluster()
            else:
                await self._start_cluster()

            # Start the monitoring loop
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            self.logger.debug(f"Monitoring task started with status interval: {self.status_interval_seconds}s")
            self.logger.info("Ray cluster started successfully")

        except Exception as e:
            self.logger.error(f"Error in cluster startup: {e}")
            self.is_running = False
            # Clean up on startup failure
            if self.monitoring_task and not self.monitoring_task.done():
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass
            self.monitoring_task = None
            raise e

    async def stop(self) -> None:
        """Stop the Ray cluster and all worker nodes.

        This method will stop the Ray cluster, disconnect from it, and clean up
        any resources used by the cluster. It will also stop the autoscaler if
        running in SLURM mode and cancel the monitoring task.

        The shutdown process:
        1. Sets is_running to False to stop monitoring loop
        2. Cancels and waits for monitoring task completion
        3. Calls _shutdown() to clean up Ray cluster and resources

        Raises:
            Exception: If an error occurs during shutdown.
        """
        if not self.is_running:
            self.logger.warning("Ray cluster is not running")
            return

        self.is_running = False

        # Cancel monitoring task
        if (
            hasattr(self, "monitoring_task")
            and self.monitoring_task
            and not self.monitoring_task.done()
        ):
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        # Shutdown the cluster
        await self._shutdown()


if __name__ == "__main__":
    import os
    from pathlib import Path

    # TODO: add this to tests

    async def test_ray_cluster_single_machine():
        print("\n===== Testing RayCluster manager in single-machine mode =====\n")

        ray_cluster = RayCluster(
            mode="single-machine",
            head_num_cpus=1,
            head_num_gpus=0,
            ray_temp_dir=Path(os.environ["HOME"]) / ".bioengine" / "ray",
            status_interval_seconds=3,
            debug=True,
        )
        await ray_cluster.start()
        for _ in range(5):
            await asyncio.sleep(3)
            history = ray_cluster.worker_nodes_history
            print("\n=== Worker Nodes History ===\n", history, end="\n\n")

        print("\n=== Cluster status ===\n", ray_cluster.status, end="\n\n")
        
        await ray_cluster.stop()

    async def test_ray_cluster_slurm():
        print("\n===== Testing RayCluster in SLURM mode =====\n")

        ray_cluster = RayCluster(
            mode="slurm",
            ray_temp_dir=Path(os.environ["HOME"]) / ".bioengine" / "ray",
            image=str(
                Path(__file__).parent.parent
                / ".bioengine"
                / "apptainer_images"
                / f"bioengine-worker_{__version__}.sif"
            ),
            worker_cache_dir=str(Path(__file__).parent.parent / ".bioengine"),
            # further_slurm_args=["-C 'thin'"]
            debug=True,
        )
        await ray_cluster.start_cluster(force_clean_up=True)
        status = await ray_cluster.get_status()
        print("\n=== Cluster status ===\n", status, end="\n\n")

        job_id = await ray_cluster.slurm_workers.add_worker(
            time_limit="00:30:00",
            num_cpus=1,
            num_gpus=0,
        )

        status = await ray_cluster.get_status()
        print("\n=== Cluster status ===\n", status, end="\n\n")

        # Test running a remote function on worker node
        @ray.remote(
            num_cpus=1,
            num_gpus=1,
            runtime_env={"pip": ["pandas"]},
        )
        def test_remote():
            import os
            import time

            # Check if runtime environment is set up correctly
            import pandas as pd

            # # Check Ray's temporary directory
            # assert (
            #     Path(__file__).parent.parent / ".bioengine" / "ray"
            # ).exists(), "Temporary directory does not exist"
            # Check data directory
            # num_files = len(os.listdir("/data"))
            # assert num_files, "Data directory is empty"

            time.sleep(1)

            return f"Successfully run a task in runtime environment on the worker node!"

        obj_ref = test_remote.remote()
        print(ray.get(obj_ref))

        # Test closing a worker node
        ray_cluster.remove_worker(job_id)

        await ray_cluster._shutdown()

    # Run the tests
    asyncio.run(test_ray_cluster_single_machine())
    # asyncio.run(test_ray_cluster_slurm())

    # # Test submitting a worker job
    # print("Adding a worker...")
    # worker_id = ray_cluster.add_worker(time_limit="00:30:00")

    # # Wait for worker to start
    # status = ""
    # while status != "RUNNING":
    #     # Wait for worker node to appear in cluster status
    #     print("Waiting for job to start...")
    #     time.sleep(3)
    #     jobs = ray_cluster.slurm_workers.get_jobs()
    #     if worker_id not in jobs:
    #         raise RuntimeError(
    #             f"Job died before worker node appeared in cluster status"
    #         )
    #     status = jobs[worker_id]["state"]

    # while status != "alive":
    #     print("Waiting for worker node to start...")
    #     time.sleep(3)
    #     status = ray_cluster._get_worker_status(worker_id)

    # # Test cluster status
    # cluster_status = ray_cluster.get_status()
    # print("\n=== Cluster status ===\n", cluster_status, end="\n\n")

    # # Test running a remote function on worker node
    # @ray.remote(
    #     num_cpus=1,
    #     num_gpus=1,
    #     # runtime_env={"pip": ["hypha-rpc"]},
    # )
    # def test_remote():
    #     import os
    #     import time

    #     # Check if runtime environment is set up correctly
    #     from hypha_rpc.sync import connect_to_server

    #     # Check Ray's temporary directory
    #     assert (
    #         Path(__file__).parent.parent / ".bioengine" / "ray"
    #     ).exists(), "Temporary directory does not exist"

    #     # Check data directory
    #     num_files = len(os.listdir("/data"))
    #     assert num_files, "Data directory is empty"

    #     time.sleep(1)

    #     return f"Successfully run a task in runtime environment on the worker node! (Number of files in data directory: {num_files})"

    # obj_ref = test_remote.remote()
    # print(ray.get(obj_ref))

    # # Test closing a worker node
    # ray_cluster.remove_worker(worker_id)

    # # Test shutting down Ray cluster
    # ray_cluster._shutdown()
