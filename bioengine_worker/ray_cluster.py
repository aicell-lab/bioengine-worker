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
from typing import Dict, List, Literal, Optional, Tuple, Union

import ray
from ray.util.state import StateApiClient
from ray.util.state.common import (
    DEFAULT_LIMIT,
    DEFAULT_RPC_TIMEOUT,
    ActorState,
    ClusterEventState,
    GetApiOptions,
    JobState,
    ListApiOptions,
    NodeState,
    ObjectState,
    PlacementGroupState,
    PredicateType,
    RuntimeEnvState,
    StateResource,
    SupportedFilterType,
    TaskState,
    WorkerState,
)

from bioengine_worker import __version__
from bioengine_worker.slurm_workers import SlurmWorkers
from bioengine_worker.utils import create_logger, stream_logging_format


class RayCluster:
    """
    Manages Ray cluster lifecycle across different deployment environments.

    This class provides a unified interface for managing Ray clusters in various environments:
    - SLURM-managed HPC systems with automatic worker scaling
    - Single-machine deployments for local development
    - Connection to existing Ray clusters

    The class handles the complete lifecycle from cluster initialization through
    worker management to graceful shutdown, with robust error handling and
    logging throughout. It includes intelligent autoscaling for SLURM environments,
    automatic port allocation to avoid conflicts, and comprehensive monitoring.

    Key Features:
    - Multi-environment support (SLURM, single-machine, connect)
    - Dynamic port allocation with conflict resolution
    - Container-based worker deployment via SLURM with Apptainer/Singularity
    - Automatic scaling based on resource utilization and task demand
    - Ray Serve integration for model serving capabilities
    - Comprehensive cluster monitoring with historical status tracking
    - Graceful shutdown with proper resource cleanup
    - Robust error handling with automatic reconnection

    Attributes:
        mode (str): Deployment mode ('slurm', 'single-machine', 'connect')
        ray_cluster_config (dict): Configuration for Ray head node
        slurm_worker_config (dict): Configuration for SLURM workers (if applicable)
        is_running (bool): Whether the cluster is currently active
        ray_start_time (float): Timestamp when cluster was started
        status_interval_seconds (int): Interval for status monitoring
        max_status_history_length (int): Maximum entries in status history
        cluster_status_history (OrderedDict): Historical status of cluster
        monitoring_task (asyncio.Task): Background monitoring task
        slurm_workers (SlurmWorkers): SLURM worker manager instance
        logger: Logger instance for cluster operations
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
        runtime_env_pip_cache_size_gb: int = 30,  # Ray default is 10 GB
        connection_address: str = None,  # 'ip:port' format
        force_clean_up: bool = True,
        # Cluster Monitoring parameters
        status_interval_seconds: int = 10,
        max_status_history_length: int = 100,
        # SLURM Worker Configuration parameters
        image: str = f"ghcr.io/aicell-lab/bioengine-worker:{__version__}",
        worker_cache_dir: Optional[str] = None,
        worker_data_dir: Optional[str] = None,
        default_num_gpus: int = 1,
        default_num_cpus: int = 8,
        default_mem_per_cpu: int = 16,
        default_time_limit: str = "4:00:00",
        further_slurm_args: Optional[List[str]] = None,
        # Autoscaling configuration parameters
        min_workers: int = 0,
        max_workers: int = 4,
        scale_up_cooldown_seconds: int = 60,
        scale_down_check_interval_seconds: int = 60,
        scale_down_threshold_seconds: int = 300,
        scale_down_cooldown_seconds: int = 60,
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
            runtime_env_pip_cache_size_gb: Size of pip cache for runtime environments in GB. Default 30.
            connection_address: Address to connect to existing cluster ('auto' or 'ip:port').
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
        self.serve_exec_path = (
            self.ray_exec_path[:-3] + "serve"
        )  # Replace 'ray' with 'serve'

        # Check if mode is valid
        self.mode = mode
        if self.mode == "slurm":
            self._check_slurm_available()
        elif self.mode not in ["single-machine", "connect"]:
            raise ValueError(
                f"Invalid mode '{self.mode}'. Supported modes are 'slurm', 'single-machine' and 'connect'."
            )

        # Check number of CPUs and GPUs
        if self.mode in ["slurm", "connect"]:
            if head_num_cpus > 0:
                self.logger.warning(
                    f"Ignoring 'head_num_cpus' setting in '{self.mode}' mode - will be set to 0"
                )
                head_num_cpus = 0
            if head_num_gpus > 0:
                self.logger.warning(
                    f"Ignoring 'head_num_gpus' setting in '{self.mode}' mode - will be set to 0"
                )
                head_num_gpus = 0
        elif (
            self.mode == "single-machine" and head_num_cpus <= 0 and head_num_gpus <= 0
        ):
            raise ValueError(
                "When running on a single machine, either 'head_num_cpus' or 'head_num_gpus' must be greater than 0"
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

        if self.mode == "connect":
            self._parse_connection_address(connection_address)

        # Set runtime environment pip cache size
        if runtime_env_pip_cache_size_gb <= 0:
            raise ValueError("runtime_env_pip_cache_size_gb must be greater than 0")
        os.environ["RAY_RUNTIME_ENV_PIP_CACHE_SIZE_GB"] = str(
            runtime_env_pip_cache_size_gb
        )
        self.logger.debug(
            f"Setting RAY_RUNTIME_ENV_PIP_CACHE_SIZE_GB to {runtime_env_pip_cache_size_gb}"
        )

        if self.mode == "slurm":
            self.slurm_worker_config = {
                "image": image,
                "worker_cache_dir": worker_cache_dir,
                "worker_data_dir": worker_data_dir,
                "default_num_gpus": default_num_gpus,
                "default_num_cpus": default_num_cpus,
                "default_mem_per_cpu": default_mem_per_cpu,
                "default_time_limit": default_time_limit,
                "further_slurm_args": further_slurm_args or [],
                "min_workers": min_workers,
                "max_workers": max_workers,
                "scale_up_cooldown_seconds": scale_up_cooldown_seconds,
                "scale_down_check_interval_seconds": scale_down_check_interval_seconds,
                "scale_down_threshold_seconds": scale_down_threshold_seconds,
                "scale_down_cooldown_seconds": scale_down_cooldown_seconds,
                "log_file": log_file,
                "debug": debug,
            }

        self.cluster_status_history = OrderedDict()
        self.last_cluster_status = 0
        self.max_status_history_length = max_status_history_length
        self.monitoring_task = None
        self.slurm_workers = None
        self.start_time = None
        self.status_interval_seconds = status_interval_seconds

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
            # Choose client server port for remote head node
            port = self.ray_cluster_config["client_server_port"]
        else:
            # Choose GCS server port for local head node
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
        if self.mode == "connect":
            status = {
                "head_address": self.ray_cluster_config["head_node_address"],
                "start_time": "N/A",
            }
        else:
            status = {
                "head_address": self.ray_cluster_config["head_node_address"],
                "start_time": self.start_time,
            }

        last_status = (
            next(reversed(self.cluster_status_history.values()))
            if self.cluster_status_history
            else None
        )
        status["cluster"] = last_status["cluster"] if last_status else {}
        status["worker_nodes"] = last_status["nodes"] if last_status else {}

        return status

    def _check_slurm_available(self) -> None:
        """
        Check if SLURM is available on the system.

        Verifies that the SLURM workload manager is installed and accessible
        by attempting to run the 'sinfo' command.

        Raises:
            RuntimeError: If SLURM is not available or 'sinfo' command fails
        """
        try:
            subprocess.run(["sinfo"], capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            self.logger.error(
                "SLURM is not available. Please ensure you are running this on a SLURM-managed HPC system."
            )
            raise RuntimeError("SLURM is not available") from e

    def _find_ray_executable(self) -> str:
        """
        Find the Ray executable path in the current Python environment.

        Searches for the Ray executable in the Python environment's bin directory
        based on the current Python interpreter location.

        Returns:
            str: Path to the Ray executable

        Raises:
            FileNotFoundError: If Ray executable is not found in the expected location
        """
        ray_path = Path(sys.executable).parent / "ray"
        if not ray_path.exists():
            raise FileNotFoundError("Ray executable not found")
        self.logger.debug(f"Ray executable found at: {ray_path}")
        return str(ray_path)

    def _find_internal_ip(self) -> str:
        """
        Find the internal IP address of the system.

        Uses the hostname command to retrieve the system's internal IP address.

        Returns:
            str: The internal IP address of the system
        """
        result = subprocess.run(["hostname", "-I"], capture_output=True, text=True)
        return result.stdout.strip().split()[0]  # Take the first IP

    def _parse_connection_address(self, connection_string: str) -> None:
        """
        Supports: ray://hostname:port, IPv4:port, [IPv6]:port, hostname:port
        """
        try:
            if "ray://" in connection_string:
                # If the connection string starts with 'ray://', it is a Ray client address
                port = connection_string.split(":")[-1]
                address = connection_string[:-(len(port)+1)]
                self.ray_cluster_config["head_node_address"] = address
                self.ray_cluster_config["client_server_port"] = int(port)
            else:
                # If the connection string is an IP address or hostname, parse it
                address, port = connection_string.split(":")
                self.ray_cluster_config["head_node_address"] = address
                self.ray_cluster_config["head_node_port"] = int(port)
        except (ValueError, IndexError):
            raise ValueError(
                "Invalid connection address format. Use formats like 'ip:port' or 'ray://hostname:port'."
            )
        
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
            # Calls ray.init() when connecting to a Ray client address
            self._state_api_client = await asyncio.to_thread(
                StateApiClient, address=self.head_node_address
            )

            context = await asyncio.to_thread(
                ray.init,
                address=self.head_node_address,
                logging_format=stream_logging_format,
            )
            return context
        except Exception as e:
            self.logger.error(f"Failed to connect to existing Ray cluster: {e}")
            raise e

    async def _find_available_port(self, port: int, step: int = 1) -> int:
        """
        Find next available port starting from given port number.

        Checks for port availability by attempting to bind to the port.
        If the port is in use, it increments by the step value until an
        available port is found.

        Args:
            port: Starting port number to check
            step: Increment between port numbers to check

        Returns:
            First available port number found
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
        autoscaling system for SLURM environments. Handles port allocation,
        directory setup, and symlink management for containerized environments.

        Raises:
            RuntimeError: If Ray is already initialized.
            subprocess.CalledProcessError: If Ray startup command fails.
            Exception: For other initialization errors.
        """
        if self.ray_cluster_config["force_clean_up"]:
            self.logger.info("Forcing Ray cleanup...")
            await self._shutdown_ray()
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
                f"--temp-dir={ray_temp_dir}",
            ]
            if self.mode != "single-machine":
                args.append("--memory=0")  # Disable memory limit for head node in SLURM and connect modes

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

            # Start Ray Serve
            args = [
                "start",
                "--address",
                self.head_node_address,
                "--http-host",
                "0.0.0.0",
                "--http-port",
                str(self.ray_cluster_config["serve_port"]),
            ]
            proc = await asyncio.create_subprocess_exec(
                self.serve_exec_path,
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                raise subprocess.CalledProcessError(
                    proc.returncode, f"serve start", stderr=error_msg
                )

            # Log the start time and head node address
            ray_address = self.head_node_address
            self.logger.info(
                f"Ray cluster with Ray Serve started on '{ray_address}'"
            )

            # Change '<ray_temp_dir>/session_latest' symlink to use relative path instead of absolute (container) path
            # (needed when starting ray in container)
            symlink_path = ray_temp_dir / "session_latest"

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

            # If running on a HPC system, use SlurmWorkers to manage worker nodes
            if self.mode == "slurm":
                # Initialize SlurmWorkers
                self.slurm_workers = SlurmWorkers(
                    ray_cluster=self, **self.slurm_worker_config
                )

        except subprocess.CalledProcessError as e:
            self.logger.error(
                f"Ray start command failed with error code {e.returncode}:\n{e.stderr}"
            )
            raise e
        except Exception as e:
            self.logger.error(f"Error starting Ray cluster: {e}")
            raise e
        
    async def _get_pending_resources(self) -> int:
        """
        Check if there are any pending resources in the cluster.
        """
        pending_actors = await self.list_resources(
            resource=StateResource.ACTORS,
            filters=[("state", "=", "PENDING_CREATION")],
        )
        pending_tasks = await self.list_resources(
            resource=StateResource.TASKS,
            filters=[("state", "=", "PENDING_NODE_ASSIGNMENT")],
        )
        pending_jobs = await self.list_resources(
            resource=StateResource.JOBS,
            # TODO: check if status is correct
            filters=[("status", "=", "PENDING")],
        )
        return len(pending_actors) + len(pending_tasks) + len(pending_jobs)

    async def _get_cluster_status(self) -> Dict[str, Union[str, dict]]:
        """Get current cluster state including head node and worker information.

        Returns a detailed status report including information about all worker nodes
        including their resources and SLURM job IDs if applicable. This method
        excludes the head node unless running in single-machine mode.

        Returns:
            Dict containing worker nodes grouped by state (e.g., 'ALIVE', 'DEAD').
            Each node entry includes:
                - node_id: Unique identifier for the node
                - node_ip: IP address of the worker node
                - total_gpu: GPU resource information
                - available_gpu: Available GPU resource information
                - total_cpu: CPU resource information
                - available_cpu: Available CPU resource information
                - total_memory: Memory resource information
                - available_memory: Available memory resource information

        Raises:
            RuntimeError: If Ray cluster is not initialized.
            Exception: For other cluster status retrieval errors.
        """
        try:
            # Get the status and resources of all nodes (run in thread to avoid blocking)
            # ray.nodes() took on average 0.0012 seconds
            # list_nodes() took on average 0.0074 seconds

            # TODO: use client_mode_hook to access per node cluster resources
            # TODO: set memory of head node to 0 if running in SLURM mode
            cluster_resources = await asyncio.to_thread(ray.cluster_resources)
            available_resources = await asyncio.to_thread(ray.available_resources)
            cluster_status = {
                "cluster": {
                    "total_gpu": cluster_resources.get("GPU", 0),
                    "available_gpu": available_resources.get("GPU", 0),
                    "total_cpu": cluster_resources.get("CPU", 0),
                    "available_cpu": available_resources.get("CPU", 0),
                    "total_memory": cluster_resources.get("memory", 0),
                    "available_memory": available_resources.get("memory", 0),
                    "total_object_store_memory": cluster_resources.get(
                        "object_store_memory", 0
                    ),
                    "available_object_store_memory": available_resources.get(
                        "object_store_memory", 0
                    ),
                },
                "nodes": {},
            }

            # Skip the head node if it is not a worker node
            filters = (
                [("is_head_node", "=", False)]
                if self.mode != "single-machine"
                else None
            )

            all_nodes = await self.list_resources(
                resource=StateResource.NODES, filters=filters
            )
            for node in all_nodes:
                if not node.resources_total:
                    self.logger.warning(
                        f"Encountered worker node without node resources and state '{node.state}'"
                    )
                node_status = {
                    "node_id": node.node_id,
                    "node_ip": node.node_ip,
                    "total_gpu": node.resources_total.get("GPU", 0),
                    "total_cpu": node.resources_total.get("CPU", 0),
                    "total_memory": node.resources_total.get("memory", 0),
                }
                cluster_status["nodes"].setdefault(node.state, []).append(node_status)

            if self.mode == "slurm":
                n_pending_resources = await self._get_pending_resources()
                cluster_status["cluster"]["pending_resources"] = n_pending_resources

            return cluster_status

        except Exception as e:
            self.logger.error(f"Error checking ray cluster: {e}")
            raise e
        
    async def _shutdown_ray(self, grace_period: int = 60) -> None:
        """Stop Ray cluster and all worker nodes.

        Performs a graceful shutdown of the Ray cluster including stopping
        all workers (if running in SLURM mode), disconnecting from the
        cluster, stopping Ray Serve, and canceling any remaining SLURM jobs.

        Args:
            grace_period: Seconds to wait for graceful shutdown

        Raises:
            OSError: If Ray executable is not reachable.
            subprocess.CalledProcessError: If Ray stop command fails.
            Exception: For other shutdown errors.
        """
        try:
            # Disconnect from Ray cluster if it was initialized
            if ray.is_initialized():
                # Disconnect current client from Ray cluster
                self.logger.info("Disconnecting from Ray cluster...")
                await asyncio.to_thread(ray.shutdown)
        
            # Shutdown all SLURM workers if running in SLURM mode
            if self.slurm_workers:
                await self.slurm_workers.close_all()

            # Shutdown the Ray cluster head node if it is not in connect mode
            if self.mode != "connect":
                self.logger.info("Starting shutdown of Ray head node...")
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

    async def _monitoring_task(self, max_consecutive_errors: int = 5) -> None:
        """Continuously monitor cluster status and update worker nodes history.

        This loop runs while the cluster is active, periodically collecting
        cluster status information and updating the history. It handles
        connection errors by attempting to reconnect to the Ray cluster.

        The monitoring task:
        1. Collects cluster status every <status_interval_seconds>
        2. Updates cluster_status_history with timestamped entries
        3. Maintains history size within max_status_history_length
        4. Handles Ray connection errors with automatic reconnection
        5. Gracefully handles task cancellation during shutdown

        Args:
            max_consecutive_errors: Maximum number of consecutive errors before stopping

        Raises:
            Exception: If an unrecoverable error occurs during monitoring.
        """
        self.logger.debug("Starting monitoring task")
        self.start_time = time.time()
        consecutive_errors = 0
        while self.start_time:
            try:
                current_time = time.time()
                if current_time - self.last_cluster_status < self.status_interval_seconds:
                    continue  # Skip if within check interval
                self.last_cluster_status = current_time

                # Check if Ray cluster is initialized and connected
                await self.check_connection()
                # Get the current status of the cluster
                cluster_status = await self._get_cluster_status()
                self.cluster_status_history[time.time()] = cluster_status

                # Limit the history size
                if len(self.cluster_status_history) > self.max_status_history_length:
                    self.cluster_status_history.popitem(last=False)

                # Check if SLURM workers need to scale
                if self.slurm_workers:
                    await self.slurm_workers.check_scaling()
                consecutive_errors = 0  # Reset error counter on success

                
            except Exception as e:
                self.logger.error(f"Error in monitoring task: {e}")
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    self.logger.error(
                        f"Stopping monitoring loop after {consecutive_errors} consecutive errors"
                    )
                    self.start_time = None

                # Don't raise the exception to avoid crashing the monitoring task
                # Instead, wait a bit and continue monitoring

            # Sleep for 1 second before next iteration
            await asyncio.sleep(1)

        self.logger.debug("Monitoring task stopped. Shutting down Ray cluster...")

        # Clear history on shutdown
        self.cluster_status_history.clear()
        self.monitoring_task = None

        # Trigger shutdown of Ray cluster
        await self._shutdown_ray()

    async def check_connection(self) -> None:
        """Check if Ray cluster is initialized and connected.

        Raises:
            RuntimeError: If Ray is not initialized.
        """
        # Initialize Ray if RayCluster is running and Ray is not initialized
        if self.start_time is None:
            raise RuntimeError("Ray cluster is not running")
            
        if not ray.is_initialized():
            self.logger.warning(f"Ray client disconnected. Reconnecting...")
            await self._connect_to_cluster()
            
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
        if self.start_time is not None:
            raise RuntimeError("Ray cluster is already running")
        
        if ray.is_initialized():
            raise RuntimeError(
                "Ray is already initialized. Please stop the existing Ray instance before starting the cluster."
            )

        try:
            if self.mode != "connect":
                await self._start_cluster()

            await self._connect_to_cluster()

            # Start the monitoring task
            self.monitoring_task = asyncio.create_task(
                self._monitoring_task(),
                name="RayClusterMonitoring",
            )
            self.logger.debug(
                f"Monitoring task started with status interval: {self.status_interval_seconds}s"
            )
            self.logger.info("Ray cluster started successfully")

        except Exception as e:
            self.logger.error(f"Error in cluster startup: {e}")
            self.stop()
            raise e

    async def notify(self, delay_s: int = 3) -> None:
        """
        Notify SLURM workers' autoscaling system of a change in cluster state.

        This method triggers the autoscaling system to check for scaling opportunities
        after a specified delay. It's typically called when new tasks are submitted
        or when the cluster state changes in a way that might require scaling.

        Args:
            delay_s: Delay in seconds before triggering scaling decision

        Raises:
            RuntimeError: If SLURM workers are not initialized
        """
        if self.mode != "slurm":
            raise RuntimeError("notify() is only available in SLURM mode")

        if self.slurm_workers:
            self.logger.info("Notifying SLURM workers of cluster state change")
            self.last_cluster_status = time.time() - self.status_interval_seconds + delay_s
        
    async def stop(self) -> None:
        """Stop the Ray cluster"""

        # Trigger end of monitoring task
        self.start_time = None

        # Wait for the monitoring task to finish if it is running
        if self.monitoring_task and not self.monitoring_task.done():
            await self.monitoring_task
        else:
            await self._shutdown_ray()

    async def list_resources(
        self,
        resource: StateResource,
        filters: Optional[List[Tuple[str, PredicateType, SupportedFilterType]]] = None,
        limit: int = DEFAULT_LIMIT,
        timeout: int = DEFAULT_RPC_TIMEOUT,
        detail: bool = False,
        raise_on_missing_output: bool = True,
        _explain: bool = False,
    ) -> List[
        Union[
            ActorState,
            JobState,
            NodeState,
            TaskState,
            ObjectState,
            PlacementGroupState,
            RuntimeEnvState,
            WorkerState,
            ClusterEventState,
        ]
    ]:
        """List resources states

        Args:
            resource: State resource to list, e.g. `StateResource.ACTORS`.
            options: List options. See `ListApiOptions` for details.
            raise_on_missing_output: When True, raise an exception if the output
                is incomplete. Output can be incomplete if
                (1) there's a partial network failure when the source is distributed.
                (2) data is truncated because it is too large.
                Set it to False to avoid throwing an exception on missing data.
            _explain: Print the API information such as API
                latency or failed query information.

        Returns:
            A list of queried result from `ListApiResponse`,

        Raises:
            Exception: This doesn't catch any exceptions raised when the
                underlying request call raises exceptions. For example, it could
                raise `requests.Timeout` when timeout occurs.

        """
        return await asyncio.to_thread(
            self._state_api_client.list,
            resource=resource,
            options=ListApiOptions(
                limit=limit,
                timeout=timeout,
                filters=filters,
                detail=detail,
                explain=_explain,
            ),
            raise_on_missing_output=raise_on_missing_output,
        )

    async def get_resource(
        self,
        resource: StateResource,
        id: str,
        timeout: int = DEFAULT_RPC_TIMEOUT,
        _explain: bool = False,
    ) -> Optional[
        Union[
            ActorState,
            PlacementGroupState,
            NodeState,
            WorkerState,
            TaskState,
            List[ObjectState],
            JobState,
        ]
    ]:
        """Get resources states by id

        Args:
            resource: State resource to get, e.g. `StateResource.Nodes`.
            id: ID for the resource, i.e. 'node_id' for nodes.
            timeout: Timeout for the request in seconds.
            _explain: Print the API information such as API
                latency or failed query information.

        Returns:
            None if not found, and if found:
            - ActorState for actors
            - PlacementGroupState for placement groups
            - NodeState for nodes
            - WorkerState for workers
            - TaskState for tasks
            - JobState for jobs

            Empty list for objects if not found, or list of ObjectState for objects

        Raises:
            Exception: This doesn't catch any exceptions raised when the underlying request
                call raises exceptions. For example, it could raise `requests.Timeout`
                when timeout occurs.
            ValueError:
                if the resource could not be GET by id, i.e. jobs and runtime-envs.

        """
        return await asyncio.to_thread(
            self._state_api_client.get,
            resource=resource,
            id=id,
            options=GetApiOptions(timeout=timeout),
            _explain=_explain,
        )



if __name__ == "__main__":
    import os
    from pathlib import Path

    # TODO: add this to tests

    async def test_ray_cluster_single_machine():
        print("\n===== Testing RayCluster in single-machine mode =====\n")

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
            history = ray_cluster.cluster_status_history
            print("\n=== Worker Nodes History ===\n", history, end="\n\n")

        # Test automatic reconnection
        ray.shutdown()
        await asyncio.sleep(10)

        print("\n=== Cluster status ===\n", ray_cluster.status, end="\n\n")

        await ray_cluster.stop()

    async def test_ray_cluster_slurm():
        print("\n===== Testing RayCluster in SLURM mode =====\n")

        bioengine_cache_dir = Path(os.environ["HOME"]) / ".bioengine"
        bioengine_data_dir = Path(__file__).parent.parent / "data"
        ray_cluster = RayCluster(
            mode="slurm",
            ray_temp_dir=bioengine_cache_dir / "ray",
            status_interval_seconds=3,
            worker_cache_dir=bioengine_cache_dir,
            worker_data_dir=bioengine_data_dir,
            # further_slurm_args=["-C 'thin'"],
            check_interval_seconds=30,
            scale_down_threshold_seconds=15,
            debug=True,
        )
        await ray_cluster.start()

        await asyncio.sleep(5)
        print("\n=== Cluster status ===\n", ray_cluster.status, end="\n\n")

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

            # Check Bioengine cache directory
            assert (
                Path(bioengine_cache_dir)
            ).exists(), "Bioengine cache directory does not exist"
            # Check data directory
            num_files = len(os.listdir("/data"))
            assert num_files, "Data directory is empty"

            time.sleep(1)

            return (
                f"Successfully run a task in a runtime environment on the worker node!"
            )

        # Submit some test tasks
        obj_refs = [test_remote.remote() for _ in range(5)]

        await ray_cluster.notify(delay_s=5)

        results = await asyncio.gather(*obj_refs)
        print("\n=== Test Remote Function Results ===\n", results, end="\n\n")

        print("\n=== Cluster status ===\n", ray_cluster.status, end="\n\n")

        await ray_cluster.stop()

    # Run the tests
    # asyncio.run(test_ray_cluster_single_machine())
    asyncio.run(test_ray_cluster_slurm())
