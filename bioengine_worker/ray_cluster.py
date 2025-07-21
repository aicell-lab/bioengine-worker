import asyncio
import fcntl
import logging
import os
import re
import socket
import subprocess
import sys
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import ray
from ray import serve
from ray.serve.exceptions import RayServeException

from bioengine_worker import __version__
from bioengine_worker.ray_cluster_state import ClusterState
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
    - Multi-environment support (SLURM, single-machine, external-cluster)
    - Dynamic port allocation with conflict resolution
    - Container-based worker deployment via SLURM with Apptainer/Singularity
    - Automatic scaling based on resource utilization and task demand
    - Ray Serve integration for model serving capabilities
    - Comprehensive cluster monitoring with historical status tracking
    - Graceful shutdown with proper resource cleanup
    - Robust error handling with automatic reconnection

    Attributes:
        mode (str): Deployment mode ('slurm', 'single-machine', 'external-cluster')
        ray_cluster_config (dict): Configuration for Ray head node
        slurm_worker_config (dict): Configuration for SLURM workers (if applicable)
        is_ready (asyncio.Event): Whether the cluster is currently active
        start_time (float): Timestamp when cluster was started
        max_status_history_length (int): Maximum entries in status history
        cluster_status_history (OrderedDict): Historical status of cluster
        slurm_workers (SlurmWorkers): SLURM worker manager instance
        logger: Logger instance for cluster operations
    """

    def __init__(
        self,
        mode: Literal["slurm", "single-machine", "external-cluster"] = "slurm",
        # Ray Head Node Configuration parameters
        head_node_address: Optional[str] = None,
        head_node_port: int = 6379,
        node_manager_port: int = 6700,
        object_manager_port: int = 6701,
        redis_shard_port: int = 6702,
        serve_port: int = 8000,
        dashboard_port: int = 8265,
        client_server_port: int = 10001,
        redis_password: Optional[str] = None,
        ray_temp_dir: str = "/tmp/bioengine/ray",
        head_num_cpus: int = 0,
        head_num_gpus: int = 0,
        head_memory_in_gb: int = 0,  # Only memory limit
        runtime_env_pip_cache_size_gb: int = 30,  # Ray default is 10 GB
        force_clean_up: bool = True,
        # SLURM Worker Configuration parameters
        image: str = f"ghcr.io/aicell-lab/bioengine-worker:{__version__}",
        worker_cache_dir: Optional[str] = None,
        worker_data_dir: Optional[str] = None,
        default_num_gpus: int = 1,
        default_num_cpus: int = 8,
        default_mem_in_gb_per_cpu: int = 16,
        default_time_limit: str = "4:00:00",
        further_slurm_args: Optional[List[str]] = None,
        # Autoscaling configuration parameters
        min_workers: int = 0,
        max_workers: int = 4,
        scale_up_cooldown_seconds: int = 60,
        scale_down_check_interval_seconds: int = 60,
        scale_down_threshold_seconds: int = 300,
        # Logger configuration
        log_file: Optional[str] = None,
        debug: bool = False,
    ):
        """Initialize cluster manager with networking and resource configurations.

        Ray ports configuration: https://docs.ray.io/en/latest/ray-core/configure.html#ports-configurations
        SLURM networking caveats: https://github.com/ray-project/ray/blob/1000ae9671967994f7bfdf7b1e1399223ad4fc61/doc/source/cluster/vms/user-guides/community/slurm.rst#id22

        Args:
            mode: Mode of operation ('slurm', 'single-machine', or 'external-cluster').
            head_node_address: IP address for head node. Uses first system IP if None.
            head_node_port: Port for Ray head node and GCS server. Default 6379.
            node_manager_port: Base port for Ray node manager services. Default 6700.
            object_manager_port: Port for object manager service. Default 6701.
            redis_shard_port: Port for Redis sharding. Default 6702.
            serve_port: Port for Ray Serve HTTP server. Default 8000.
            dashboard_port: Port for Ray dashboard. Default 8265.
            client_server_port: Base port for Ray client services. Default 10001.
            redis_password: Password for Redis server. Generated randomly if None.
            ray_temp_dir: Temporary directory for Ray. Default '/tmp/bioengine/ray'.
            head_num_cpus: Number of CPUs for head node (single-machine mode). Default 0.
            head_num_gpus: Number of GPUs for head node (single-machine mode). Default 0.
            runtime_env_pip_cache_size_gb: Size of pip cache for runtime environments in GB. Default 30.
            force_clean_up: Force cleanup of previous Ray cluster on start. Default True.
            image: Container image for workers (SLURM mode). Default bioengine-worker.
            worker_cache_dir: Cache directory mounted to worker containers (SLURM mode).
            worker_data_dir: Data directory mounted to worker containers (SLURM mode).
            default_num_gpus: Default GPU count per worker. Default 1.
            default_num_cpus: Default CPU count per worker. Default 8.
            default_mem_in_gb_per_cpu: Default memory per CPU in GB. Default 16.
            default_time_limit: Default SLURM job time limit. Default '4:00:00'.
            further_slurm_args: Additional SLURM arguments for job submission.
            min_workers: Minimum number of workers for autoscaling. Default 0.
            max_workers: Maximum number of workers for autoscaling. Default 4.
            scale_up_cooldown_seconds: Cooldown between scale-up operations. Default 60.
            scale_down_check_interval_seconds: Interval between scale-down checks. Default 60.
            scale_down_threshold_seconds: Idle time before scaling down. Default 300.
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
        elif self.mode not in ["single-machine", "external-cluster"]:
            raise ValueError(
                f"Unsupported Ray cluster mode: '{self.mode}'. "
                "Supported modes are 'slurm', 'single-machine' and 'external-cluster'."
            )

        # Check number of CPUs and GPUs
        if self.mode == "single-machine" and head_num_cpus <= 0:
            raise ValueError(
                "When running on a single machine, 'head_num_cpus' must be greater than 0"
            )

        # Check if Ray temp dir does not exceed length of 107 bytes
        self._check_ray_temp_dir_length(ray_temp_dir)

        self.ray_cluster_config = {
            "head_node_address": str(head_node_address or self._find_internal_ip()),
            "head_node_port": int(head_node_port),  # GCS server port
            "node_manager_port": int(node_manager_port),
            "object_manager_port": int(object_manager_port),
            "redis_shard_port": int(redis_shard_port),
            "serve_port": int(serve_port),
            "dashboard_port": int(dashboard_port),
            "client_server_port": int(client_server_port),
            "redis_password": str(redis_password or os.urandom(16).hex()),
            "ray_temp_dir": str(ray_temp_dir),
            "head_num_cpus": int(head_num_cpus),
            "head_num_gpus": int(head_num_gpus),
            "head_memory_in_gb": int(head_memory_in_gb),
            "force_clean_up": bool(force_clean_up),
        }

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
            if worker_cache_dir is None:
                raise ValueError(
                    "worker_cache_dir must be provided when mode is 'slurm'"
                )
            self.slurm_worker_config = {
                "image": image,
                "worker_cache_dir": str(worker_cache_dir),
                "worker_data_dir": str(worker_data_dir) if worker_data_dir else None,
                "default_num_gpus": int(default_num_gpus),
                "default_num_cpus": int(default_num_cpus),
                "default_mem_in_gb_per_cpu": int(default_mem_in_gb_per_cpu),
                "default_time_limit": str(default_time_limit),
                "further_slurm_args": further_slurm_args or [],
                "min_workers": int(min_workers),
                "max_workers": int(max_workers),
                "scale_up_cooldown_seconds": int(scale_up_cooldown_seconds),
                "scale_down_check_interval_seconds": int(
                    scale_down_check_interval_seconds
                ),
                "scale_down_threshold_seconds": int(scale_down_threshold_seconds),
                "log_file": log_file,
                "debug": bool(debug),
            }

        # Initialize cluster state and monitoring attributes
        self.is_ready = asyncio.Event()
        self.start_time = None
        self.head_node_address = None
        self.serve_http_url = None

        self.proxy_handle = None
        self.cluster_status_history = OrderedDict()
        self.max_status_history_length = 100

        self.slurm_workers = None

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
                - nodes: Most recent worker nodes status grouped by state
        """
        status = {
            "head_address": self.head_node_address,
            "start_time": self.start_time if self.mode != "external-cluster" else "N/A",
            "mode": self.mode,
        }

        last_status = (
            next(reversed(self.cluster_status_history.values()))
            if self.cluster_status_history
            else None
        )
        status["cluster"] = last_status["cluster"] if last_status else {}
        status["nodes"] = last_status["nodes"] if last_status else {}

        return status

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

    def _check_ray_temp_dir_length(self, ray_temp_dir: str) -> None:
        # Simulate the longest possible session directory name
        session_name = datetime.now().strftime(
            "session_%Y-%m-%d_%H-%M-%S_999999_9999999"
        )
        full_path = str(Path(ray_temp_dir) / session_name / "sockets" / "plasma_store")
        path_length = len(full_path.encode("utf-8"))

        if path_length > 107:
            self.logger.debug(f"Simulated path: {full_path}")
            self.logger.debug(f"Path length in bytes: {path_length}")
            raise ValueError(
                "Plasma store path length would exceed 107 bytes (current length: "
                f"{path_length}) with the specified Ray temp directory configuration "
                f"'{ray_temp_dir}'. Please choose a shorter path for the Ray temp directory."
            )

    def _find_internal_ip(self) -> str:
        """
        Find the internal IP address of the system.

        Uses the hostname command to retrieve the system's internal IP address.

        Returns:
            str: The internal IP address of the system
        """
        result = subprocess.run(["hostname", "-I"], capture_output=True, text=True)
        return result.stdout.strip().split()[0]  # Take the first IP

    def _find_available_port(self, port: int, step: int = 1) -> int:
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
            available = check_port(out_port)
            if not available:
                out_port += step

        if out_port != port:
            self.logger.warning(
                f"Port {port} is not available. Using {out_port} instead."
            )
        return out_port

    def _set_cluster_ports(self) -> None:
        """
        Update cluster configuration with available ports.

        Checks for port availability and updates the ray_cluster_config
        dictionary with the next available ports for all Ray services.
        Also sets worker port ranges based on the client server port.
        """
        # Update ports to available ones
        self.ray_cluster_config["head_node_port"] = self._find_available_port(
            self.ray_cluster_config["head_node_port"], step=1
        )
        self.ray_cluster_config["node_manager_port"] = self._find_available_port(
            self.ray_cluster_config["node_manager_port"], step=100
        )
        self.ray_cluster_config["object_manager_port"] = self._find_available_port(
            self.ray_cluster_config["object_manager_port"], step=100
        )
        self.ray_cluster_config["redis_shard_port"] = self._find_available_port(
            self.ray_cluster_config["redis_shard_port"], step=100
        )
        self.ray_cluster_config["serve_port"] = self._find_available_port(
            self.ray_cluster_config["serve_port"], step=1
        )
        self.ray_cluster_config["dashboard_port"] = self._find_available_port(
            self.ray_cluster_config["dashboard_port"], step=1
        )
        self.ray_cluster_config["client_server_port"] = self._find_available_port(
            self.ray_cluster_config["client_server_port"], step=10000
        )
        self.ray_cluster_config["min_worker_port"] = (
            self.ray_cluster_config["client_server_port"] + 1
        )
        self.ray_cluster_config["max_worker_port"] = (
            self.ray_cluster_config["client_server_port"] + 9998
        )

    def _check_ray_session_is_active(self, ray_temp_dir: Path) -> None:
        """
        Check if a Ray session is already active in the specified temporary directory.

        Raises an error if a Ray session is found to be active, preventing
        multiple Ray clusters from being started in the same directory.
        """
        session_latest = ray_temp_dir / "session_latest"
        lock_file = session_latest / "node_ip_address.json.lock"

        if session_latest.exists() and lock_file.exists():
            try:
                fd = os.open(lock_file, os.O_RDWR)
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                os.close(fd)
                # Lock acquired, no Ray process is holding it
                return
            except BlockingIOError:
                # Lock not acquired, Ray likely running
                raise RuntimeError(
                    f"Ray session is already active in '{ray_temp_dir}'. "
                    "Please stop the existing Ray cluster before starting a new one."
                )

    def _update_symlink(self, ray_temp_dir: Path) -> None:
        """
        Update the 'session_latest' symlink to use a relative path.

        Raises an error if the symlink does not exist or is not a symlink.
        """
        symlink_path = ray_temp_dir / "session_latest"
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
            try:
                await self._shutdown_head_node()
            except Exception as e:
                self.logger.error(f"Error during Ray cleanup: {e}")
        try:
            self.logger.info("Starting Ray cluster...")

            # Check and set cluster ports
            await asyncio.to_thread(self._set_cluster_ports)

            # Make sure the temporary directory exists (triggers better error message than Ray)
            ray_temp_dir = Path(self.ray_cluster_config["ray_temp_dir"])
            await asyncio.to_thread(ray_temp_dir.mkdir, parents=True, exist_ok=True)

            # Check if a Ray cluster is already running using this temporary directory
            self._check_ray_session_is_active(ray_temp_dir)

            # Start ray as the head node with the specified parameters
            args = [
                "start",
                "--head",
                f"--num-cpus={self.ray_cluster_config['head_num_cpus']}",
                f"--num-gpus={self.ray_cluster_config['head_num_gpus']}",
                f"--memory={self.ray_cluster_config['head_memory_in_gb'] * 1024**3}",
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

            # Prevent logging of Redis password in debug logs
            censored_args = [
                arg if "redis-password" not in arg else "--redis-password=****"
                for arg in args
            ]
            self.logger.debug(
                f"Ray start command: {self.ray_exec_path} {' '.join(censored_args)}"
            )

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
            ray_address = f"{self.ray_cluster_config['head_node_address']}:{self.ray_cluster_config['head_node_port']}"
            args = [
                "start",
                "--address",
                ray_address,
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

            # Change '<ray_temp_dir>/session_latest' symlink to use relative path instead of absolute (container) path
            # (needed when starting ray in container)
            await asyncio.to_thread(self._update_symlink, ray_temp_dir)

            # If running on a HPC system, use SlurmWorkers to manage worker nodes
            if self.mode == "slurm":
                # Initialize SlurmWorkers
                self.slurm_workers = SlurmWorkers(
                    ray_cluster=self, **self.slurm_worker_config
                )

            self.logger.info(f"Ray cluster with Ray Serve started on '{ray_address}'")

        except subprocess.CalledProcessError as e:
            self.logger.error(
                f"Ray start command failed with error code {e.returncode}:\n{e.stderr}"
            )
            raise e
        except Exception as e:
            self.logger.error(f"Error starting Ray cluster: {e}")
            raise e

    def _set_head_node_address(self) -> None:
        """Set the head node address based on the cluster configuration."""
        head_node_address = str(self.ray_cluster_config["head_node_address"])
        if head_node_address.startswith("ray://"):
            # Choose client server port for remote head node
            port = self.ray_cluster_config["client_server_port"]
        else:
            # Choose GCS server port for local head node
            port = self.ray_cluster_config["head_node_port"]
        self.head_node_address = f"{head_node_address}:{port}"
        self.logger.debug(f"Head node address set to: {self.head_node_address}")

    def _set_serve_http_url(self) -> None:
        """Set the Ray Serve HTTP API base URL based on the head node address and port."""
        address = self.ray_cluster_config["head_node_address"].split("://")[-1]
        self.serve_http_url = (
            f"http://{address}:{self.ray_cluster_config['serve_port']}"
        )
        self.logger.debug(f"Ray Serve HTTP URL set to: {self.serve_http_url}")

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
            # Connect to the Ray cluster
            context = await asyncio.to_thread(
                ray.init,
                address=self.head_node_address,
                logging_format=stream_logging_format,
            )

            # Create BioEngineProxy to access cluster state
            # TODO: Check 'exclude_head_node' setting when head cpus and gpus are set
            exclude_head_node = self.mode == "slurm"
            check_pending_resources = self.mode == "slurm"

            cluster_state_app = ClusterState.bind(
                exclude_head_node=exclude_head_node,
                check_pending_resources=check_pending_resources,
            )
            self.proxy_handle = await asyncio.to_thread(
                serve.run,
                target=cluster_state_app,
                name="BioEngineProxy",
                route_prefix=None,
                blocking=False,
            )

            return context
        except Exception as e:
            self.logger.error(f"Failed to connect to existing Ray cluster: {e}")
            raise e

    async def _shutdown_head_node(self, grace_period: int = 30) -> None:
        """Shutdown the Ray cluster head node.

        Args:
            grace_period: Grace period in seconds for graceful shutdown.
                         If negative, performs force shutdown instead.
        """
        self.logger.info(
            f"Starting graceful shutdown of Ray head node (grace period: {grace_period}s)..."
        )
        args = [self.ray_exec_path, "stop", "--grace-period", str(grace_period)]

        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            error_msg = stderr.decode() if stderr else "Unknown error"
            self.logger.error(
                f"Ray stop command failed with error code {proc.returncode}:\n{error_msg}"
            )
            raise subprocess.CalledProcessError(
                proc.returncode, " ".join(args), stderr=error_msg
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

    async def check_connection(self) -> None:
        """Check if Ray cluster is initialized and connected.

        Raises:
            RuntimeError: If Ray is not initialized.
        """
        # Initialize Ray if RayCluster is running and Ray is not initialized
        if not self.is_ready.is_set():
            raise RuntimeError("Ray cluster is not running")

        if not ray.is_initialized():
            self.logger.warning(f"Ray client disconnected. Reconnecting...")

            # TODO: Re-import ray to reset the connection
            # importlib.reload(ray)
            # importlib.reload(ClusterState)
            await self._connect_to_cluster()

    async def monitor_cluster(self) -> None:
        """Monitor cluster status and update worker nodes history."""
        # Get the current status of the cluster from the BioEngineProxy
        try:
            cluster_status = await asyncio.wait_for(
                self.proxy_handle.get_cluster_state.remote(), timeout=5
            )
        except TimeoutError:
            self.logger.warning(
                f"Timeout while getting cluster state. Trying to get new handle..."
            )
            self.proxy_handle = await asyncio.to_thread(
                serve.get_app_handle, "BioEngineProxy"
            )
            try:
                cluster_status = await asyncio.wait_for(
                    self.proxy_handle.get_cluster_state.remote(), timeout=5
                )
            except TimeoutError:
                raise RuntimeError(
                    "Failed to get cluster state within 5 seconds after retrying with a new handle"
                )

        self.cluster_status_history[time.time()] = cluster_status

        # Limit the history size
        if len(self.cluster_status_history) > self.max_status_history_length:
            self.cluster_status_history.popitem(last=False)

        # Check if SLURM workers need to scale
        if self.slurm_workers:
            await self.slurm_workers.check_scaling()

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
        if self.is_ready.is_set():
            raise RuntimeError("Ray cluster is already running")

        if ray.is_initialized():
            raise RuntimeError(
                "Ray is already initialized. Please stop the existing Ray instance before starting the cluster."
            )

        try:
            # If cluster is not an already running external cluster, start the Ray cluster head node
            if self.mode != "external-cluster":
                await self._start_cluster()

            self.start_time = time.time()

            # Connect a client to the Ray cluster
            self._set_head_node_address()
            self._set_serve_http_url()
            await self._connect_to_cluster()

            # Do a first cluster status check
            await self.monitor_cluster()

            self.is_ready.set()

            self.logger.info("Ray cluster started successfully.")

        except Exception as e:
            self.logger.error(f"Error in cluster startup: {e}")
            self.stop()
            raise e

    async def stop(self) -> None:
        """Stop Ray cluster and all worker nodes.

        Performs a graceful shutdown of the Ray cluster including stopping
        all workers (if running in SLURM mode), disconnecting from the
        cluster, stopping Ray Serve, and canceling any remaining SLURM jobs.

        Raises:
            OSError: If Ray executable is not reachable.
            subprocess.CalledProcessError: If Ray stop command fails.
            Exception: For other shutdown errors.
        """
        try:
            self.is_ready.clear()
            self.start_time = None

            # Shutdown all SLURM workers if running in SLURM mode
            if self.slurm_workers:
                try:
                    await self.slurm_workers.close_all()
                except Exception as e:
                    # Log the error but do not raise, as we still want to attempt Ray shutdown
                    self.logger.error(f"Error shutting down SLURM workers: {e}")

            # Shutdown the BioEngineProxy deployed by Ray Serve
            if self.proxy_handle:
                self.logger.info("Removing down BioEngineProxy...")
                try:
                    await asyncio.to_thread(serve.delete, name="BioEngineProxy")
                    self.proxy_handle = None
                    self.logger.info("BioEngineProxy removed successfully.")
                except Exception as e:
                    self.logger.error(f"Error removing BioEngineProxy: {e}")

            # Shutdown the Ray cluster head node if it is not in external-cluster mode
            if self.mode != "external-cluster":
                # Shutdown Ray Serve first while Ray client is still connected
                self.logger.info("Shutting down Ray Serve...")
                try:
                    await asyncio.to_thread(serve.shutdown)
                    self.logger.info("Ray Serve shut down successfully.")
                except Exception as e:
                    # Log the error but do not raise, as we still want to attempt Ray shutdown
                    self.logger.error(f"Error shutting down Ray Serve: {e}")

                # Disconnect from Ray cluster
                if ray.is_initialized():
                    self.logger.info("Disconnecting from Ray cluster...")
                    try:
                        await asyncio.to_thread(ray.shutdown)
                        self.logger.info("Ray cluster disconnected successfully.")
                    except Exception as e:
                        self.logger.error(f"Error disconnecting from Ray cluster: {e}")

                # Shutdown the Ray cluster head node
                try:
                    await self._shutdown_head_node(grace_period=30)
                except OSError as e:
                    if (
                        e.errno == 107
                        and os.environ.get("APPTAINER_BIND", None) is not None
                    ):
                        self.logger.warning(
                            "Ray executable is not reachable. This may be due to the container's overlay filesystem being torn down."
                        )
            else:
                # Just disconnect from Ray cluster for external clusters
                if ray.is_initialized():
                    self.logger.info("Disconnecting from Ray cluster...")
                    await asyncio.to_thread(ray.shutdown)
                    self.logger.info("Ray cluster disconnected successfully.")

        except Exception as e:
            self.logger.error(f"Error shutting down Ray cluster: {e}")
            raise e
        finally:
            self.cluster_status_history.clear()
