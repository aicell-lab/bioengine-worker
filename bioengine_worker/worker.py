import asyncio
import inspect
import logging
import os
import textwrap
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import cloudpickle
import ray
from hypha_rpc import connect_to_server
from hypha_rpc.sync import login
from hypha_rpc.utils.schema import schema_method

from bioengine_worker import __version__
from bioengine_worker.apps_manager import AppsManager
from bioengine_worker.datasets_manager import DatasetsManager
from bioengine_worker.ray_cluster import RayCluster
from bioengine_worker.utils import check_permissions, create_context, create_logger


class BioEngineWorker:
    """
    Manages Ray cluster lifecycle and model deployments on HPC systems or pre-existing Ray environments.

    This class provides a unified interface for managing BioEngine workers across different
    deployment environments, handling Ray cluster operations, autoscaling, and model
    deployments through Ray Serve. It integrates with the Hypha server to provide
    remote access and management capabilities.

    The BioEngineWorker orchestrates multiple component managers:
    - RayCluster: Manages Ray cluster lifecycle and worker nodes
    - AppsManager: Handles model deployment and artifact management
    - DatasetsManager: Manages dataset loading and access

    Key Features:
    - Multi-environment support (SLURM, single-machine, connect to existing)
    - Hypha server integration for remote management
    - Automatic Ray cluster management with autoscaling
    - Model deployment and artifact management
    - Dataset management and access
    - Python code execution in Ray tasks
    - Comprehensive status monitoring and logging

    Attributes:
        mode (str): Deployment mode ('slurm', 'single-machine', 'external-cluster')
        admin_users (List[str]): List of user emails with admin permissions
        cache_dir (Path): Directory for temporary files and Ray data
        data_dir (Path): Directory for dataset storage
        startup_applications (List[str]): List of deployments to start automatically
        server_url (str): URL of the Hypha server
        workspace (str): Hypha workspace name
        client_id (str): Client ID for Hypha connection
        service_id (str): Service ID for registration
        ray_cluster (RayCluster): Ray cluster manager instance
        apps_manager (AppsManager): Model deployment manager
        dataset_manager (DatasetsManager): Dataset manager
        server: Hypha server connection
        logger: Logger instance for worker operations
    """

    def __init__(
        self,
        mode: Literal["slurm", "single-machine", "external-cluster"] = "slurm",
        admin_users: Optional[List[str]] = None,
        cache_dir: str = "/tmp/bioengine",
        data_dir: str = "/data",
        startup_applications: Optional[List[Union[str, Tuple[str, str]]]] = None,
        monitoring_interval_seconds: int = 10,
        # Hypha server connection configuration
        server_url: str = "https://hypha.aicell.io",
        workspace: Optional[str] = None,
        token: Optional[str] = None,
        client_id: Optional[str] = None,
        # Ray cluster configuration
        ray_cluster_config: Optional[Dict[str, Any]] = None,
        # BioEngine dashboard URL
        dashboard_url: str = "https://dev.bioimage.io/#/bioengine",
        # Logger configuration
        log_file: Optional[str] = None,
        debug: bool = False,
    ):
        """
        Initialize BioEngine worker with component managers.

        Sets up the worker with the specified configuration and initializes
        all component managers (RayCluster, AppsManager, DatasetsManager).
        Handles authentication with the Hypha server and configures logging.

        Args:
            mode: Ray cluster mode ('slurm', 'single-machine', 'external-cluster')
            admin_users: List of user emails with admin permissions
            cache_dir: Directory for temporary files and Ray data
            data_dir: Directory for dataset storage
            startup_applications: List of deployments to start automatically
            server_url: URL of the Hypha server to register with
            workspace: Hypha workspace to connect to (defaults to user's workspace)
            token: Authentication token for Hypha server (uses HYPHA_TOKEN env var if None)
            client_id: Client ID for Hypha connection (auto-generated if None)
            ray_cluster_config: Configuration dictionary for RayCluster component
            log_file: File path for logging output (auto-generated if None)
            debug: Enable debug-level logging

        Raises:
            Exception: If initialization of any component fails
        """
        self.admin_users = admin_users or []
        self.cache_dir = Path(cache_dir).resolve()
        self.data_dir = Path(data_dir).resolve()
        self.dashboard_url = dashboard_url.rstrip("/")
        self.monitoring_interval_seconds = monitoring_interval_seconds

        self.server_url = server_url
        self.workspace = workspace
        self._token = token or os.environ.get("HYPHA_TOKEN")
        self.client_id = client_id
        self.service_id = "bioengine-worker"

        self.start_time = None
        self._last_monitoring = 0
        self._server = None
        self.is_ready = asyncio.Event()
        self._serve_event = asyncio.Event()
        self._monitoring_task = None
        self.full_service_id = None

        self.logger = create_logger(
            name="BioEngineWorker",
            level=logging.DEBUG if debug else logging.INFO,
            log_file=log_file,
        )
        self.logger.info(
            f"Initializing BioEngineWorker v{__version__} with mode '{mode}'..."
        )
        try:
            # If token is not provided, attempt to login
            if not self._token:
                print("\n" + "=" * 60)
                print("NO HYPHA TOKEN FOUND - USER LOGIN REQUIRED")
                print("-" * 60, end="\n\n")
                self._token = login(
                    {"server_url": self.server_url, "expires_in": 3600 * 24 * 365}
                )
                print("\n" + "-" * 60)
                print("Login completed successfully!")
                print("=" * 60, end="\n\n")

            # Set parameters for RayCluster
            ray_cluster_config = ray_cluster_config or {}

            # Overwrite existing 'mode', 'ray_temp_dir', 'force_clean_up', 'log_file', and 'debug' parameters if provided
            self._set_parameter(ray_cluster_config, "mode", mode)
            self._set_parameter(
                ray_cluster_config, "ray_temp_dir", self.cache_dir / "ray"
            )
            force_clean_up = not ray_cluster_config.pop("skip_cleanup", False)
            self._set_parameter(ray_cluster_config, "force_clean_up", force_clean_up)
            self._set_parameter(ray_cluster_config, "log_file", log_file)
            self._set_parameter(ray_cluster_config, "debug", debug)

            # Initialize RayCluster and update mode
            self.ray_cluster = RayCluster(**ray_cluster_config)

            # Initialize AppsManager
            self.apps_manager = AppsManager(
                ray_cluster=self.ray_cluster,
                token=self._token,
                apps_cache_dir=self.cache_dir / "apps",
                apps_data_dir=self.data_dir,
                startup_applications=startup_applications,
                log_file=log_file,
                debug=debug,
            )

            # Initialize DatasetsManager
            self.dataset_manager = DatasetsManager(
                data_dir=self.data_dir,
                log_file=log_file,
                debug=debug,
            )
        except Exception as e:
            self.logger.error(f"Error initializing BioEngineWorker: {e}")
            raise

    def _set_parameter(
        self,
        kwargs: Dict[str, Any],
        key: str,
        value: Any,
        overwrite: bool = True,
    ) -> dict:
        """
        Set parameter in configuration dictionary with optional overwrite control.

        Args:
            kwargs: Configuration dictionary to modify
            key: Parameter key to set
            value: Value to set for the parameter
            overwrite: Whether to overwrite existing values

        Returns:
            Modified configuration dictionary
        """
        if overwrite:
            if key in kwargs and kwargs[key] != value:
                self.logger.warning(
                    f"Overwriting provided {key} value: {kwargs[key]!r} -> {value!r}"
                )
            kwargs[key] = value
        else:
            if key not in kwargs or kwargs[key] is None:
                kwargs[key] = value

    async def _connect_to_server(self) -> None:
        """
        Connect to Hypha server and authenticate user.

        Establishes connection to the specified Hypha server using the provided
        token and workspace. Updates admin users list with the authenticated user.

        Raises:
            ValueError: If connection to Hypha server fails
        """
        if self._server:
            self.logger.debug("Closing existing Hypha server connection...")
            try:
                await self._server.disconnect()
            except Exception as e:
                self.logger.error(f"Error closing Hypha server connection: {e}")
        self.logger.debug(f"Connecting to Hypha server at {self.server_url}...")
        self._server = await connect_to_server(
            {
                "server_url": self.server_url,
                "token": self._token,
                "workspace": self.workspace,
                "client_id": self.client_id,
            }
        )

        user_id = self._server.config.user["id"]
        user_email = self._server.config.user["email"]

        self.workspace = self._server.config.workspace
        self.client_id = self._server.config.client_id
        self.logger.info(
            f"User '{user_id}' connected to workspace '{self.workspace}' with client ID '{self.client_id}'."
        )

        # Update admin users list with the authenticated user and ensure it's at the top
        if user_id in self.admin_users:
            self.admin_users.remove(user_id)
        if user_email in self.admin_users:
            self.admin_users.remove(user_email)
        self.admin_users.insert(0, user_id)
        self.admin_users.insert(1, user_email)
        self._admin_context = create_context(user_id, user_email)
        self.logger.debug(
            f"Admin users for this BioEngine worker: {', '.join(self.admin_users)}"
        )

    async def _check_hypha_connection(self, reconnect: bool = True) -> None:
        try:
            await asyncio.wait_for(self._server.echo("ping"), timeout=10)
        except Exception as e:
            if reconnect:
                self.logger.warning(
                    f"Hypha server connection error. Attempting to reconnect..."
                )
                await self._connect_to_server()
                await self._register_bioengine_worker_service()
            else:
                raise RuntimeError(f"Hypha server connection error: {e}")

    async def _register_bioengine_worker_service(self) -> None:
        # Register service interface
        description = "Manages BioEngine Apps and Datasets"
        if self.ray_cluster.mode == "slurm":
            description += " on a HPC system with Ray Autoscaler support for dynamic resource management."
        elif self.ray_cluster.mode == "single-machine":
            description += " on a single machine Ray instance."
        else:
            description += " in a pre-existing Ray environment."

        # TODO: return more informative error messages, e.g. return error
        service_info = await self._server.register_service(
            {
                "id": self.service_id,
                "name": "BioEngine Worker",
                "type": "bioengine-worker",
                "description": description,
                "config": {
                    "visibility": "public",
                    "require_context": True,
                },
                "is_ready": lambda context: self.is_ready.is_set(),
                "get_status": self.get_status,
                "load_dataset": self.dataset_manager.load_dataset,
                "close_dataset": self.dataset_manager.close_dataset,
                "cleanup_datasets": self.dataset_manager.cleanup,
                "execute_python_code": self.execute_python_code,
                "create_artifact": self.apps_manager.create_artifact,
                "delete_artifact": self.apps_manager.delete_artifact,
                "deploy_application": self.apps_manager.deploy_application,
                "deploy_applications": self.apps_manager.deploy_applications,
                "undeploy_application": self.apps_manager.undeploy_application,
                "cleanup_deployments": self.apps_manager.cleanup,
                "stop_worker": self.stop,
            },
            {"overwrite": True},
        )
        self.full_service_id = service_info.id

        self.logger.info(
            f"Manage BioEngine worker at: {self.dashboard_url}/worker?service_id={self.full_service_id}"
        )

    async def _create_monitoring_task(self, max_consecutive_errors: int = 5) -> None:
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
        self.logger.debug(
            "Starting monitoring task with interval "
            f"{self.monitoring_interval_seconds} seconds..."
        )

        self.start_time = time.time()
        consecutive_errors = 0
        while self.start_time:
            try:
                current_time = time.time()
                if (
                    current_time - self._last_monitoring
                    < self.monitoring_interval_seconds
                ):
                    continue  # Skip if within check interval
                self._last_monitoring = current_time

                # Check connection to Hypha server
                await self._check_hypha_connection()

                # Check connection to Ray cluster
                await self.ray_cluster.check_connection()

                # Run cluster monitoring
                await self.ray_cluster.monitor_cluster()

                # Run BioEngine Apps monitoring
                await self.apps_manager.monitor_applications()

                # Run BioEngine Datasets monitoring
                await self.dataset_manager.monitor_datasets()

                # Reset error counter on success
                consecutive_errors = 0

            except Exception as e:
                self.logger.error(f"Error in monitoring task: {e}")
                consecutive_errors += 1
                # Don't raise the exception to avoid crashing the monitoring task
                # Instead, continue monitoring until the maximum of consecutive errors is reached

                if consecutive_errors >= max_consecutive_errors:
                    self.logger.error(
                        f"Stopping monitoring loop after {consecutive_errors} consecutive errors"
                    )
                    self.start_time = None

            # Sleep for 1 second before next iteration
            await asyncio.sleep(1)

        self.logger.debug("Monitoring task stopped. Shutting down BioEngine worker...")

        # Remove the service from Hypha server
        try:
            await self._server.unregister_service(self.full_service_id)
            self.logger.info(
                f"BioEngine worker service '{self.full_service_id}' removed."
            )
        except Exception:
            pass

        # Clean up all components
        await self.dataset_manager.cleanup(self._admin_context)
        await self.apps_manager.cleanup(self._admin_context)
        await self.ray_cluster.stop()

        # Signal the serve loop to exit
        self._serve_event.set()
        self.is_ready.clear()

        # Clear the monitoring task reference
        self._monitoring_task = None

    async def start(self) -> str:
        """
        Start the BioEngine worker and all component services.

        Initializes the Ray cluster (or connects to existing one), establishes
        connection to the Hypha server, registers the service interface, and
        starts all component managers. Also deploys any configured startup
        deployments.

        Returns:
            str: Service ID assigned after successful registration with Hypha

        Raises:
            RuntimeError: If Ray is already initialized when connecting to existing cluster
            Exception: If startup of any component fails
        """
        self.start_time = time.time()

        # Start the Ray cluster
        await self.ray_cluster.start()

        # Register the BioEngine worker service with the Hypha server
        await self._connect_to_server()
        await self._check_hypha_connection(reconnect=False)

        # Initialize component managers with the server connection
        await self.apps_manager.initialize(
            server=self._server, admin_users=self.admin_users
        )
        await self.dataset_manager.initialize(
            server=self._server, admin_users=self.admin_users
        )

        # Start the monitoring task
        self._monitoring_task = asyncio.create_task(
            self._create_monitoring_task(),
            name="BioEngineWorker Monitoring Task",
        )

        # Register the BioEngine worker service interface
        await self._register_bioengine_worker_service()

        # Signal that the worker is ready
        self.is_ready.set()

        # Start the serve loop
        self._serve_event.clear()
        await self._serve_event.wait()

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
        if self.ray_cluster.mode == "slurm":
            self.logger.info("Notifying SLURM workers of cluster state change")
            self.last_cluster_status = time.time() - self._last_monitoring + delay_s

    @schema_method
    async def stop(self, context: Dict[str, Any]) -> None:
        """
        Clean up resources and stop the Ray cluster if managed by this worker.

        Performs cleanup of all components including datasets, deployments, and
        the Ray cluster. Signals the serve loop to exit gracefully.

        Args:
            context: Optional context information from Hypha request

        Raises:
            Exception: If cleanup of any component fails
        """
        check_permissions(
            context=context,
            authorized_users=self.admin_users,
            resource_name=f"shutdown the BioEngine worker",
        )
        self.start_time = None

        # Wait for the monitoring task to finish if it is running
        if self.monitoring_task and not self.monitoring_task.done():
            await self.monitoring_task
        else:
            await self._shutdown_ray()

    @schema_method
    async def get_status(self, context: Optional[Dict[str, Any]] = None) -> dict:
        """
        Get comprehensive status of the BioEngine worker and all components.

        Returns detailed status information including service uptime, Ray cluster
        status, deployment status, and dataset status.

        Args:
            context: Optional context information from Hypha request

        Returns:
            Dict containing:
                - service: Service start time and uptime information
                - ray_cluster: Ray cluster status and worker node information
                - bioengine_apps: Deployment manager status and active deployments
                - bioengine_datasets: Dataset manager status and loaded datasets

        Raises:
            RuntimeError: If Ray is not initialized
        """
        status = {
            "service_start_time": self.start_time,
            "ray_cluster": self.ray_cluster.status,
            "bioengine_apps": await self.apps_manager.get_status(),
            "bioengine_datasets": await self.dataset_manager.get_status(),
        }

        return status

    # TODO: Does not work for type hint 'callable'
    # @schema_method
    async def execute_python_code(
        self,
        code: str = None,
        function_name: str = "analyze",
        func_bytes: bytes = None,
        mode: Literal["source", "pickle"] = "source",
        remote_options: Dict[str, Any] = None,
        args: List[Any] = None,
        kwargs: Dict[str, Any] = None,
        write_stdout: Optional[callable] = None,
        write_stderr: Optional[callable] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute Python code in a Ray task with resource allocation and output streaming.

        Executes user-provided Python code either from source or pickled function
        in a Ray task with specified resource requirements. Supports both synchronous
        and asynchronous functions with output streaming capabilities.

        Args:
            code: Python source code string containing the function to execute
            function_name: Name of the function to execute from the source code
            func_bytes: Pickled function bytes (used when mode='pickle')
            mode: Execution mode - 'source' for code string, 'pickle' for serialized function
            remote_options: Ray remote options for resource allocation (num_cpus, num_gpus, etc.)
            args: Positional arguments to pass to the function
            kwargs: Keyword arguments to pass to the function
            write_stdout: Optional callback function for streaming stdout output
            write_stderr: Optional callback function for streaming stderr output
            context: Optional context information from Hypha request

        Returns:
            Dict containing:
                - result: Function return value (if successful)
                - error: Error message (if function failed)
                - traceback: Full traceback (if function failed)
                - stdout: Captured stdout output
                - stderr: Captured stderr output

        Raises:
            RuntimeError: If Ray is not initialized
            Exception: If function deserialization or execution fails
        """
        check_permissions(
            context=context,
            authorized_users=self.admin_users,
            resource_name=f"execute Python code in a Ray task",
        )

        self.logger.info("Executing Python code in Ray task...")

        args = args or []
        kwargs = kwargs or {}
        # The @ray.remote decorator requires arguments when using parentheses
        remote_options = remote_options or {"num_cpus": 1}

        # Deserialize function before Ray execution
        if mode == "pickle":
            try:
                user_func = cloudpickle.loads(func_bytes)
            except Exception as e:
                return {
                    "error": f"Failed to unpickle function: {e}",
                    "traceback": traceback.format_exc(),
                }
        else:
            exec_namespace = {}
            exec(textwrap.dedent(code), exec_namespace)
            user_func = exec_namespace.get(function_name)
            if not user_func:
                return {
                    "error": f"Function '{function_name}' not found in source code",
                    "available_functions": [
                        k for k, v in exec_namespace.items() if inspect.isfunction(v)
                    ],
                }

        # The Ray task itself (pure, pickle-safe)
        @ray.remote(**remote_options)
        def ray_task(func, args, kwargs):
            import asyncio
            import contextlib
            import io
            import traceback

            stdout_buffer = io.StringIO()
            stderr_buffer = io.StringIO()

            with (
                contextlib.redirect_stdout(stdout_buffer),
                contextlib.redirect_stderr(stderr_buffer),
            ):
                try:
                    result = func(*args, **kwargs)
                    if asyncio.iscoroutine(result):
                        result = asyncio.run(result)
                    return {
                        "result": result,
                        "stdout": stdout_buffer.getvalue(),
                        "stderr": stderr_buffer.getvalue(),
                    }
                except Exception as e:
                    return {
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                        "stdout": stdout_buffer.getvalue(),
                        "stderr": stderr_buffer.getvalue(),
                    }

        obj_ref = ray_task.remote(user_func, args, kwargs)
        if self.ray_cluster.mode == "slurm":
            await self.ray_cluster.notify()
        result = await asyncio.wait_for(obj_ref, timeout=600)

        # Stream output to client
        if write_stdout and result.get("stdout"):
            for line in result["stdout"].splitlines():
                await write_stdout(line)
        if write_stderr and result.get("stderr"):
            for line in result["stderr"].splitlines():
                await write_stderr(line)

        return result


if __name__ == "__main__":
    """Test the BioEngineWorker class functionality"""
    import aiohttp

    async def test_bioengine_worker(keep_running=True):
        try:
            # Create BioEngine worker instance
            server_url = "https://hypha.aicell.io"
            token = os.environ["HYPHA_TOKEN"] or await login({"server_url": server_url})
            bioengine_worker = BioEngineWorker(
                workspace="chiron-platform",
                server_url=server_url,
                token=token,
                service_id="bioengine-worker",
                dataset_config={
                    "data_dir": str(Path(__file__).parent.parent / "data"),
                    "service_id": "bioengine-dataset",
                },
                ray_cluster_config={
                    "head_num_cpus": 4,
                    "ray_temp_dir": str(
                        Path(__file__).parent.parent / ".bioengine" / "ray"
                    ),
                    "image": str(
                        Path(__file__).parent.parent
                        / "apptainer_images"
                        / f"bioengine-worker_{__version__}.sif"
                    ),
                },
                ray_autoscaling_config={
                    "metrics_interval_seconds": 10,
                },
                ray_deployment_config={
                    "service_id": "bioengine-apps",
                },
                debug=True,
            )

            # Initialize worker
            sid = await bioengine_worker.start()

            # Test registered service
            server = await connect_to_server(
                {
                    "server_url": server_url,
                    "token": token,
                    "workspace": bioengine_worker.workspace,
                }
            )
            worker_service = await server.get_service(sid)

            # Get initial status
            status = await worker_service.get_status()
            print("\nInitial status:", status)

            # Try accessing the dataset manager
            dataset_url = await worker_service.load_dataset(dataset_id="blood")
            print("Dataset URL:", dataset_url)

            # Get dataset info
            headers = {"Authorization": f"Bearer {token}"}
            async with aiohttp.ClientSession() as session:
                async with session.get(dataset_url, headers=headers) as response:
                    response.raise_for_status()
                    dataset_info = await response.json()
            print("Dataset info:", dataset_info)

            # Test deploying an artifact
            artifact_id = "example-deployment"
            deployment_name = await worker_service.deploy_artifact(
                artifact_id=artifact_id,
            )
            worker_status = await worker_service.get_status()
            assert deployment_name in worker_status["deployments"]

            # Test registered deployment service
            deployment_service_id = worker_status["deployments"]["service_id"]
            deployment_service = await server.get_service(deployment_service_id)

            result = await deployment_service[deployment_name]()
            print(result)

            # Keep server running if requested
            if keep_running:
                print("Server running. Press Ctrl+C to stop.")
                await server.serve()

        except Exception as e:
            print(f"Test error: {e}")
            raise e
        finally:
            # Cleanup
            await bioengine_worker.cleanup(context=bioengine_worker._admin_context)

    # Run the test
    asyncio.run(test_bioengine_worker())
