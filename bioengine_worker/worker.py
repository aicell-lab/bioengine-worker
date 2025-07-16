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
    Enterprise-grade BioEngine worker for distributed AI model deployment and execution.

    The BioEngineWorker provides a comprehensive platform for managing AI model deployments
    across diverse computational environments, from high-performance computing clusters with
    SLURM job scheduling to single-machine deployments and external Ray clusters. It serves
    as the central orchestration layer for the BioEngine ecosystem.

    Architecture Overview:
    The worker orchestrates three primary component managers, each handling specialized
    functionality while maintaining enterprise-grade security, monitoring, and lifecycle management:

    • RayCluster: Manages distributed Ray cluster lifecycle including SLURM-based autoscaling,
      resource allocation, and worker node management across HPC environments
    • AppsManager: Handles AI model deployment lifecycle through Ray Serve, including artifact
      management, deployment orchestration, and application scaling
    • DatasetsManager: Provides secure dataset access through HTTP streaming services with
      permission-based file access and manifest-driven configuration

    Core Capabilities:
    - Multi-environment deployment support (SLURM HPC, single-machine, external clusters)
    - Enterprise-grade security with two-level permission systems (admin + resource-specific)
    - Hypha server integration for remote management and service discovery
    - Automatic Ray cluster lifecycle management with intelligent autoscaling
    - AI model deployment and serving through Ray Serve with health monitoring
    - Secure dataset management with streaming access and authorization controls
    - Python code execution in distributed Ray tasks with resource allocation
    - Comprehensive monitoring, logging, and status reporting
    - Graceful shutdown and resource cleanup with signal handling

    Security Architecture:
    - Admin-level permissions for cluster and deployment management operations
    - Resource-specific authorization for dataset access and model execution
    - Context-aware permission checking with detailed audit logging
    - Secure artifact management with version control and validation
    - Isolated execution environments with resource limits and monitoring

    Deployment Modes:
    1. **SLURM Mode**: Full HPC integration with automatic worker scheduling, resource allocation,
       and cluster autoscaling based on computational demand
    2. **Single-Machine Mode**: Local Ray cluster for development and small-scale deployments
       with configurable resource limits
    3. **External-Cluster Mode**: Connection to pre-existing Ray clusters with service registration
       and management capabilities

    Integration Points:
    - Hypha Server: Service registration, remote access, and workspace integration
    - Ray Ecosystem: Distributed computing, model serving, and resource management
    - SLURM: HPC job scheduling, resource allocation, and cluster management
    - File Systems: Dataset access, artifact storage, and temporary file management

    Attributes:
        admin_users (List[str]): List of user IDs/emails authorized for admin operations
        cache_dir (Path): Directory for temporary files, Ray data, and worker state
        data_dir (Path): Root directory for dataset storage and access
        dashboard_url (str): URL of the BioEngine dashboard for worker management
        monitoring_interval_seconds (int): Interval for status monitoring and health checks
        server_url (str): URL of the Hypha server for service registration
        workspace (str): Hypha workspace name for service isolation
        client_id (str): Unique client identifier for Hypha connection
        service_id (str): Service identifier for registration ("bioengine-worker")
        full_service_id (str): Complete service ID including workspace and user context
        ray_cluster (RayCluster): Ray cluster management component
        apps_manager (AppsManager): Application deployment management component
        dataset_manager (DatasetsManager): Dataset access management component
        start_time (float): Timestamp when worker was started
        is_ready (asyncio.Event): Event signaling worker initialization completion
        logger (logging.Logger): Structured logger for worker operations

    Example Usage:
        ```python
        # Initialize worker for SLURM HPC environment
        worker = BioEngineWorker(
            mode="slurm",
            admin_users=["admin@institution.edu"],
            cache_dir="/tmp/bioengine",
            data_dir="/shared/datasets",
            server_url="https://hypha.aicell.io",
            ray_cluster_config={
                "max_workers": 10,
                "default_num_gpus": 1,
                "default_num_cpus": 8
            }
        )

        # Start all services
        service_id = await worker.start()

        # Worker is now ready for model deployments and dataset access
        status = await worker.get_status()
        ```

    Note:
        The BioEngineWorker requires proper configuration of the deployment environment,
        including access to storage systems, network connectivity for Hypha server
        communication, and appropriate permissions for the target deployment mode.
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
        dashboard_url: str = "https://bioimage.io/#/bioengine",
        # Logger configuration
        log_file: Optional[str] = None,
        debug: bool = False,
    ):
        """
        Initialize BioEngine worker with enterprise-grade configuration and component managers.

        Sets up the worker with comprehensive configuration management, initializes all
        component managers (RayCluster, AppsManager, DatasetsManager), configures security
        settings, and establishes logging infrastructure. Handles authentication with the
        Hypha server and prepares the worker for service registration.

        The initialization process:
        1. Validates and normalizes configuration parameters
        2. Sets up secure logging infrastructure with optional file output
        3. Attempts Hypha server authentication (interactive login if needed)
        4. Initializes RayCluster with environment-specific configuration
        5. Prepares AppsManager and DatasetsManager for later initialization
        6. Configures monitoring and health check systems

        Args:
            mode: Ray cluster deployment mode determining the operational environment:
                  - 'slurm': HPC environment with SLURM job scheduling and autoscaling
                  - 'single-machine': Local Ray cluster for development/small deployments
                  - 'external-cluster': Connect to existing Ray cluster
            admin_users: List of user IDs/emails authorized for administrative operations.
                        Auto-includes the authenticated user from Hypha connection.
            cache_dir: Directory path for temporary files, Ray data storage, and worker state.
                      Must be accessible and have sufficient space for Ray operations.
            data_dir: Root directory path for dataset storage and access. Should be mounted
                     storage accessible across worker nodes in distributed deployments.
            startup_applications: List of applications to deploy automatically during worker
                                 startup. Can be artifact IDs or (artifact_id, app_id) tuples.
            monitoring_interval_seconds: Interval in seconds for status monitoring, health
                                       checks, and cluster state updates.
            server_url: URL of the Hypha server for service registration and remote access.
                       Must be accessible from the deployment environment.
            workspace: Hypha workspace name for service isolation. Defaults to user's
                      workspace if not specified.
            token: Authentication token for Hypha server. Uses HYPHA_TOKEN environment
                  variable if not provided, prompts for interactive login otherwise.
            client_id: Unique client identifier for Hypha connection. Auto-generated if
                      not specified to ensure unique service registration.
            ray_cluster_config: Configuration dictionary for RayCluster component including
                              SLURM job parameters, resource limits, and autoscaling settings.
            dashboard_url: Base URL of the BioEngine dashboard for worker management and
                          monitoring interfaces.
            log_file: File path for structured logging output. Auto-generated timestamp-based
                     filename if not specified.
            debug: Enable debug-level logging for detailed troubleshooting and development.

        Raises:
            ValueError: If configuration parameters are invalid or incompatible
            ConnectionError: If unable to authenticate with Hypha server
            PermissionError: If insufficient permissions for cache/data directories
            Exception: If Ray cluster initialization fails

        Note:
            The worker is not ready for use until `start()` is called, which completes
            the initialization process by starting the Ray cluster and registering with
            the Hypha server.
        """
        self.admin_users = admin_users or []
        self.cache_dir = Path(cache_dir).resolve()
        self.data_dir = Path(data_dir).resolve()
        self.dashboard_url = dashboard_url.rstrip("/")
        self.monitoring_interval_seconds = monitoring_interval_seconds

        # Hypha server configuration
        self.server_url = server_url
        self.workspace = workspace
        self._token = token or os.environ.get("HYPHA_TOKEN")
        self.client_id = client_id
        self.service_id = "bioengine-worker"

        # Worker state management
        self.start_time = None
        self._last_monitoring = 0
        self._server = None
        self.is_ready = asyncio.Event()
        self._monitoring_task = None
        self.full_service_id = None
        self._admin_context = None

        # Initialize structured logging
        self.logger = create_logger(
            name="BioEngineWorker",
            level=logging.DEBUG if debug else logging.INFO,
            log_file=log_file,
        )
        self.logger.info(
            f"Initializing BioEngineWorker v{__version__} with mode '{mode}'"
        )

        try:
            # Ensure cache and data directories exist with proper permissions
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.data_dir.mkdir(parents=True, exist_ok=True)
            
            # Attempt interactive login if no token provided
            if not self._token:
                self.logger.info("No authentication token provided, attempting interactive login...")
                print("\n" + "=" * 60)
                print("NO HYPHA TOKEN FOUND - USER LOGIN REQUIRED")
                print("-" * 60, end="\n\n")
                self._token = login(
                    {"server_url": self.server_url, "expires_in": 3600 * 24 * 365}
                )
                print("\n" + "-" * 60)
                print("Login completed successfully!")
                print("=" * 60, end="\n\n")
                self.logger.info("Interactive login completed successfully")

            # Configure Ray cluster with environment-specific parameters
            ray_cluster_config = ray_cluster_config or {}
            
            # Set core parameters with precedence for explicit values
            self._set_parameter(ray_cluster_config, "mode", mode)
            self._set_parameter(ray_cluster_config, "ray_temp_dir", self.cache_dir / "ray")
            force_clean_up = not ray_cluster_config.pop("skip_cleanup", False)
            self._set_parameter(ray_cluster_config, "force_clean_up", force_clean_up)
            self._set_parameter(ray_cluster_config, "log_file", log_file)
            self._set_parameter(ray_cluster_config, "debug", debug)

            # Initialize Ray cluster manager
            self.ray_cluster = RayCluster(**ray_cluster_config)
            
            # Initialize component managers with enhanced configuration
            self.apps_manager = AppsManager(
                ray_cluster=self.ray_cluster,
                token=self._token,
                apps_cache_dir=self.cache_dir / "apps",
                apps_data_dir=self.data_dir,
                startup_applications=startup_applications,
                log_file=log_file,
                debug=debug,
            )

            self.dataset_manager = DatasetsManager(
                data_dir=self.data_dir,
                log_file=log_file,
                debug=debug,
            )

            self.logger.info("BioEngineWorker initialization completed successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize BioEngineWorker: {e}")
            raise

    def _set_parameter(
        self,
        kwargs: Dict[str, Any],
        key: str,
        value: Any,
        overwrite: bool = True,
    ) -> None:
        """
        Set parameter in configuration dictionary with optional overwrite control.

        Utility method for safely updating configuration dictionaries while respecting
        existing values when needed. Provides a consistent interface for parameter
        management across component initialization.

        Args:
            kwargs: Configuration dictionary to modify in-place
            key: Parameter key to set or update
            value: Value to assign to the parameter
            overwrite: Whether to overwrite existing values (default: True)
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
        Establish connection to Hypha server and configure admin user permissions.

        Authenticates with the Hypha server using the configured token and workspace,
        then updates the admin users list with the authenticated user information.
        This ensures the authenticated user has administrative privileges for the worker.

        The connection process:
        1. Closes any existing connection
        2. Establishes new connection with authentication
        3. Extracts user information from the server configuration
        4. Updates admin users list with authenticated user (ID and email)
        5. Creates admin context for internal operations

        Raises:
            ConnectionError: If unable to connect to Hypha server
            AuthenticationError: If token authentication fails
            ValueError: If server configuration is invalid
        """
        if self._server:
            self.logger.debug("Closing existing Hypha server connection")
            try:
                await self._server.disconnect()
            except Exception as e:
                self.logger.error(f"Error closing Hypha server connection: {e}")
                
        self.logger.info(f"Connecting to Hypha server at {self.server_url}")
        self._server = await connect_to_server(
            {
                "server_url": self.server_url,
                "token": self._token,
                "workspace": self.workspace,
                "client_id": self.client_id,
            }
        )

        # Extract authenticated user information
        user_id = self._server.config.user["id"]
        user_email = self._server.config.user["email"]

        # Update connection configuration from server response
        self.workspace = self._server.config.workspace
        self.client_id = self._server.config.client_id
        
        self.logger.info(
            f"User '{user_id}' connected to workspace '{self.workspace}' "
            f"with client ID '{self.client_id}'"
        )

        # Update admin users list with authenticated user (ensure at top of list)
        if user_id in self.admin_users:
            self.admin_users.remove(user_id)
        if user_email in self.admin_users:
            self.admin_users.remove(user_email)
        self.admin_users.insert(0, user_id)
        self.admin_users.insert(1, user_email)
        
        # Create admin context for internal operations
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
        try:
            # Signal that the worker is ready
            self.is_ready.set()

            self.logger.debug(
                "Starting monitoring task with interval "
                f"{self.monitoring_interval_seconds} seconds..."
            )
            consecutive_errors = 0
            while self.is_ready.is_set():
                try:
                    # Sleep for 1 second before next iteration
                    await asyncio.sleep(1)

                    # Check if enough time has passed since the last monitoring
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
                        # Reset the is_ready event to indicate worker is no longer ready
                        self.is_ready.clear()

        except asyncio.CancelledError:
            self.logger.info("Monitoring task cancelled. Stopping BioEngine worker...")
        except Exception as e:
            self.logger.error(f"Unexpected error in monitoring task: {e}")
            raise
        finally:
            # Ensure the is_ready event is reset
            self.is_ready.clear()

            # Remove the service from Hypha server
            self.start_time = None
            try:
                await self._server.unregister_service(self.full_service_id)
                self.logger.info(
                    f"BioEngine worker service '{self.full_service_id}' removed."
                )
            except Exception:
                self.logger.error(
                    f"Failed to remove BioEngine worker service '{self.full_service_id}'"
                )

            # Clean up all components
            await self.dataset_manager.cleanup(self._admin_context)
            await self.apps_manager.cleanup(self._admin_context)
            await self.ray_cluster.stop()

            # Disconnect from the Hypha server
            try:
                await self._server.disconnect()
            except Exception as e:
                self.logger.error(f"Error closing Hypha server connection: {e}")

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

        # Register the BioEngine worker service interface
        await self._register_bioengine_worker_service()

        # Start the monitoring task
        self._monitoring_task = asyncio.create_task(
            self._create_monitoring_task(),
            name="BioEngineWorker Monitoring Task",
        )

    async def notify(self, delay_s: int = 0) -> None:
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
            self._last_monitoring = time.time() - self._last_monitoring + delay_s

    @schema_method
    async def stop(self, timeout: int = 30, context: Dict[str, Any] = None) -> None:
        """
        Gracefully shutdown the BioEngine worker and cleanup all resources.

        Performs comprehensive cleanup of all worker components including datasets,
        deployments, Ray cluster, and monitoring tasks. Ensures proper resource
        cleanup and service deregistration from the Hypha server.

        The shutdown process:
        1. Validates admin permissions for shutdown operation
        2. Cancels monitoring tasks and waits for completion
        3. Cleanup applications and deployments through AppsManager
        4. Cleanup datasets and HTTP services through DatasetsManager
        5. Shutdown Ray cluster (if managed by this worker)
        6. Disconnect from Hypha server
        7. Clear worker state and mark as not ready

        Args:
            timeout: Maximum time in seconds to wait for each cleanup operation
            context: Request context containing user information for permission checking

        Raises:
            PermissionError: If user is not authorized for shutdown operations
            TimeoutError: If cleanup operations exceed the specified timeout
            Exception: If critical cleanup operations fail
        """
        check_permissions(
            context=context,
            authorized_users=self.admin_users,
            resource_name="shutdown the BioEngine worker",
        )

        self.logger.info("Starting graceful shutdown of BioEngine worker")

        # Check if the worker is ready
        if not self.is_ready.is_set() or not self._monitoring_task:
            self.logger.info("BioEngine worker is not running. Nothing to stop.")
            return

        # Cancel and wait for monitoring task to finish
        self._monitoring_task.cancel()
        try:
            await asyncio.wait_for(self._monitoring_task, timeout=timeout)
        except (TimeoutError, asyncio.CancelledError):
            self.logger.warning(f"Monitoring task did not finish cleanly within {timeout} seconds")

        # Clear monitoring task reference
        self._monitoring_task = None

    @schema_method
    async def get_status(self, context: Optional[Dict[str, Any]] = None) -> dict:
        """
        Retrieve comprehensive status information for the BioEngine worker and all components.

        Provides detailed status information including Ray cluster health, active
        deployments, loaded datasets, resource utilization, and worker uptime.
        This method is used for monitoring, debugging, and dashboard displays.

        The status report includes:
        - Worker service information (start time, uptime, service ID)
        - Ray cluster status (nodes, resources, health)
        - Applications manager status (active deployments, resource usage)
        - Datasets manager status (loaded datasets, service endpoints)
        - System resource information and health metrics

        Args:
            context: Optional request context for permission checking and audit logging

        Returns:
            Dict containing comprehensive status information:
                - service_start_time: Timestamp when worker was started
                - service_uptime: Duration since worker startup in seconds
                - service_id: Full service identifier for Hypha registration
                - worker_mode: Deployment mode (slurm/single-machine/external-cluster)
                - ray_cluster: Ray cluster status including nodes and resources
                - bioengine_apps: Applications manager status and active deployments
                - bioengine_datasets: Datasets manager status and loaded datasets
                - admin_users: List of users with administrative privileges
                - is_ready: Boolean indicating if worker is fully initialized

        Raises:
            RuntimeError: If Ray cluster is not properly initialized
            ConnectionError: If unable to retrieve status from components
        """
        current_time = time.time()
        status = {
            "service_start_time": self.start_time,
            "service_uptime": current_time - self.start_time if self.start_time else 0,
            "service_id": self.full_service_id,
            "worker_mode": self.ray_cluster.mode,
            "workspace": self.workspace,
            "client_id": self.client_id,
            "ray_cluster": self.ray_cluster.status,
            "bioengine_apps": await self.apps_manager.get_status(),
            "bioengine_datasets": await self.dataset_manager.get_status(),
            "admin_users": self.admin_users,
            "is_ready": self.is_ready.is_set(),
        }

        return status

    @schema_method(arbitrary_types_allowed=True)
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
        Execute Python code in a distributed Ray task with comprehensive resource management.

        Provides secure execution of user-provided Python code in isolated Ray tasks with
        configurable resource allocation, output streaming, and error handling. Supports
        both source code execution and pre-serialized function execution for maximum
        flexibility in distributed computing scenarios.

        The execution process:
        1. Validates admin permissions for code execution
        2. Deserializes function (from source or pickle) with error handling
        3. Creates isolated Ray task with specified resource requirements
        4. Executes function with stdout/stderr capture and timeout protection
        5. Returns results with comprehensive output and error information
        6. Streams output to client if callback functions provided

        Security Features:
        - Admin-only access control with permission validation
        - Isolated execution environment in Ray tasks
        - Resource limits through Ray remote options
        - Timeout protection to prevent infinite execution
        - Comprehensive error handling and logging

        Args:
            code: Python source code string containing the function to execute.
                 Must define a function with the specified function_name.
            function_name: Name of the function to execute from the source code.
                          Defaults to "analyze" for conventional usage.
            func_bytes: Pre-serialized function bytes using cloudpickle.
                       Used when mode='pickle' for optimized execution.
            mode: Execution mode determining input format:
                 - 'source': Execute function from source code string
                 - 'pickle': Execute pre-serialized function from func_bytes
            remote_options: Ray remote decorator options for resource allocation:
                          - num_cpus: Number of CPU cores to allocate
                          - num_gpus: Number of GPU devices to allocate  
                          - memory: Memory allocation in bytes
                          - runtime_env: Python environment configuration
            args: Positional arguments to pass to the target function
            kwargs: Keyword arguments to pass to the target function
            write_stdout: Optional async callback function for streaming stdout output.
                         Called with each line of stdout as it's generated.
            write_stderr: Optional async callback function for streaming stderr output.
                         Called with each line of stderr as it's generated.
            context: Request context containing user information for permission checking

        Returns:
            Dict containing comprehensive execution results:
                - result: Function return value (if execution successful)
                - error: Error message string (if execution failed)
                - traceback: Full Python traceback (if execution failed)
                - stdout: Captured stdout output as string
                - stderr: Captured stderr output as string
                - available_functions: List of functions found in source (if function not found)

        Raises:
            PermissionError: If user is not authorized to execute Python code
            ValueError: If both code and func_bytes are None, or invalid mode specified
            RuntimeError: If Ray cluster is not initialized or task execution fails
            TimeoutError: If function execution exceeds 600 second timeout
            Exception: If function deserialization or execution encounters critical errors

        Example:
            ```python
            # Execute source code with resource allocation
            result = await worker.execute_python_code(
                code=\"\"\"
                def analyze(data):
                    return {"mean": sum(data) / len(data)}
                \"\"\",
                args=[[1, 2, 3, 4, 5]],
                remote_options={"num_cpus": 2, "memory": 1000000000}
            )
            print(result["result"])  # {"mean": 3.0}
            ```

        Note:
            This method requires admin privileges as it allows arbitrary code execution
            in the distributed Ray environment. The execution is isolated but should
            only be used by trusted administrators.
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
        await self.notify()
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
