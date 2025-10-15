import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from haikunator import Haikunator
from hypha_rpc.rpc import RemoteService
from hypha_rpc.utils.schema import schema_method
from pydantic import Field
from ray import serve

from bioengine import __version__
from bioengine.applications.app_builder import AppBuilder
from bioengine.ray import RayCluster
from bioengine.utils import (
    check_permissions,
    create_application_from_files,
    create_context,
    create_logger,
    ensure_applications_collection,
)


class AppsManager:
    """
    Manages Ray Serve deployments using Hypha artifact manager integration.

    This class provides comprehensive management of application deployments by integrating
    with the Hypha artifact manager to deploy containerized applications as Ray Serve
    deployments. It handles the complete lifecycle from artifact creation through
    deployment management to cleanup, with robust error handling and permission control.

    The AppsManager orchestrates:
    - Artifact creation and management through Hypha
    - Ray Serve deployment lifecycle management with proper task tracking
    - Dynamic service registration and exposure
    - Resource allocation and monitoring
    - Permission-based access control
    - Graceful deployment and undeployment operations with automatic cleanup

    Key Features:
    - Artifact-based deployment system with versioning support
    - Dynamic Hypha service registration for deployed applications
    - Multi-mode deployment configurations
    - Resource-aware deployment with CPU/GPU allocation
    - Permission-based access control for deployments
    - Comprehensive error handling and logging with proper state management
    - Startup deployment automation
    - Real-time deployment status monitoring
    - Robust deployment task tracking and cleanup

    Deployment State Management:
    The class maintains deployment state in `_deployed_applications` dictionary where each
    entry contains deployment metadata, resource allocation, and an active asyncio task
    reference. Proper cleanup ensures no resource leaks or orphaned deployments.

    Attributes:
        ray_cluster (RayCluster): Ray cluster manager instance
        admin_users (List[str]): List of user emails with admin permissions
        apps_cache_dir (Path): Cache directory for deployment artifacts
        server: Hypha server connection
        artifact_manager: Hypha artifact manager service proxy
        app_builder (AppBuilder): Application builder instance
        startup_applications (List[dict]): Deployments to start automatically
        logger: Logger instance for deployment operations
        _deployed_applications (Dict): Internal tracking of active deployments with task references
        _deployment_lock (asyncio.Lock): Ensures single deployment operation at a time

    Parameter Conventions (API):
        - application_kwargs: Dictionary of keyword arguments for each deployment class
        - application_env_vars: Dictionary of environment variables for each deployment class
        - hypha_token: Hypha authentication token for application deployments (set as env var 'HYPHA_TOKEN')
            Used for authenticating to BioEngine datasets and Hypha APIs as the logged-in user.
    """

    def __init__(
        self,
        ray_cluster: RayCluster,
        apps_cache_dir: Union[str, Path] = f"{os.environ['HOME']}/.bioengine/apps",
        data_server_url: Optional[str] = None,
        data_server_workspace: str = "public",
        startup_applications: Optional[List[dict]] = None,
        # Logger
        log_file: Optional[str] = None,
        debug: bool = False,
    ):
        """
        Initialize the AppsManager with configuration for managing BioEngine application deployments.

        Sets up the deployment manager to handle BioEngine applications using Ray Serve and
        Hypha artifact manager integration. Configures directory paths based on the Ray
        cluster mode and initializes all necessary components.

        Directory Configuration by Mode:
        - SLURM mode: Uses fixed paths (/home/<user>/.bioengine/apps) for container compatibility
        - Single-machine mode: Uses provided paths, converted to absolute paths
        - External-cluster mode: Uses provided paths directly

        Args:
            ray_cluster: Ray cluster manager instance for compute resource management
            apps_cache_dir: Directory for caching application artifacts and build files
            data_server_url: URL for the data server
            data_server_workspace: Workspace on the data server (default: "public")
            startup_applications: List of application configurations to deploy automatically
                                 when the manager initializes
            log_file: Optional path to log file for deployment operations
            debug: Enable detailed debug logging for troubleshooting

        Raises:
            ValueError: If the Ray cluster mode is not supported
            Exception: If component initialization fails

        Note:
            Admin users are set later during the initialize() call, not during construction.
        """
        # Set up logging
        self.logger = create_logger(
            name="AppsManager",
            level=logging.DEBUG if debug else logging.INFO,
            log_file=log_file,
        )

        # Initialize components
        self.ray_cluster = ray_cluster

        self.app_builder = AppBuilder(
            apps_cache_dir=apps_cache_dir,
            data_server_url=data_server_url,
            data_server_workspace=data_server_workspace,
            log_file=log_file,
            debug=debug,
        )

        self.haikunator = Haikunator()

        # Initialize state variables
        self.server = None
        self.artifact_manager = None
        self.admin_users = None
        self.worker_service_id = None
        self.startup_applications = startup_applications
        self._deployment_lock = asyncio.Lock()
        self._deployed_applications = {}
        self.debug = debug

    def _check_initialized(self) -> None:
        """
        Verify that the manager has been properly initialized with server connections.

        Ensures both the Hypha server connection and artifact manager service are available
        before attempting any operations that require them.

        Raises:
            RuntimeError: If server connection or artifact manager is not initialized
        """
        if not self.server:
            raise RuntimeError(
                "Hypha server connection not available. Call initialize() first."
            )
        if not self.artifact_manager:
            raise RuntimeError(
                "Artifact manager not initialized. Call initialize() first."
            )

    def _get_full_artifact_id(self, artifact_id: str) -> str:
        """
        Convert a potentially short artifact ID to its full workspace-qualified form.

        Hypha artifacts can be referenced either by their full ID (workspace/artifact-name)
        or by just the artifact name if it belongs to the current workspace. This method
        ensures we always have the full form for consistent handling.

        Args:
            artifact_id: The artifact identifier to convert

        Returns:
            Full artifact ID in the format 'workspace/artifact-name'

        Raises:
            TypeError: If artifact_id is not a string

        Examples:
            - "my-app" → "workspace123/my-app"
            - "workspace123/my-app" → "workspace123/my-app" (unchanged)
        """
        if not isinstance(artifact_id, str):
            raise TypeError("Artifact ID must be a string.")

        if "/" in artifact_id:
            return artifact_id
        else:
            # If artifact_id does not contain a slash, prepend the workspace
            return f"{self.server.config.workspace}/{artifact_id}"

    async def _generate_application_id(self) -> str:
        """
        Generate a unique identifier for a new application deployment.

        Creates a human-readable, unique application ID using the Haikunator library
        which generates names like "delighted-mouse-a4f2". Ensures the generated ID
        is not already in use by checking both internal tracking and Ray Serve status.

        ID Format:
        - Two descriptive words separated by hyphens
        - 4-character hexadecimal suffix for uniqueness
        - Example: "ambitious-whale-f3c1"

        Returns:
            Unique application ID that can be used for deployment

        Note:
            This method will retry generation until a unique ID is found.
        """
        while True:
            application_id = self.haikunator.haikunate(
                delimiter="-",  # Use hyphen as delimiter
                token_length=4,  # 4-character random suffix
                token_hex=True,  # Use hexadecimal characters for the suffix
            )
            if application_id in self._deployed_applications:
                # Application ID already exists, generate a new one
                continue

            serve_status = await asyncio.to_thread(serve.status)
            if application_id not in serve_status.applications:
                # Application ID is unique and not already deployed
                break

        return application_id

    async def _check_resources(
        self, application_id: str, required_resources: Dict[str, int]
    ) -> None:
        """
        Verify that sufficient resources are available for deploying an application.

        Checks current Ray cluster resource availability against the application's
        requirements for CPU, GPU, and memory. For SLURM clusters, also considers
        the ability to scale up by adding additional worker nodes.

        Resource Validation Process:
        1. Waits for Ray cluster to be ready and connected
        2. Checks each node for sufficient available resources
        3. For SLURM mode: Evaluates if new workers can be spawned if needed
        4. For external clusters: Issues warning but allows deployment

        Args:
            application_id: ID of the application being deployed (for logging)
            required_resources: Dictionary containing 'num_cpus', 'num_gpus', and 'memory' requirements

        Raises:
            ValueError: If insufficient resources are available and cluster cannot scale

        Note:
            External clusters are assumed to have autoscaling capabilities, so only
            warnings are issued rather than blocking deployment.
        """
        # Check if the required resources are available
        insufficient_resources = True

        # Wait for Ray cluster to be ready
        await self.ray_cluster.is_ready.wait()

        for node_resource in self.ray_cluster.status["nodes"].values():
            if (
                node_resource["available_cpu"] >= required_resources["num_cpus"]
                and node_resource["available_gpu"] >= required_resources["num_gpus"]
                and node_resource["available_memory"] >= required_resources["memory"]
            ):
                insufficient_resources = False

        if self.ray_cluster.mode == "slurm" and insufficient_resources:
            # Check if additional SLURM workers can be created that meet the resource requirements
            # TODO: Remove resource check when SLURM workers can adjust resources dynamically
            num_worker_jobs = await self.ray_cluster.slurm_workers.get_num_worker_jobs()
            default_num_cpus = self.ray_cluster.slurm_workers.default_num_cpus
            default_num_gpus = self.ray_cluster.slurm_workers.default_num_gpus
            default_memory = (
                self.ray_cluster.slurm_workers.default_mem_in_gb_per_cpu
                * default_num_cpus
            )
            if (
                num_worker_jobs < self.ray_cluster.slurm_workers.max_workers
                and default_num_cpus >= required_resources["num_cpus"]
                and default_num_gpus >= required_resources["num_gpus"]
                and default_memory >= required_resources["memory"]
            ):
                insufficient_resources = False

        if insufficient_resources:
            if self.ray_cluster.mode != "external-cluster":
                raise ValueError(
                    f"Insufficient resources for application '{application_id}'. "
                    f"Requested: {required_resources}"
                )
            else:
                self.logger.warning(
                    f"Currently insufficient resources for application '{application_id}'. "
                    "Assuming Ray autoscaling is available. "
                    f"Requested resources: {required_resources}"
                )

    async def _deploy_application(
        self,
        application_id: str,
    ) -> None:
        """
        Execute the deployment of a BioEngine application and manage its lifecycle.

        This is the core deployment method that runs as a long-lived asyncio task to
        manage an application's entire lifecycle from startup to cleanup. It handles
        the actual deployment process, monitors health, and ensures proper resource
        cleanup when the deployment ends.

        Deployment Lifecycle:
        1. Reset deployment status and prepare for deployment
        2. Use the pre-built application from _deployed_applications or build if not available
        3. Deploy to Ray Serve with unique routing
        4. Keep the deployment active until cancelled
        5. Perform cleanup when task is cancelled or fails

        The method runs indefinitely to maintain the deployment until explicitly
        cancelled. All cleanup is handled in the finally block to ensure proper
        resource management regardless of how the deployment ends.

        Note:
            Application building and resource validation are now done in deploy_application()
            before this task is created, providing immediate user feedback for errors.
            The built app is stored in _deployed_applications["built_app"] to avoid duplicate builds.

        Args:
            application_id: Unique identifier for the application deployment

        Raises:
            RuntimeError: If deployment validation fails
            Exception: If deployment startup or initialization fails

        Note:
            This method is designed to run as a managed asyncio task and should
            only exit when cancelled or on unrecoverable error.
        """
        try:
            # Reset deployment status (if already deployed)
            self._deployed_applications[application_id]["is_deployed"].clear()

            artifact_id = self._deployed_applications[application_id]["artifact_id"]
            version = self._deployed_applications[application_id]["version"]

            # Use the pre-built app from _deployed_applications
            # This was built and validated in deploy_application() before this task was created
            app = self._deployed_applications[application_id]["built_app"]

            # Run the deployment in Ray Serve with unique route prefix
            await asyncio.to_thread(
                serve.run,
                target=app,
                name=application_id,
                route_prefix=f"/{application_id}",
                blocking=False,
            )

            # Update application metadata in the internal state
            self._deployed_applications[application_id]["display_name"] = app.metadata[
                "name"
            ]
            self._deployed_applications[application_id]["description"] = app.metadata[
                "description"
            ]
            self._deployed_applications[application_id]["application_resources"] = (
                app.metadata["resources"]
            )
            self._deployed_applications[application_id]["authorized_users"] = (
                app.metadata["authorized_users"]
            )
            self._deployed_applications[application_id]["available_methods"] = (
                app.metadata["available_methods"]
            )

            # Track the application in the internal state
            self.logger.info(
                f"Successfully completed deployment of application '{application_id}' from "
                f"artifact '{artifact_id}', version '{version}'."
            )

            # Mark the application as deployed
            self._deployed_applications[application_id]["is_deployed"].set()

            # Keep the deployment task running until cancelled
            event = asyncio.Event()
            await event.wait()

        except asyncio.CancelledError:
            self.logger.info(
                f"Deployment task for application '{application_id}' was cancelled."
            )
        except Exception as e:
            self.logger.error(
                f"Failed to deploy application '{application_id}' with error: {e}"
            )
            try:
                serve_status = await asyncio.to_thread(serve.status)
                application = serve_status.applications.get(application_id)
                if application:
                    for deployment_name, deployment in application.deployments.items():
                        if deployment.status.value == "UNHEALTHY":
                            self.logger.error(
                                f"Ray Serve application '{application_id}' deployment "
                                f"'{deployment_name}' reported error:\n{deployment.message}",
                            )
            except Exception as status_error:
                self.logger.error(
                    f"Failed to get Ray Serve status for application '{application_id}': {status_error}"
                )
        finally:
            # Signal other processes to stop waiting for this deployment
            self._deployed_applications[application_id]["is_deployed"].set()

            # Cleanup: Remove from Ray Serve and update tracking
            if self._deployed_applications[application_id]["remove_on_exit"]:
                try:
                    await asyncio.to_thread(serve.delete, application_id)
                    self.logger.debug(
                        f"Deleted Ray Serve application '{application_id}'."
                    )
                except Exception as delete_err:
                    self.logger.error(
                        f"Error deleting Ray Serve application '{application_id}': {delete_err}"
                    )

                # Remove from deployment tracking
                self._deployed_applications.pop(application_id, None)
                self.logger.debug(
                    f"Removed application '{application_id}' from deployment tracking."
                )

                self.logger.info(
                    f"Undeployment of application '{application_id}' completed."
                )

    async def initialize(
        self, server: RemoteService, admin_users: List[str], worker_service_id: str
    ) -> None:
        """
        Initialize the deployment manager with Hypha server connections and admin permissions.

        Sets up the complete infrastructure needed for application deployment management
        including server connections, artifact manager access, and automatic startup
        application deployment.

        Initialization Process:
        1. Establishes Hypha server connection and stores admin user list
        2. Connects to the artifact manager service for application storage
        3. Configures the AppBuilder with server connections
        4. Deploys any configured startup applications automatically
        5. Waits for all startup deployments to be ready

        Args:
            server: Active Hypha server connection instance with proper authentication
            admin_users: List of user IDs or email addresses with administrative permissions
                        for managing applications and deployments
            worker_service_id: BioEngine worker service ID

        Raises:
            Exception: If server connection fails, artifact manager is unavailable,
                      or startup application deployment fails
        """
        if not server or not isinstance(server, RemoteService):
            raise ValueError("Invalid server connection provided.")
        if not isinstance(admin_users, list) or not all(
            isinstance(user, str) for user in admin_users
        ):
            raise ValueError("Invalid admin_users list provided.")

        # Store server connection and list of admin users
        self.server = server
        self.admin_users = admin_users

        try:
            # Get artifact manager service
            self.artifact_manager = await self.server.get_service(
                "public/artifact-manager"
            )
            self.logger.info("Successfully connected to artifact manager.")
        except Exception as e:
            self.logger.error(f"Error initializing Ray Deployment Manager: {e}")
            self.server = None
            self.artifact_manager = None
            raise

        # Ensure applications collection exists
        workspace = self.server.config.workspace
        await ensure_applications_collection(
            artifact_manager=self.artifact_manager,
            workspace=workspace,
            logger=self.logger,
        )

        # Initialize the AppBuilder with the server and artifact manager
        self.app_builder.complete_initialization(
            server=self.server,
            artifact_manager=self.artifact_manager,
            worker_service_id=worker_service_id,
            serve_http_url=self.ray_cluster.serve_http_url,
        )

        # Deploy any startup applications if provided
        if self.startup_applications:
            self.logger.info(
                f"Deploying {len(self.startup_applications)} startup application(s)..."
            )

            # Pass the worker owner's token to the startup applications (if not already set)
            startup_applications_token = await self.server.generate_token(
                {
                    "workspace": self.server.config.workspace,
                    "permission": "read_write",
                    "expires_in": 3600 * 24 * 30,  # support application for 30 days
                }
            )
            for app_config in self.startup_applications:
                if "hypha_token" not in app_config:
                    app_config["hypha_token"] = startup_applications_token

            # Deploy the startup applications
            admin_context = create_context(admin_users[0])
            application_ids = await self.deploy_applications(
                app_configs=self.startup_applications,
                context=admin_context,
            )

            # Wait for all startup applications to be deployed
            for application_id in application_ids:
                app_info = self._deployed_applications.get(application_id)
                if app_info:
                    await app_info["is_deployed"].wait()

    async def get_status(self) -> Dict[str, Union[str, list, dict]]:
        """
        Retrieve comprehensive status information for all deployed applications.

        Provides detailed information about each currently tracked application deployment,
        cross-referencing internal state with Ray Serve status to ensure accuracy.
        Includes deployment metadata, resource usage, service endpoints, and health status.

        Status Information Includes:
        - Application metadata (name, description, artifact details)
        - Deployment status and health from Ray Serve
        - Resource allocation and usage
        - Service endpoint information for client connections
        - Failure tracking and monitoring data
        - Replica states and deployment details

        Consistency Validation:
        The method validates deployment state by comparing internal tracking with
        Ray Serve status, providing warnings for any inconsistencies found.

        Returns:
            Dictionary mapping application IDs to their detailed status information.
            Empty dictionary if no applications are currently deployed.

        Raises:
            RuntimeError: If Ray cluster connection is unavailable

        Note:
            Only applications that exist in both internal tracking and are accessible
            via Ray Serve are included to ensure data accuracy.
        """
        output = {}

        if not self._deployed_applications:
            return output

        # Get status of actively running deployments
        await self.ray_cluster.check_connection()
        serve_status = await asyncio.to_thread(serve.status)

        for application_id, application_info in self._deployed_applications.items():
            application = serve_status.applications.get(application_id)

            if application:
                start_time = application.last_deployed_time_s
                status = application.status.value
                message = application.message
                deployments = {
                    class_name: {
                        "status": deployment_info.status.value,
                        "message": deployment_info.message,
                        "replica_states": deployment_info.replica_states,
                    }
                    for class_name, deployment_info in application.deployments.items()
                }

            else:
                start_time = None
                if application_info["is_deployed"].is_set():
                    status = "UNHEALTHY"
                    message = f"Application '{application_id}' is marked as deployed but not found in Ray Serve status."
                    self.logger.warning(
                        f"Application '{application_id}' for artifact '{application_info['artifact_id']}' "
                        "is marked as deployed but not found in Ray Serve status."
                    )
                else:
                    status = "NOT_STARTED"
                    message = (
                        f"Application '{application_id}' has not been deployed yet."
                    )
                deployments = {}

            # Construct the service IDs for the application using the replica IDs
            workspace = self.server.config.workspace
            replica_ids = (
                await self.ray_cluster.proxy_actor_handle.get_deployment_replica.remote(
                    app_name=application_id, deployment_name="BioEngineProxyDeployment"
                )
            )
            if replica_ids:
                worker_client_id = self.server.config.client_id
                service_ids = [
                    {
                        "websocket_service_id": f"{workspace}/{worker_client_id}-{replica_id}:{application_id}",
                        "webrtc_service_id": f"{workspace}/{worker_client_id}-{replica_id}:{application_id}-rtc",
                    }
                    for replica_id in replica_ids
                ]
            else:
                service_ids = [
                    {
                        "websocket_service_id": None,
                        "webrtc_service_id": None,
                    }
                ]

            # class ApplicationStatus(str, Enum):
            #     NOT_STARTED = "NOT_STARTED"
            #     DEPLOYING = "DEPLOYING"
            #     DEPLOY_FAILED = "DEPLOY_FAILED"
            #     RUNNING = "RUNNING"
            #     UNHEALTHY = "UNHEALTHY"
            #     DELETING = "DELETING"

            # class DeploymentStatus(str, Enum):
            #     UPDATING = "UPDATING"
            #     HEALTHY = "HEALTHY"
            #     UNHEALTHY = "UNHEALTHY"
            #     UPSCALING = "UPSCALING"
            #     DOWNSCALING = "DOWNSCALING"

            # class ReplicaState(str, Enum):
            #     STARTING = "STARTING"
            #     UPDATING = "UPDATING"
            #     RECOVERING = "RECOVERING"
            #     RUNNING = "RUNNING"
            #     STOPPING = "STOPPING"
            #     PENDING_MIGRATION = "PENDING_MIGRATION"

            output[application_id] = {
                "display_name": application_info["display_name"],
                "description": application_info["description"],
                "artifact_id": application_info["artifact_id"],
                "version": application_info["version"] or "latest",
                "start_time": start_time,
                "status": status,
                "message": message,
                "deployments": deployments,
                "consecutive_failures": application_info["consecutive_failures"],
                "application_kwargs": application_info["application_kwargs"],
                "application_env_vars": application_info["application_env_vars"],
                "gpu_enabled": application_info["disable_gpu"],
                "application_resources": application_info["application_resources"],
                "authorized_users": application_info["authorized_users"],
                "available_methods": application_info["available_methods"],
                "service_ids": service_ids,
                "last_updated_by": application_info["last_updated_by"],
            }

        return output

    async def monitor_applications(self) -> None:
        """
        Monitor the health of all deployed applications and handle failures automatically.

        Continuously monitors application health by checking Ray Serve status against
        internal deployment tracking. Handles failure recovery, redeployment, and
        cleanup of consistently failing applications.

        Monitoring Actions:
        - Tracks consecutive failures for each application
        - Triggers redeployment for applications experiencing issues
        - Removes applications that fail repeatedly (>3 consecutive failures)
        - Resets failure counters for healthy applications

        Failure Handling:
        - 1-3 failures: Attempts automatic redeployment
        - >3 failures: Removes application and stops monitoring
        - Logs warnings and status changes for debugging

        This method is typically called periodically by the worker to ensure
        application health and availability.

        Raises:
            Exception: If monitoring process encounters unrecoverable errors
        """
        if not self._deployed_applications:
            return

        try:

            # Get status of actively running deployments
            await self.ray_cluster.check_connection()
            serve_status = await asyncio.to_thread(serve.status)

            application_ids = list(self._deployed_applications.keys())
            for application_id in application_ids:
                application_info = self._deployed_applications.get(application_id)
                if not application_info:
                    # Application no longer tracked, skip monitoring
                    continue
                if not application_info["is_deployed"].is_set():
                    # Application not yet deployed, skip monitoring
                    continue

                # Get the application status from Ray Serve
                application = serve_status.applications.get(application_id)
                if not application:
                    # Application not found in Ray Serve, increment failure count and clear is_deployed
                    application_info["consecutive_failures"] += 1
                elif application.status.value == "DEPLOY_FAILED":
                    # If the application deployment failed, increment failure count
                    # Allow application to recover from transient issues (UNHEALTHY -> HEALTHY)
                    application_info["consecutive_failures"] += 1
                else:
                    # Reset consecutive failures if the application is healthy
                    application_info["consecutive_failures"] = 0

                if application_info["consecutive_failures"] > 3:
                    # Application has failed multiple times, undeploy and remove from tracking
                    self.logger.warning(
                        f"Application '{application_id}' for artifact '{application_info['artifact_id']}', "
                        f"version '{application_info['version']}' has failed multiple times. It will be "
                        "undeployed and removed from tracking."
                    )
                    application_info["deployment_task"].cancel()
                    self._deployed_applications.pop(application_id, None)

                elif application_info["consecutive_failures"] > 0:
                    # Application is experiencing issues, trigger redeployment
                    self.logger.warning(
                        f"Application '{application_id}' for artifact '{application_info['artifact_id']}' "
                        f"is experiencing issues. Consecutive failures: {application_info['consecutive_failures']}"
                    )

                    # Re-deploy the application
                    deployment_task = asyncio.create_task(
                        self._deploy_application(application_id=application_id),
                        name=f"Deploy_{application_id}",
                    )
                    # Overwrite the existing deployment task
                    application_info["deployment_task"] = deployment_task
        except Exception as e:
            self.logger.error(f"Error monitoring applications: {e}")
            raise e

    @schema_method
    async def list_applications(
        self,
        context: Dict[str, Any] = Field(
            ...,
            description="Authentication context containing user information, automatically provided by Hypha during service calls.",
        ),
    ) -> Dict[str, List[str]]:
        """
        Lists all available BioEngine application artifacts stored in the Hypha artifact manager.

        This method retrieves and returns information about all BioEngine applications that have been
        uploaded to the current workspace, including their manifest data and associated files.
        Each application artifact contains the necessary code, configuration, and dependencies
        needed to deploy a BioEngine application.

        Returns:
            Dictionary mapping artifact IDs to their metadata and files. Each entry contains:
            - manifest: Application configuration and deployment specifications
            - files: List of file names included in the artifact package

        Raises:
            PermissionError: If the user lacks admin permissions to list applications
            RuntimeError: If the Hypha server connection is not initialized
        """
        self._check_initialized()

        check_permissions(
            context=context,
            authorized_users=self.admin_users,
            resource_name=f"listing applications",
        )

        collection_id = f"{self.server.config.workspace}/applications"
        bioengine_apps_artifacts = await self.artifact_manager.list(collection_id)
        bioengine_apps = {}
        for artifact in bioengine_apps_artifacts:
            try:
                manifest = await self.artifact_manager.read(artifact.id)
                files = await self.artifact_manager.list_files(artifact.id)
                file_names = [file.name for file in files]
                bioengine_apps[artifact.id] = {
                    "manifest": manifest,
                    "files": file_names,
                }
            except Exception as e:
                self.logger.error(f"Error reading artifact '{artifact.id}': {e}")

        return bioengine_apps

    @schema_method
    async def create_application(
        self,
        files: List[dict] = Field(
            ...,
            description="List of application files to upload. Each file must be a dictionary with 'name' (string), 'content' (file content), and 'type' ('text' for text files or 'base64' for binary files). Must include a 'manifest.yaml' file with application configuration.",
        ),
        context: Dict[str, Any] = Field(
            ...,
            description="Authentication context containing user information, automatically provided by Hypha during service calls.",
        ),
    ) -> str:
        """
        Creates or updates a BioEngine application artifact in the Hypha artifact manager.

        This method allows you to upload a complete BioEngine application package including
        all necessary files, code, and configuration. The application can then be deployed
        to Ray Serve using the deploy_application method.

        Application Structure:
        - manifest.yaml: Required configuration file defining the application metadata,
          deployment settings, and entry points
        - Python files: Application code and dependencies
        - Data files: Any additional resources needed by the application

        The manifest.yaml file must contain:
        - id: Application identifier
        - type: Must be "ray-serve"
        - name: Human-readable application name
        - description: Application description
        - Additional deployment configuration

        Returns:
            The artifact ID of the created or updated application artifact

        Raises:
            ValueError: If manifest is missing, invalid, or artifact ID format is incorrect
            PermissionError: If user lacks admin permissions to create/modify applications
            RuntimeError: If artifact creation fails or Hypha connection is unavailable
        """
        self._check_initialized()

        check_permissions(
            context=context,
            authorized_users=self.admin_users,
            resource_name=f"creating or modifying an application",
        )

        # Create or update the artifact using the utility function
        try:
            created_artifact_id = await create_application_from_files(
                artifact_manager=self.artifact_manager,
                files=files,
                workspace=self.server.config.workspace,
                user_id=context["user"]["id"],
                logger=self.logger,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create/update artifact: {e}")

        # Verify the artifact is in the collection
        available_artifacts = await self.list_applications(context=context)
        if created_artifact_id not in available_artifacts:
            raise ValueError(
                f"Artifact '{created_artifact_id}' could not be created or is not in the collection."
            )

        self.logger.info(
            f"Successfully created/updated application artifact '{created_artifact_id}'."
        )

        return created_artifact_id

    @schema_method
    async def delete_application(
        self,
        artifact_id: str = Field(
            ...,
            description="Unique identifier of the application artifact to delete. Can be either the full artifact ID (workspace/artifact-name) or just the artifact name if it belongs to the current workspace.",
        ),
        context: Dict[str, Any] = Field(
            ...,
            description="Authentication context containing user information, automatically provided by Hypha during service calls.",
        ),
    ) -> None:
        """
        Permanently deletes a BioEngine application artifact from the Hypha artifact manager.

        This method removes the specified application artifact and all its associated files
        from the artifact manager. Once deleted, the artifact cannot be recovered and any
        running deployments using this artifact should be undeployed first.

        Warning: This operation is irreversible. Ensure you have backups if needed before deletion.

        Args:
            artifact_id: The artifact to delete. Must belong to the current workspace.

        Raises:
            ValueError: If the artifact doesn't exist, doesn't belong to current workspace,
                       or cannot be deleted
            PermissionError: If user lacks admin permissions to delete applications
            RuntimeError: If Hypha connection is not available
        """
        self._check_initialized()

        # Check user permissions
        check_permissions(
            context=context,
            authorized_users=self.admin_users,
            resource_name=f"deleting the artifact '{artifact_id}'",
        )

        # Get the full artifact ID
        self.logger.debug(f"Deleting artifact '{artifact_id}'...")
        artifact_id = self._get_full_artifact_id(artifact_id)

        # Ensure the artifact is in the correct workspace
        workspace = self.server.config.workspace
        if not artifact_id.startswith(f"{workspace}/"):
            raise ValueError(
                f"Artifact ID '{artifact_id}' does not belong to the current workspace '{workspace}'."
            )

        # Delete the artifact
        await self.artifact_manager.delete(artifact_id)

        # Verify deletion
        available_artifacts = await self.list_applications(context=context)
        if artifact_id in available_artifacts:
            raise ValueError(
                f"Artifact '{artifact_id}' could not be deleted. It still exists in the collection."
            )

        self.logger.info(f"Successfully deleted artifact '{artifact_id}'.")

    @schema_method
    async def deploy_application(
        self,
        artifact_id: str = Field(
            ...,
            description="Identifier of the application artifact to deploy. Can be either the full artifact ID (workspace/artifact-name) or just the artifact name if it belongs to the current workspace.",
        ),
        version: Optional[str] = Field(
            None,
            description="Specific version of the artifact to deploy. If not provided, deploys the latest available version of the artifact.",
        ),
        application_id: Optional[str] = Field(
            None,
            description="Unique identifier for this deployment instance. If not provided, a random unique ID will be automatically generated. Use this to manage multiple deployments of the same artifact.",
        ),
        application_kwargs: Optional[Dict[str, Dict[str, Any]]] = Field(
            None,
            description="Advanced deployment configuration parameters. Dictionary where keys are deployment class names and values are dictionaries of keyword arguments to pass to those classes during initialization.",
        ),
        application_env_vars: Optional[Dict[str, Dict[str, str]]] = Field(
            None,
            description="Environment variables to set for each deployment. Dictionary where keys are deployment class names and values are dictionaries of environment variable names and their values.",
        ),
        hypha_token: str = Field(
            None,
            description="Hypha connection token for authentication. The token will be set as environment variable 'HYPHA_TOKEN' in the application deployments. An already existing environment variable named 'HYPHA_TOKEN' will not be overwritten. The token is used to authenticate to BioEngine datasets and enables Hypha API calls as logged in user.",
        ),
        disable_gpu: bool = Field(
            False,
            description="Set to true to disable GPU usage for this deployment, forcing it to run on CPU only. Useful for testing or when GPU resources are limited.",
        ),
        max_ongoing_requests: int = Field(
            10,
            description="Maximum number of concurrent requests this application instance can handle simultaneously. Higher values allow more parallelism but use more memory.",
        ),
        context: Dict[str, Any] = Field(
            ...,
            description="Authentication context containing user information, automatically provided by Hypha during service calls.",
        ),
    ) -> str:
        """
        Deploys a BioEngine application from an artifact to Ray Serve with comprehensive lifecycle management.

        This method downloads the specified application artifact, configures it according to the
        provided parameters, and creates a running deployment in the Ray Serve cluster. The
        deployment will be automatically registered as a service and made available for use.

        Deployment Process:
        1. Downloads and validates the application artifact
        2. Checks resource availability (CPU, GPU, memory)
        3. Configures the Ray Serve deployment with specified parameters
        4. Starts the deployment and monitors its health
        5. Registers the application as a callable service

        The deployment runs asynchronously - this method returns immediately after starting
        the deployment process. Use get_status() to monitor deployment progress and health.

        Resource Management:
        - Automatically scales Ray cluster if needed (SLURM mode)
        - Validates resource requirements before deployment
        - Supports both CPU and GPU deployments
        - Configurable request concurrency limits

        Returns:
            Unique application ID for the created deployment. Use this ID to monitor,
            update, or undeploy the application.

        Raises:
            ValueError: If artifact doesn't exist, deployment configuration is invalid,
                       or insufficient resources are available
            PermissionError: If user lacks admin permissions to deploy applications
            RuntimeError: If Ray cluster is unavailable or deployment initialization fails
        """
        # Only allow one deployment task creation at a time
        # This ensures unique application_id and prevents race conditions
        async with self._deployment_lock:
            self._check_initialized()

            # Get the full artifact ID
            artifact_id = self._get_full_artifact_id(artifact_id)

            # Check user permissions
            check_permissions(
                context=context,
                authorized_users=self.admin_users,
                resource_name=f"deploying an application from artifact '{artifact_id}'",
            )
            user_id = context["user"]["id"]

            # Verify Ray is initialized
            await self.ray_cluster.check_connection()

            # Generate a new application ID if not provided
            if not application_id:
                application_id = await self._generate_application_id()

            # Validate application_kwargs
            if application_kwargs is not None:
                if not isinstance(application_kwargs, dict):
                    raise ValueError(
                        "application_kwargs must be a dictionary of keyword arguments."
                    )
                # Ensure all values are dictionaries
                for key, value in application_kwargs.items():
                    if not isinstance(value, dict):
                        raise ValueError(
                            f"Value for '{key}' in application_kwargs must be a dictionary."
                        )

                kwargs_str = ", ".join(
                    f"{deployment_class}("
                    + ", ".join(
                        [f"{key}={value!r}" for key, value in deployment_kwargs.items()]
                    )
                    + ")"
                    for deployment_class, deployment_kwargs in application_kwargs.items()
                )
            else:
                application_kwargs = {}
                kwargs_str = "None"

            if application_env_vars is not None:
                # Validate application_env_vars
                if not isinstance(application_env_vars, dict):
                    raise ValueError(
                        "application_env_vars must be a dictionary of environment variables."
                    )
                for key, value in application_env_vars.items():
                    if not isinstance(value, dict):
                        raise ValueError(
                            f"Value for '{key}' in application_env_vars must be a dictionary."
                        )

                # Construct the environment variables string, hide secret env vars starting with underscore as *****
                env_vars_str = ", ".join(
                    f"{deployment_class}: "
                    + " ".join(
                        [
                            f"{key}={value!r}"
                            for key, value in env_vars.items()
                            if not key.startswith("_")
                        ]
                    )
                    + " ".join(
                        [
                            f'{key}="*****"'
                            for key in env_vars.keys()
                            if key.startswith("_")
                        ]
                    )
                    for deployment_class, env_vars in application_env_vars.items()
                )
            else:
                application_env_vars = {}
                env_vars_str = "None"

            if application_id not in self._deployed_applications:
                # Create a new application if application_id is not provided
                self.logger.info(
                    f"User '{user_id}' is deploying new application '{application_id}' from artifact '{artifact_id}', "
                    f"version '{version}'; kwargs: {kwargs_str}; env_vars: {env_vars_str}"
                )
            else:
                # If already deployed, cancel the existing deployment task to update deployment in a new task
                application_info = self._deployed_applications[application_id]
                self.logger.info(
                    f"User '{user_id}' is updating existing application '{application_id}' from artifact '{artifact_id}', "
                    f"version '{version}'; kwargs: {kwargs_str}; env_vars: {env_vars_str}"
                )
                application_info["remove_on_exit"] = False
                application_info["deployment_task"].cancel()
                timeout = 60
                try:
                    await asyncio.wait_for(
                        application_info["is_deployed"].wait(), timeout=timeout
                    )
                except asyncio.TimeoutError:
                    self.logger.warning(
                        f"Cancellation of existing deployment task for application '{application_id}' "
                        f"did not finish in time ({timeout} seconds). Proceeding with new deployment."
                    )

            # Build the application first to validate format and catch errors early
            app = await self.app_builder.build(
                application_id=application_id,
                artifact_id=artifact_id,
                version=version,
                application_kwargs=application_kwargs,
                application_env_vars=application_env_vars,
                hypha_token=hypha_token,
                disable_gpu=disable_gpu,
                max_ongoing_requests=max_ongoing_requests,
            )

            # Check resources before creating deployment task
            self.logger.debug(
                f"Checking resources for application '{application_id}': {app.metadata['resources']}"
            )
            await self._check_resources(
                application_id=application_id,
                required_resources=app.metadata["resources"],
            )

            self._deployed_applications[application_id] = {
                "display_name": app.metadata["name"],
                "description": app.metadata["description"],
                "artifact_id": artifact_id,
                "version": version,
                "application_kwargs": application_kwargs,
                "application_env_vars": application_env_vars,
                "hypha_token": hypha_token,
                "disable_gpu": disable_gpu,
                "max_ongoing_requests": max_ongoing_requests,
                "application_resources": app.metadata["resources"],
                "authorized_users": app.metadata["authorized_users"],
                "available_methods": app.metadata["available_methods"],
                "last_updated_by": user_id,
                "deployment_task": None,  # Track the deployment task
                "is_deployed": asyncio.Event(),  # Track if the deployment has been started
                "remove_on_exit": not self.debug,  # Remove on exit unless in debug mode
                "consecutive_failures": 0,  # Track consecutive failures for monitoring
                "built_app": app,  # Store the pre-built app to avoid duplicate builds
            }

            # Create and start the deployment task
            deployment_task = asyncio.create_task(
                self._deploy_application(application_id=application_id),
                name=f"Deploy_{application_id}",
            )
            self._deployed_applications[application_id][
                "deployment_task"
            ] = deployment_task

            return application_id

    @schema_method
    async def deploy_applications(
        self,
        app_configs: List[dict] = Field(
            ...,
            description=(
                "List of application deployment configurations. Each configuration must be a dictionary containing "
                "'artifact_id' (required) and optionally 'version', 'application_id', 'application_kwargs', 'application_env_vars', 'hypha_token', "
                "'disable_gpu', and 'max_ongoing_requests'. Allows batch deployment of multiple applications with different settings."
            ),
        ),
        context: Dict[str, Any] = Field(
            ...,
            description="Authentication context containing user information, automatically provided by Hypha during service calls.",
        ),
    ) -> List[str]:
        """
        Deploys multiple BioEngine applications simultaneously from a list of configurations.

        This method allows you to deploy several applications at once, each with their own
        configuration settings. It's useful for setting up complex environments or deploying
        related applications together.

        Configuration Format:
        Each app_config dictionary supports the same parameters as deploy_application:
        - artifact_id (required): The application artifact to deploy
        - version (optional): Specific version to deploy
        - application_id (optional): Custom deployment ID
        - application_kwargs (optional): Keyword arguments for deployment class(es)
        - application_env_vars (optional): Environment variables for deployment class(es)
        - disable_gpu (optional): Force CPU-only deployment
        - max_ongoing_requests (optional): Concurrency limit

        Example Configuration:
        ```
        [
            {
                "artifact_id": "my-workspace/cell-segmentation",
                "version": "v1.2.0",
                "disable_gpu": false,
                "max_ongoing_requests": 5
            },
            {
                "artifact_id": "my-workspace/image-analysis",
                "application_kwargs": {"ModelDeployment": {"batch_size": 16}}
            }
        ]
        ```

        Returns:
            List of application IDs for all successfully started deployments.
            Use these IDs to monitor, manage, or undeploy the applications.

        Raises:
            ValueError: If any configuration is invalid or missing required fields
            PermissionError: If user lacks admin permissions to deploy applications
            RuntimeError: If Ray cluster connection fails or any deployment cannot start
        """
        if not isinstance(app_configs, list) or not app_configs:
            raise ValueError("Provided app_configs must be a non-empty list.")

        application_ids = []
        for app_config in app_configs:
            if not isinstance(app_config, dict):
                raise ValueError("Each app_config must be a dictionary.")

            if "artifact_id" not in app_config:
                raise ValueError("Each app_config must contain an 'artifact_id'.")

            application_id = await self.deploy_application(
                artifact_id=app_config["artifact_id"],
                version=app_config.get("version"),
                application_id=app_config.get("application_id"),
                application_kwargs=app_config.get("application_kwargs"),
                application_env_vars=app_config.get("application_env_vars"),
                hypha_token=app_config.get("hypha_token"),
                disable_gpu=app_config.get("disable_gpu", False),
                max_ongoing_requests=app_config.get("max_ongoing_requests", 10),
                context=context,
            )
            application_ids.append(application_id)

        return application_ids

    @schema_method
    async def undeploy_application(
        self,
        application_id: str = Field(
            ...,
            description="Unique identifier of the deployed application to remove. This is the application ID that was returned when the application was deployed using deploy_application().",
        ),
        context: Dict[str, Any] = Field(
            ...,
            description="Authentication context containing user information, automatically provided by Hypha during service calls.",
        ),
    ) -> None:
        """
        Gracefully removes a deployed application from the Ray Serve cluster.

        This method stops the specified application deployment, cleans up all associated
        resources, and removes it from the internal tracking system. The application
        will no longer be available for serving requests after undeployment.

        Undeployment Process:
        1. Validates that the application is currently deployed
        2. Cancels the deployment task to initiate graceful shutdown
        3. Removes the application from Ray Serve
        4. Cleans up internal state and resource tracking
        5. Updates service registrations

        The undeployment happens asynchronously - this method returns immediately
        after initiating the shutdown process. The actual cleanup occurs in the
        background and may take a few moments to complete.

        Args:
            application_id: The deployment to remove. Must be currently deployed.

        Raises:
            RuntimeError: If the application is not currently deployed or if
                         Ray cluster connection is unavailable
            PermissionError: If user lacks admin permissions to undeploy applications

        Note:
            Once undeployed, the application ID becomes available for reuse.
            Any clients connecting to this application will receive errors.
        """
        self._check_initialized()

        # Verify Ray is initialized
        await self.ray_cluster.check_connection()

        # Check user permissions
        check_permissions(
            context=context,
            authorized_users=self.admin_users,
            resource_name=f"undeploying the application '{application_id}'",
        )
        user_id = context["user"]["id"]

        # Check if application is currently deployed
        if application_id not in self._deployed_applications:
            raise RuntimeError(
                f"Application '{application_id}' is currently not deployed."
            )

        self.logger.info(
            f"User '{user_id}' is starting undeployment of application '{application_id}'..."
        )
        # Ensure the application will be removed on exit
        self._deployed_applications[application_id]["remove_on_exit"] = True
        self._deployed_applications[application_id]["deployment_task"].cancel()

    @schema_method
    async def cleanup(
        self,
        context: Dict[str, Any] = Field(
            ...,
            description="Authentication context containing user information, automatically provided by Hypha during service calls.",
        ),
    ) -> None:
        """
        Removes all currently deployed applications and cleans up associated resources.

        This method performs a complete cleanup of the deployment environment by
        undeploying all currently running applications and removing them from both
        the Ray Serve cluster and internal tracking systems.

        Cleanup Process:
        1. Identifies all currently deployed applications
        2. Initiates graceful shutdown for each deployment
        3. Waits for all undeployment tasks to complete (with timeout)
        4. Cleans up any remaining resources and state
        5. Updates service registrations

        This operation is useful for:
        - Preparing for system maintenance
        - Clearing all deployments before shutdown
        - Recovering from problematic deployment states
        - Development and testing scenarios

        The cleanup process handles timeout scenarios gracefully and provides
        detailed logging of the cleanup status for each application.

        Raises:
            PermissionError: If user lacks admin permissions for cleanup operations
            RuntimeError: If Ray cluster connection is unavailable

        Note:
            This operation cannot be undone. All running applications will be stopped
            and will need to be redeployed manually if needed again.
        """
        # Check if any deployments exist
        if not self._deployed_applications:
            self.logger.info("No applications are currently deployed.")
            return

        self._check_initialized()

        # Check user permissions
        check_permissions(
            context=context,
            authorized_users=self.admin_users,
            resource_name="cleaning up all deployments",
        )
        user_id = context["user"]["id"]

        self.logger.info(f"User '{user_id}' is starting cleanup of all deployments...")

        # Cancel all deployment tasks
        for artifact_id in list(self._deployed_applications.keys()):
            await self.undeploy_application(artifact_id, context)

        # Wait for all undeployment tasks to complete
        failed_attempts = 0
        for artifact_id in list(self._deployed_applications.keys()):
            deployment_info = self._deployed_applications.get(artifact_id)
            if deployment_info:
                try:
                    await asyncio.wait_for(
                        deployment_info["deployment_task"],
                        timeout=60,
                    )
                except asyncio.TimeoutError:
                    failed_attempts += 1
                    self.logger.error(
                        f"Deletion of deployment task for artifact '{artifact_id}' did not finish in time."
                    )

        if failed_attempts != 0:
            self.logger.warning(
                f"Failed to clean up all deployments, {failed_attempts} remaining."
            )
