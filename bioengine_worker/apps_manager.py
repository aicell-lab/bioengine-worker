import asyncio
import base64
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import httpx
import yaml
from haikunator import Haikunator
from hypha_rpc.rpc import RemoteService
from hypha_rpc.utils.schema import schema_method
from ray import serve

from bioengine_worker import __version__
from bioengine_worker.app_builder import AppBuilder
from bioengine_worker.ray_cluster import RayCluster
from bioengine_worker.utils import check_permissions, create_context, create_logger


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
        apps_data_dir (Path): Data directory for deployment access
        server: Hypha server connection
        artifact_manager: Hypha artifact manager service proxy
        app_builder (AppBuilder): Application builder instance
        startup_applications (List[dict]): Deployments to start automatically
        logger: Logger instance for deployment operations
        _deployed_applications (Dict): Internal tracking of active deployments with task references
        _deployment_lock (asyncio.Lock): Ensures single deployment operation at a time
    """

    def __init__(
        self,
        ray_cluster: RayCluster,
        token: str,
        apps_cache_dir: str = "/tmp/bioengine/apps",
        apps_data_dir: str = "/data",
        startup_applications: Optional[List[dict]] = None,
        # Logger
        log_file: Optional[str] = None,
        debug: bool = False,
    ):
        """
        Initialize the Ray Deployment Manager.

        Sets up the deployment manager with the specified configuration and
        initializes state variables for tracking deployments.

        Args:
            ray_cluster: RayCluster instance for managing compute resources
            token: Authentication token for service access
            apps_cache_dir: Caching directory used in Ray Serve deployments
            apps_data_dir: Data directory accessible to deployments
            startup_applications: List of application configurations to deploy on startup
            log_file: Optional log file path for output
            debug: Enable debug logging

        Raises:
            Exception: If initialization of any component fails

        Note:
            The admin_users parameter is passed during initialize() call,
            not during construction.
        """
        # Set up logging
        self.logger = create_logger(
            name="AppsManager",
            level=logging.DEBUG if debug else logging.INFO,
            log_file=log_file,
        )

        # Initialize components
        self.ray_cluster = ray_cluster

        if self.ray_cluster.mode == "slurm":
            # SLURM workers always mount to /tmp/bioengine and /data
            apps_cache_dir = Path("/tmp/bioengine/apps")
            apps_data_dir = Path("/data")
        elif self.ray_cluster.mode == "single-machine":
            # Resolve local paths to ensure they are absolute
            apps_cache_dir = Path(apps_cache_dir)
            apps_data_dir = Path(apps_data_dir)
        elif self.ray_cluster.mode == "external-cluster":
            # For external clusters, use the provided paths directly
            apps_cache_dir = Path(apps_cache_dir)
            apps_data_dir = Path(apps_data_dir)
        else:
            raise ValueError(
                f"Unsupported Ray cluster mode: {self.ray_cluster.mode}. "
                "Supported modes are 'slurm', 'single-machine', and 'external-cluster'."
            )

        self.app_builder = AppBuilder(
            token=token,
            apps_cache_dir=apps_cache_dir,
            apps_data_dir=apps_data_dir,
            log_file=log_file,
            debug=debug,
        )

        self.haikunator = Haikunator()

        # Initialize state variables
        self.server = None
        self.artifact_manager = None
        self.collection_id = None
        self.admin_users = None
        self.startup_applications = startup_applications
        self._deployment_lock = asyncio.Lock()
        self._deployed_applications = {}
        self.debug = debug

    def _check_initialized(self) -> None:
        """
        Check if the server and artifact manager are initialized.
        """
        if not self.server:
            raise RuntimeError(
                "Hypha server connection not available. Call initialize() first."
            )
        if not self.artifact_manager:
            raise RuntimeError(
                "Artifact manager not initialized. Call initialize() first."
            )

    async def _ensure_bioengine_apps_collection(self) -> None:
        """
        Ensure the 'bioengine-apps' collection exists in the Hypha artifact manager.
        Creates the collection if it does not exist, allowing for organized storage of BioEngine applications.

        Raises:
            RuntimeError: If the collection cannot be created or accessed
        """
        try:
            await self.artifact_manager.read(self.collection_id)
        except Exception as collection_error:
            expected_error = (
                f"KeyError: \"Artifact with ID '{self.collection_id}' does not exist.\""
            )
            if str(collection_error).strip().endswith(expected_error):
                self.logger.info(
                    f"Collection '{self.collection_id}' does not exist. Creating it."
                )

                collection_manifest = {
                    "name": "BioEngine Apps",
                    "description": "A collection of Ray deployments for the BioEngine.",
                }
                collection = await self.artifact_manager.create(
                    alias=self.collection_id,
                    type="collection",
                    manifest=collection_manifest,
                    config={"permissions": {"*": "r", "@": "r+"}},
                )
                self.logger.info(
                    f"Bioengine Apps collection created with ID: {collection.id}."
                )

    def _get_full_artifact_id(self, artifact_id: str) -> str:
        """
        Convert artifact ID to a full artifact ID.

        Prepends workspace prefix if the artifact ID doesn't already contain one.

        Args:
            artifact_id: The artifact ID to convert

        Returns:
            str: The converted full artifact ID in format 'workspace/artifact_id'
        """
        if "/" in artifact_id:
            return artifact_id
        else:
            # If artifact_id does not contain a slash, prepend the workspace
            return f"{self.server.config.workspace}/{artifact_id}"

    async def _generate_application_id(self) -> str:
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
        Execute deployment of a Ray Serve application with comprehensive lifecycle management.

        This method runs the actual deployment process, monitors the deployment status,
        executes initialization hooks, and maintains the deployment until cancellation.
        It ensures proper cleanup of both Ray Serve resources and internal state tracking
        regardless of how the deployment ends (success, failure, or cancellation).

        The method runs indefinitely to keep the deployment active until explicitly
        cancelled, at which point it performs automatic cleanup including:
        - Removal from Ray Serve
        - Deletion from internal deployment tracking
        - Service registration updates

        Args:
            artifact_id: Full artifact ID being deployed
            deployment_name: Ray Serve deployment name
            app: Configured Ray Serve application

        Raises:
            RuntimeError: If deployment validation fails
            Exception: If deployment startup or initialization fails

        Note:
            This method is designed to run as a long-lived asyncio task and will
            only exit when cancelled or on error. Cleanup is guaranteed via finally block.
        """
        try:
            # Reset deployment status (if already deployed)
            self._deployed_applications[application_id]["is_deployed"].clear()

            artifact_id = self._deployed_applications[application_id]["artifact_id"]
            version = self._deployed_applications[application_id]["version"]

            # Create the deployment from the artifact
            app = await self.app_builder.build(
                application_id=application_id,
                artifact_id=artifact_id,
                version=version,
                deployment_kwargs=self._deployed_applications[application_id][
                    "deployment_kwargs"
                ],
                enable_gpu=self._deployed_applications[application_id]["enable_gpu"],
            )

            # Check if the required resources are available
            await self._check_resources(
                application_id=application_id,
                required_resources=app.metadata["resources"],
            )

            # Run the deployment in Ray Serve with unique route prefix
            deployment_coroutine = asyncio.to_thread(
                serve.run,
                target=app,
                name=application_id,
                route_prefix=f"/{application_id}",
                blocking=False,
            )

            # Await the coroutine to start the deployment
            app_handle = await deployment_coroutine

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
                f"artifact '{artifact_id}', version '{version or 'latest'}'."
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
            # TODO: Get the error message from the Ray Serve application
            try:
                serve_status = await asyncio.to_thread(serve.status)
                application = serve_status.applications.get(application_id)
                if application:
                    error_message = application.status.error_message
                    self.logger.error(
                        f"Ray Serve application '{application_id}' reported error: {error_message}"
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

    async def initialize(self, server: RemoteService, admin_users: List[str]) -> None:
        """
        Initialize the deployment manager with a Hypha server connection.

        Establishes connection to the Hypha server and artifact manager service
        for deployment operations.

        Args:
            server: Hypha server connection instance
            admin_users: List of user IDs or emails with admin permissions

        Raises:
            Exception: If server connection or artifact manager initialization fails
        """
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

        # Set the collection ID for BioEngine applications
        workspace = self.server.config.workspace
        self.collection_id = f"{workspace}/bioengine-apps"

        # Initialize the AppBuilder with the server and artifact manager
        self.app_builder.initialize(
            server=self.server,
            artifact_manager=self.artifact_manager,
            serve_http_url=self.ray_cluster.serve_http_url,
        )

        # Deploy any startup applications if provided
        if self.startup_applications:
            self.logger.info(
                f"Deploying {len(self.startup_applications)} startup application(s)..."
            )
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
        Get comprehensive status of all deployed artifacts with task state validation.

        Returns detailed status information for all currently tracked deployments,
        including deployment metadata, resource usage, and service availability.
        Validates deployment state against both internal tracking and Ray Serve status
        to ensure consistency.

        The method cross-references _deployed_artifacts with Ray Serve status to
        identify any inconsistencies and provides warnings for deployments that
        may be in an unexpected state.

        Returns:
            Dict containing:
                - service_id: Hypha service ID for deployments (if registered)
                - Per artifact: deployment name, available methods, timing info,
                  status, replica states, resource allocation, and task state

        Raises:
            RuntimeError: If Ray cluster is not running

        Note:
            Only deployments that exist in both internal tracking and Ray Serve
            status are included in the output to ensure accuracy.
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
                "deployment_kwargs": application_info["deployment_kwargs"],
                "gpu_enabled": application_info["enable_gpu"],
                "application_resources": application_info["application_resources"],
                "authorized_users": application_info["authorized_users"],
                "available_methods": application_info["available_methods"],
                "service_ids": service_ids,
                "last_updated_by": application_info["last_updated_by"],
            }

        return output

    async def monitor_applications(self) -> None:
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
                        f"version '{application_info['version'] or 'latest'}' has failed multiple times. It will be "
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
        self, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[str]]:
        """
        List all BioEngine application artifacts in the Hypha artifact manager.

        Args:
            context: User context information automatically injected by Hypha.

        Returns:
            Dict[str, List[str]]: Mapping of artifact IDs to their file names in the BioEngine Apps collection
        """
        self._check_initialized()

        check_permissions(
            context=context,
            authorized_users=self.admin_users,
            resource_name=f"listing applications",
        )

        # Check if the 'bioengine-apps' collection exists
        await self._ensure_bioengine_apps_collection()

        bioengine_apps_artifacts = await self.artifact_manager.list(self.collection_id)

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
        self, files: List[dict], artifact_id: str = None, context: Optional[dict] = None
    ) -> str:
        """
        Create a BioEngine application artifact in the Hypha artifact manager.

        Args:
            files: List of file dictionaries with 'name', 'content', and 'type' keys
                   type can be 'text' or 'base64'
            artifact_id: Optional artifact ID. If provided, will edit existing artifact.
                        If not provided, will create new artifact using alias from manifest.
            context: User context information automatically injected by Hypha.

        Returns:
            str: The artifact ID of the created/updated artifact
        """
        self._check_initialized()

        check_permissions(
            context=context,
            authorized_users=self.admin_users,
            resource_name=f"creating or modifying an application",
        )

        # Find the manifest file to extract metadata
        manifest_file = None
        for file in files:
            if file["name"].lower() == "manifest.yaml":
                manifest_file = file
                break

        if not manifest_file:
            raise ValueError(
                "No manifest file found in files list. Expected 'manifest.yaml'"
            )

        # Load the manifest content
        if manifest_file["type"] == "text":
            manifest_content = manifest_file["content"]
        else:
            # Remove `data:...` prefix from the base64 content
            if manifest_file["content"].startswith("data:"):
                manifest_content = manifest_file["content"].split(",")[1]
            # Decode base64 content
            manifest_content = base64.b64decode(manifest_content).decode("utf-8")
            # remove data:... prefix from the base64 content
            if manifest_content.startswith("data:"):
                manifest_content = manifest_content.split(",")[1]

        deployment_manifest = yaml.safe_load(manifest_content)

        # Check if type is set to 'application'
        artifact_type = deployment_manifest.get("type")
        if artifact_type != "application":
            raise ValueError(f"Type must be 'application', got '{artifact_type}'")

        workspace = self.server.config.workspace
        if artifact_id is not None:
            # If artifact_id is provided, we expect an existing artifact and will edit it
            artifact_id = self._get_full_artifact_id(artifact_id)

            # Ensure the artifact is in the correct workspace
            if not artifact_id.startswith(f"{workspace}/"):
                raise ValueError(
                    f"Artifact ID '{artifact_id}' does not belong to the current workspace '{workspace}'."
                )

            try:
                # Try to edit existing artifact
                self.logger.debug(f"Editing existing artifact '{artifact_id}'...")
                artifact = await self.artifact_manager.edit(
                    artifact_id=artifact_id,
                    manifest=deployment_manifest,
                    type="application",
                    stage=True,
                )
            except Exception as e:
                # If edit fails, throw an error since we expected an existing artifact
                raise RuntimeError(f"Failed to edit artifact '{artifact_id}': {e}")
        else:
            # If artifact_id is not provided, create new artifact using alias from manifest
            deployment_manifest["created_by"] = context["user"]["id"]

            # Validate the artifact ID
            if "id" not in deployment_manifest:
                raise ValueError(
                    "No artifact_id provided and no 'id' field found in manifest"
                )

            alias = deployment_manifest["id"]

            # Validate alias format (can contain -, but not / or other special characters)
            # Must be a valid Python identifier after replacing - with _
            invalid = any(
                [
                    not alias.islower(),
                    "_" in alias,
                    "/" in alias,
                    not alias.replace("-", "_").isidentifier(),
                ]
            )
            if invalid:
                raise ValueError(
                    f"Invalid artifact alias: '{alias}'. Please use lowercase letters, numbers, and hyphens only."
                )

            # Ensure the bioengine-apps collection exists
            await self._ensure_bioengine_apps_collection()

            try:
                # Create new artifact using alias
                self.logger.debug(f"Creating new artifact with alias '{alias}'...")
                artifact = await self.artifact_manager.create(
                    alias=alias,
                    parent_id=self.collection_id,
                    manifest=deployment_manifest,
                    type=deployment_manifest.get("type", "application"),
                    stage=True,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to create artifact with alias '{alias}' in workspace '{workspace}': {e}"
                )

        # Upload all files
        for file in files:
            file_name = file["name"]
            file_content = file["content"]
            file_type = file["type"]

            self.logger.debug(f"Uploading file '{file_name}' to artifact...")

            # Get upload URL
            upload_url = await self.artifact_manager.put_file(
                artifact.id, file_path=file_name
            )

            # Prepare content for upload
            if file_type == "text":
                upload_data = file_content
            elif file_type == "base64":
                # Decode base64 content for binary files
                upload_data = base64.b64decode(file_content)
            else:
                raise ValueError(
                    f"Unsupported file type '{file_type}'. Expected 'text' or 'base64'"
                )

            # Upload the file with timeout (30s for all operations)
            upload_timeout = httpx.Timeout(30.0)
            async with httpx.AsyncClient(timeout=upload_timeout) as client:
                if file_type == "text":
                    response = await client.put(upload_url, data=upload_data)
                else:
                    response = await client.put(upload_url, content=upload_data)
                response.raise_for_status()
                self.logger.debug(f"Successfully uploaded '{file_name}' to artifact.")

        # Commit the artifact
        await self.artifact_manager.commit(
            artifact_id=artifact.id,
        )
        self.logger.debug(f"Committed artifact with ID: {artifact.id}.")

        # Verify the artifact is in the collection
        available_artifacts = await self.list_applications(context=context)
        if artifact.id not in available_artifacts:
            raise ValueError(
                f"Artifact '{artifact.id}' could not be created or is not in the collection."
            )

        self.logger.info(
            f"Successfully created/updated application artifact '{artifact.id}'."
        )

        return artifact.id

    @schema_method
    async def delete_application(
        self, artifact_id: str, context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Delete a BioEngine application artifact from the Hypha artifact manager.

        Args:
            artifact_id: ID of the artifact to delete
            context: User context information automatically injected by Hypha.

        Returns:
            str: The ID of the deleted artifact

        Raises:
            ValueError: If the artifact does not exist or cannot be deleted
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
        artifact_id: str,
        version: Optional[str] = None,
        application_id: Optional[str] = None,
        deployment_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
        enable_gpu: bool = True,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Deploy a single artifact to Ray Serve with comprehensive state management.

        Downloads the artifact from Hypha, loads the deployment code, configures
        the Ray Serve deployment with appropriate resources, creates a managed
        deployment task, and registers it as a callable service. The deployment
        is tracked in _deployed_artifacts for proper lifecycle management.

        The method prevents duplicate deployments by checking existing state and
        ensures proper task creation and tracking. Returns immediately after
        starting the deployment task, which runs independently.

        Args:
            artifact_id: ID of the artifact to deploy
            mode: Optional deployment mode for multi-mode artifacts
            version: Optional version of the artifact to deploy
            application_id: Optional unique ID for the application deployment.
                        If not provided, a new ID will be generated.
            deployment_kwargs: Optional dictionary of keyword arguments for deployment classes.
            enable_gpu: Whether to enable GPU support for the deployment
            context: User context information automatically injected by Hypha.

        Raises:
            RuntimeError: If server, artifact manager, or Ray is not initialized
            PermissionError: If user lacks permission to deploy artifacts
            ValueError: If artifact mode is invalid or deployment configuration is malformed
            Exception: If artifact deployment initialization fails

        Note:
            The actual deployment runs asynchronously. Use get_status() to monitor
            deployment progress and success.
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

            # Validate deployment_kwargs
            if deployment_kwargs is not None:
                if not isinstance(deployment_kwargs, dict):
                    raise ValueError(
                        "deployment_kwargs must be a dictionary of keyword arguments."
                    )
                # Ensure all values are dictionaries
                for key, value in deployment_kwargs.items():
                    if not isinstance(value, dict):
                        raise ValueError(
                            f"Value for '{key}' in deployment_kwargs must be a dictionary."
                        )

                kwargs_str = ", ".join(
                    f"{deployment_class}("
                    + ", ".join(
                        [f"{key}={value!r}" for key, value in init_kwargs.items()]
                    )
                    + ")"
                    for deployment_class, init_kwargs in deployment_kwargs.items()
                )
            else:
                deployment_kwargs = {}
                kwargs_str = "None"

            if application_id not in self._deployed_applications:
                # Create a new application if application_id is not provided
                self.logger.info(
                    f"User '{user_id}' is deploying new application '{application_id}' from artifact '{artifact_id}', version '{version or 'latest'}' with kwargs: {kwargs_str}"
                )
            else:
                # Update existing application
                application_info = self._deployed_applications[application_id]
                if application_info["is_deployed"].is_set():
                    # If already deployed, cancel the existing deployment task to update deployment in a new task
                    self.logger.info(
                        f"User '{user_id}' is updating existing application from artifact '{artifact_id}', version '{version or 'latest'}' with kwargs: {kwargs_str}"
                    )
                    application_info["remove_on_exit"] = False
                    application_info["deployment_task"].cancel()
                    timeout = 10
                    try:
                        await asyncio.wait_for(
                            application_info["is_deployed"], timeout=timeout
                        )
                    except asyncio.TimeoutError:
                        self.logger.warning(
                            f"Cancellation of existing deployment task for application '{application_id}' "
                            f"did not finish in time ({timeout} seconds). Proceeding with new deployment."
                        )
                else:
                    raise RuntimeError(
                        f"Application '{application_id}' can not be updated as it is in an unfinished deployment process. "
                        f" Wait for the current deployment to finish or call `undeploy_application()` first."
                    )

            self._deployed_applications[application_id] = {
                "display_name": "",
                "description": "",
                "artifact_id": artifact_id,
                "version": version,
                "deployment_kwargs": deployment_kwargs,
                "enable_gpu": enable_gpu,
                "application_resources": {},
                "authorized_users": [],
                "available_methods": [],
                "last_updated_by": user_id,
                "deployment_task": None,  # Track the deployment task
                "is_deployed": asyncio.Event(),  # Track if the deployment has been started
                "remove_on_exit": not self.debug,  # Remove on exit unless in debug mode
                "consecutive_failures": 0,  # Track consecutive failures for monitoring
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
    async def deploy_applications(self, app_configs: List[dict], context: dict) -> None:
        """
        Deploy multiple BioEngine applications from a list of configurations.

        Iterates through the provided app_configs, validates each configuration,
        and deploys each application using the deploy_application method. Returns
        a list of application IDs for successfully deployed applications.

        Args:
            app_configs: List of dictionaries containing application configurations.
                         Each dictionary must contain 'artifact_id' and may include
                         'version', 'application_id', 'deployment_kwargs' and 'enable_gpu'.
            context: User context information automatically injected by Hypha.

        Raises:
            RuntimeError: If server or artifact manager is not initialized
            Exception: If deployment of any startup artifact fails
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
                deployment_kwargs=app_config.get("deployment_kwargs"),
                enable_gpu=app_config.get("enable_gpu", True),
                context=context,
            )
            application_ids.append(application_id)

        return application_ids

    @schema_method
    async def undeploy_application(
        self,
        application_id: str,
        context: Dict[str, Any],
    ) -> None:
        """
        Remove a deployment from Ray Serve with proper task management.

        Gracefully undeploys an artifact by canceling the active deployment task,
        which triggers automatic cleanup including removal from Ray Serve and
        deletion from internal state tracking. Validates that the artifact is
        currently deployed before attempting undeployment.

        The method cancels the deployment task, which causes the _deploy_application
        method to exit and perform cleanup in its finally block. This ensures
        consistent cleanup regardless of deployment state.

        Args:
            artifact_id: ID of the artifact to undeploy
            context: User context information automatically injected by Hypha.

        Raises:
            RuntimeError: If server, artifact manager, or Ray is not initialized
            PermissionError: If user lacks permission to undeploy artifacts

        Note:
            The method returns immediately after canceling the deployment task.
            Actual cleanup happens asynchronously in the deployment task's finally block.
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
    async def cleanup(self, context: Dict[str, Any]) -> None:
        """
        Clean up all Ray Serve deployments and associated resources with robust task management.

        Gracefully undeploys all artifacts by canceling deployment tasks and waiting
        for cleanup completion. Handles timeout scenarios and provides comprehensive
        logging of cleanup status. Ensures proper removal from both Ray Serve and
        internal state tracking.

        The method creates a snapshot of current deployments to avoid modification
        during iteration, cancels all deployment tasks, and waits for cleanup
        completion with timeout handling for robust operation.

        Args:
            context: User context information automatically injected by Hypha.

        Raises:
            RuntimeError: If Ray cluster is not running or connections unavailable
            PermissionError: If user lacks admin permissions for cleanup

        Note:
            Failed cleanup attempts are logged but don't prevent the method from
            completing. Service registration is updated after cleanup regardless
            of individual task failures.
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
