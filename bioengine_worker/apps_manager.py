import asyncio
import base64
import logging
import os
from functools import partial
from pathlib import Path
from time import time
from typing import Any, Dict, List, Optional, Union

import httpx
import numpy as np
import ray
import yaml
from ray import serve

from bioengine_worker import __version__
from bioengine_worker.ray_cluster import RayCluster
from bioengine_worker.utils import create_logger


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
    The class maintains deployment state in `_deployed_artifacts` dictionary where each
    entry contains deployment metadata, resource allocation, and an active asyncio task
    reference. Proper cleanup ensures no resource leaks or orphaned deployments.

    Attributes:
        service_id (str): Service ID for Hypha registration
        admin_users (List[str]): List of user emails with admin permissions
        apps_cache_dir (Path): Cache directory for deployment artifacts
        apps_data_dir (Path): Data directory for deployment access
        ray_cluster (RayCluster): Ray cluster manager instance
        server: Hypha server connection
        artifact_manager: Hypha artifact manager service
        service_info: Registered Hypha service information
        startup_deployments (List[str]): Deployments to start automatically
        logger: Logger instance for deployment operations
        _deployed_artifacts (Dict): Internal tracking of active deployments with task references
    """

    def __init__(
        self,
        ray_cluster: RayCluster,
        admin_users: Optional[List[str]] = None,
        apps_cache_dir: str = "/tmp/bioengine/apps",
        apps_data_dir: str = "/data",
        startup_deployments: Optional[List[str]] = None,
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
            admin_users: List of user IDs or emails with admin permissions
            apps_cache_dir: Caching directory used in Ray Serve deployments
            apps_data_dir: Data directory accessible to deployments
            startup_deployments: List of artifact IDs to start on initialization
            log_file: Optional log file path for output
            debug: Enable debug logging

        Raises:
            Exception: If initialization of any component fails
        """
        # Set up logging
        self.logger = create_logger(
            name="AppsManager",
            level=logging.DEBUG if debug else logging.INFO,
            log_file=log_file,
        )

        # Store parameters
        self.service_id = "bioengine-apps"
        self.admin_users = admin_users or []
        self.apps_cache_dir = (
            Path(apps_cache_dir).resolve()
            if ray_cluster.mode == "single-machine"
            else Path("/tmp/bioengine/apps")
        )
        self.apps_data_dir = (
            Path(apps_data_dir).resolve()
            if ray_cluster.mode == "single-machine"
            else Path("/data")
        )
        self.ray_cluster = ray_cluster

        # Initialize state variables
        self.server = None
        self.artifact_manager = None
        self.service_info = None
        self.startup_deployments = startup_deployments or []
        self._deployment_semaphore = asyncio.Semaphore(value=1)
        self._deployed_artifacts = {}

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

    def _get_admin_context(self) -> Dict[str, Any]:
        """
        Get context for admin users.

        Returns:
            Dict: Context dictionary containing admin user information
        """
        return {
            "user": {
                "id": self.admin_users[0] if self.admin_users else "",
                "email": self.admin_users[1] if self.admin_users else "",
            }
        }

    def _get_full_artifact_id(self, artifact_id: str) -> str:
        """
        Convert artifact ID to a full artifact ID.

        Prepends workspace prefix if the artifact ID doesn't already contain one.

        Args:
            artifact_id: The artifact ID to convert

        Returns:
            str: The converted full artifact ID in format 'workspace/artifact_id'
        """
        if "/" not in artifact_id:
            return f"{self.server.config.workspace}/{artifact_id}"
        return artifact_id

    def _check_permissions(
        self,
        context: Dict[str, Any],
        authorized_users: Union[List[str], str],
        resource_name: str,
    ) -> bool:
        """
        Check if the user in the context is authorized to access the deployment.

        Validates user permissions against the authorized users list for specific
        deployment operations.

        Args:
            context: Request context containing user information
            authorized_users: List of authorized user IDs/emails or single user string
            resource_name: Name of the resource being accessed for logging

        Returns:
            bool: True if user is authorized

        Raises:
            PermissionError: If user is not authorized to access the resource
        """
        if context is None or "user" not in context:
            raise PermissionError("Context is missing user information")
        user = context["user"]
        if isinstance(authorized_users, str):
            authorized_users = [authorized_users]
        if (
            "*" not in authorized_users
            and user["id"] not in authorized_users
            and user["email"] not in authorized_users
        ):
            raise PermissionError(
                f"User {user['id']} is not authorized to access {resource_name}"
            )

    def _create_deployment_name(self, artifact_id: str) -> str:
        """
        Create a valid deployment name from an artifact ID.

        Converts the artifact ID to a valid Python identifier by replacing
        special characters with underscores and ensuring it meets naming requirements.

        Args:
            artifact_id: The artifact ID to convert

        Returns:
            str: A valid deployment name suitable for Ray Serve

        Raises:
            ValueError: If the artifact ID cannot be converted to a valid identifier
        """
        # TODO: Convert artifact ID to a URL- and Ray-safe unique name (idea: replace / with |)
        deployment_name = artifact_id.lower()
        for char in ["|", "/", "-", "."]:
            deployment_name = deployment_name.replace(char, "_")
        if not deployment_name.isidentifier():
            raise ValueError(
                f"Artifact ID '{artifact_id}' can not be automatically converted to a "
                f"valid deployment name ('{deployment_name}' is not a valid identifier)."
            )
        return deployment_name

    async def _load_deployment_code(
        self,
        class_config: dict,
        artifact_id: str,
        version: str,
        timeout: int,
        _local: bool = False,  # Used for development
    ) -> Any:
        """
        Load and execute deployment code from an artifact directly in memory.

        Downloads and executes Python code from an artifact to create deployable classes.
        Supports both remote artifact loading and local file loading for development.

        Args:
            class_config: Configuration for the class to load including class name and file path
            artifact_id: ID of the artifact containing the deployment code
            version: Optional version of the artifact to load
            timeout: Timeout in seconds for network requests
            _local: Whether to load from local filesystem instead of artifact

        Returns:
            Any: The loaded class ready for Ray Serve deployment

        Raises:
            FileNotFoundError: If local deployment file is not found
            ValueError: If class name is not found in the code
            RuntimeError: If class loading fails
            Exception: If code execution or download fails
        """
        try:
            if _local:
                # Load the file content from local path
                deployment = artifact_id.split("/")[1].replace("-", "_")
                local_deployments_dir = (
                    Path(__file__).parent.parent.resolve() / "deployments"
                )
                local_path = (
                    local_deployments_dir / deployment / class_config["python_file"]
                )
                if not local_path.exists():
                    raise FileNotFoundError(
                        f"Local deployment file not found: {local_path}"
                    )
                with open(local_path, "r") as f:
                    code_content = f.read()
            else:
                # Get download URL for the file
                download_url = await self.artifact_manager.get_file(
                    artifact_id=artifact_id,
                    version=version,
                    file_path=class_config["python_file"],
                )

                # Download the file content with timeout
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.get(download_url)
                    response.raise_for_status()
                    code_content = response.text

            # Create a restricted globals dictionary for sandboxed execution
            safe_globals = {}
            # Execute the code in a sandboxed environment
            exec(code_content, safe_globals)
            if class_config["class_name"] not in safe_globals:
                raise ValueError(
                    f"{class_config['class_name']} not found in {artifact_id}"
                )
            deployment_class = safe_globals[class_config["class_name"]]
            if not deployment_class:
                raise RuntimeError(
                    f"Error loading {class_config['class_name']} from {artifact_id}"
                )

            if "multiplexed" in class_config:
                # Add @serve.multiplexed decorator to specified class method
                method_name = class_config["multiplexed"]["method_name"]
                max_num_models_per_replica = class_config["multiplexed"][
                    "max_num_models_per_replica"
                ]

                orig_method = getattr(deployment_class, method_name)
                decorated_method = serve.multiplexed(
                    orig_method, max_num_models_per_replica=max_num_models_per_replica
                )
                setattr(deployment_class, method_name, decorated_method)

            self.logger.info(
                f"Loaded class '{class_config['class_name']}' from {artifact_id}"
            )
            return deployment_class

        except Exception as e:
            self.logger.error(f"Error loading deployment code for {artifact_id}: {e}")
            raise e

    async def _create_deployment(
        self, artifact_id: str, mode: str, version: str, deployment_name: str
    ) -> serve.Application:
        """
        Create a Ray Serve deployment from an artifact with proper state tracking.

        Downloads artifact metadata, configures Ray Serve deployment parameters,
        validates resource requirements, and initializes deployment state tracking.
        The deployment info is stored in _deployed_artifacts for lifecycle management.

        Args:
            artifact_id: Full artifact ID to deploy
            mode: Deployment mode for multi-mode artifacts
            version: Artifact version to deploy

        Returns:
            serve.Application: Configured Ray Serve application ready for deployment

        Raises:
            ValueError: If mode is invalid, resources insufficient, or artifact malformed
            Exception: If artifact download or deployment creation fails
        """
        # Read the manifest to get deployment configuration
        artifact = await self.artifact_manager.read(artifact_id, version=version)
        manifest = artifact["manifest"]

        # Get the deployment configuration
        deployment_config = manifest["deployment_config"]
        class_config = manifest["deployment_class"]
        deployment_config["name"] = class_config["class_name"]

        # Check if different modes are supported
        modes = deployment_config.pop("modes", None)
        if modes:
            mode_key = mode or list(modes.keys())[0]
            mode_key = mode_key.lower()
            if mode_key not in modes:
                raise ValueError(
                    f"Mode '{mode_key}' not found in artifact '{artifact_id}'"
                )

            # Update deployment config with the selected mode
            for key, value in modes[mode_key].items():
                deployment_config[key] = value

            self.logger.info(f"Using mode '{mode_key}' for deployment '{artifact_id}'")

        # Add default ray_actor_options if not present
        ray_actor_options = deployment_config.setdefault("ray_actor_options", {})
        ray_actor_options.setdefault("num_cpus", 1)
        ray_actor_options.setdefault("num_gpus", 0)
        ray_actor_options.setdefault("memory", 0)

        # Check if the required resources are available
        insufficient_resources = True
        for node_resource in self.ray_cluster.status["nodes"].values():
            if (
                node_resource["available_cpu"] >= ray_actor_options["num_cpus"]
                and node_resource["available_gpu"] >= ray_actor_options["num_gpus"]
                and node_resource["available_memory"] >= ray_actor_options["memory"]
            ):
                insufficient_resources = False

        if self.ray_cluster.mode == "slurm" and insufficient_resources:
            # Check if additional SLURM workers can be created that meet the resource requirements
            # TODO: Remove resource check when SLURM workers can adjust resources dynamically
            num_worker_jobs = await self.ray_cluster.slurm_workers.get_num_worker_jobs()
            default_num_cpus = self.ray_cluster.slurm_workers.default_num_cpus
            default_num_gpus = self.ray_cluster.slurm_workers.default_num_gpus
            default_memory = (
                self.ray_cluster.slurm_workers.default_mem_per_cpu * default_num_cpus
            )
            if (
                num_worker_jobs < self.ray_cluster.slurm_workers.max_workers
                and default_num_cpus >= ray_actor_options["num_cpus"]
                and default_num_gpus >= ray_actor_options["num_gpus"]
                and default_memory >= ray_actor_options["memory"]
            ):
                insufficient_resources = False

        if insufficient_resources:
            if self.ray_cluster.mode != "external-cluster":
                raise ValueError(
                    f"Insufficient resources for deployment '{deployment_name}'. "
                    f"Requested: {ray_actor_options}"
                )
            else:
                self.logger.warning(
                    f"Currently insufficient resources for deployment '{deployment_name}'. "
                    "Assuming Ray autoscaling is available. "
                    f"Requested resources: {ray_actor_options}"
                )

        # Add cache path to deployment config environment
        runtime_env = ray_actor_options.setdefault("runtime_env", {})
        env_vars = runtime_env.setdefault("env_vars", {})
        deployment_workdir = str(self.apps_cache_dir / deployment_name)
        env_vars["BIOENGINE_WORKDIR"] = deployment_workdir
        env_vars["BIOENGINE_DATA_DIR"] = str(self.apps_data_dir)

        # Ensure the deployment only uses the specified workdir
        env_vars["TMPDIR"] = deployment_workdir
        env_vars["HOME"] = deployment_workdir

        # Pass user workspace and token to the deployment
        env_vars["HYPHA_WORKSPACE"] = self.server.config.workspace
        env_vars["HYPHA_CLIENT_ID"] = self.server.config.client_id
        env_vars["HYPHA_SERVICE_ID"] = self.service_id
        env_vars["HYPHA_TOKEN"] = self.server.config.reconnection_token

        # Load the deployment code
        deployment_class = await self._load_deployment_code(
            class_config=class_config,
            artifact_id=artifact_id,
            version=version,
            timeout=60,
        )

        # Update the deployment tracking information
        deployment_info = self._deployed_artifacts[artifact_id]
        artifact_emoji = manifest["id_emoji"] + " " if manifest.get("id_emoji") else ""
        artifact_name = manifest.get("name", manifest["id"])
        deployment_info["display_name"] = artifact_emoji + artifact_name
        deployment_info["description"] = manifest.get("description", "")
        deployment_info["deployment_name"] = deployment_name
        deployment_info["class_config"] = class_config
        deployment_info["resources"] = {
            "num_cpus": deployment_config["ray_actor_options"]["num_cpus"],
            "num_gpus": deployment_config["ray_actor_options"]["num_gpus"],
            "memory": deployment_config["ray_actor_options"].get("memory"),
        }
        deployment_info["async_init"] = hasattr(deployment_class, "async_init")

        # Create the Ray Serve deployment
        deployment = serve.deployment(**deployment_config)(deployment_class)

        # Bind the arguments to the deployment and return an Application
        kwargs = class_config.get("kwargs", {})
        app = deployment.bind(**kwargs)

        return app

    async def _update_services(self) -> None:
        """
        Update Hypha services based on currently deployed applications.

        Registers all currently deployed artifacts as callable Hypha services,
        enabling remote access to the deployed applications through the Hypha platform.

        Raises:
            RuntimeError: If Hypha server connection is not available
            Exception: If service registration fails
        """
        try:
            self._check_initialized()

            if not self._deployed_artifacts:
                self.logger.info("No deployments to register as services")
                self.service_info = None
                return

            async def create_deployment_function(
                deployment_name,
                method_name,
                authorized_users,
                *args,
                context=None,
                **kwargs,
            ):
                try:
                    self._check_permissions(
                        context=context,
                        authorized_users=authorized_users,
                        resource_name=f"deployment '{deployment_name}' method '{method_name}'",
                    )
                    user_id = context["user"]["id"]

                    self.logger.info(
                        f"User '{user_id}' is calling deployment '{deployment_name}' with method '{method_name}'"
                    )
                    app_handle = await asyncio.to_thread(
                        serve.get_app_handle, name=deployment_name
                    )

                    # Recursively put args and kwargs into ray object storage
                    args = [
                        (
                            await asyncio.to_thread(ray.put, arg)
                            if isinstance(arg, np.ndarray)
                            else arg
                        )
                        for arg in args
                    ]
                    kwargs = {
                        k: (
                            await asyncio.to_thread(ray.put, v)
                            if isinstance(v, np.ndarray)
                            else v
                        )
                        for k, v in kwargs.items()
                    }

                    result = await getattr(app_handle, method_name).remote(
                        *args, **kwargs
                    )
                    return result
                except Exception as e:
                    self.logger.error(
                        f"Failed to call deployment '{deployment_name}': {e}"
                    )
                    raise e

            # Create service functions for each deployment
            service_functions = {}
            for deployment_info in self._deployed_artifacts.values():
                deployment_name = deployment_info["deployment_name"]
                service_functions[deployment_name] = {}
                class_config = deployment_info["class_config"]
                exposed_methods = class_config.get("exposed_methods", {})
                if exposed_methods:
                    for method_name, method_config in exposed_methods.items():
                        service_functions[deployment_name][method_name] = partial(
                            create_deployment_function,
                            deployment_name=deployment_name,
                            method_name=method_name,
                            authorized_users=method_config.get("authorized_users", "*"),
                        )
                else:
                    service_functions[deployment_name] = partial(
                        create_deployment_function,
                        deployment_name=deployment_name,
                        method_name="__call__",
                        authorized_users="*",
                    )

            # Register all deployment functions as a single service
            service_info = await self.server.register_service(
                {
                    "id": self.service_id,
                    "name": "BioEngine Worker Apps",
                    "type": "bioengine-apps",
                    "description": "Calling deployed Ray Serve applications",
                    "config": {"visibility": "public", "require_context": True},
                    **service_functions,
                },
                {"overwrite": True},
            )
            self.logger.info("Successfully registered deployment service")
            server_url = self.server.config.public_base_url
            workspace, sid = service_info.id.split("/")
            service_url = f"{server_url}/{workspace}/services/{sid}"
            for deployment_name in service_functions.keys():
                self.logger.info(
                    f"Access the deployment service at: {service_url}/{deployment_name}"
                )
            self.service_info = service_info

        except Exception as e:
            self.logger.error(f"Error updating services: {e}")
            raise e

    async def _deploy_artifact(
        self,
        artifact_id: str,
        version: str,
        mode: str,
        user_id: str,
        skip_update: bool,
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
            # Check if the artifact is already deployed
            deployment_name = self._create_deployment_name(artifact_id)
            serve_status = await asyncio.to_thread(serve.status)
            if deployment_name in serve_status.applications.keys():
                self.logger.info(
                    f"User '{user_id}' is updating existing deployment for artifact '{artifact_id}'"
                )
            else:
                self.logger.info(
                    f"User '{user_id}' is starting a new deployment for artifact '{artifact_id}'"
                )

            # Create the deployment from the artifact
            app = await self._create_deployment(
                artifact_id=artifact_id,
                mode=mode,
                version=version,
                deployment_name=deployment_name,
            )

            # Run the deployment in Ray Serve
            deployment_coroutine = asyncio.to_thread(
                serve.run,
                target=app,
                name=deployment_name,
                route_prefix=None,
                blocking=False,
            )

            if not skip_update and self.ray_cluster.mode == "slurm":
                # Notify the autoscaling system of the new deployment
                await self.ray_cluster.notify()

            # Await the coroutine to start the deployment
            await deployment_coroutine

            # Validate the deployment's status
            serve_status = await asyncio.to_thread(serve.status)
            if deployment_name in serve_status.applications.keys():
                self.logger.info(
                    f"Successfully completed deployment of artifact '{artifact_id}'"
                )
            else:
                raise RuntimeError(
                    f"Deployment name '{deployment_name}' not found in serve status."
                )

            # Run async init if provided
            if self._deployed_artifacts[artifact_id]["async_init"]:
                app_handle = await asyncio.to_thread(
                    serve.get_app_handle, name=deployment_name
                )
                self.logger.info(
                    f"Calling async_init on deployment '{deployment_name}'"
                )
                await app_handle.async_init.remote()

            # Update services with the new deployment
            if not skip_update:
                await self._update_services()

            # Track the deployment in the internal state
            self._deployed_artifacts[artifact_id]["is_deployed"] = True

            # Keep the deployment task running until cancelled
            while True:
                await asyncio.sleep(1)

        except asyncio.CancelledError:
            self.logger.info(
                f"Deployment task for artifact '{artifact_id}' was cancelled"
            )
        except Exception as e:
            self.logger.error(
                f"Failed to deploy artifact '{artifact_id}' with error: {e}"
            )
        finally:
            if self._deployed_artifacts[artifact_id]["remove_on_exit"]:
                # Cleanup: Remove from Ray Serve and update tracking
                try:
                    await asyncio.to_thread(serve.delete, deployment_name)
                    self.logger.info(f"Deleted deployment '{deployment_name}'")
                except Exception as delete_err:
                    self.logger.error(
                        f"Error deleting deployment {deployment_name}: {delete_err}"
                    )

                # Remove from deployment tracking
                del self._deployed_artifacts[artifact_id]
                self.logger.info(
                    f"Removed artifact '{artifact_id}' from deployment tracking"
                )

                # Update services with removed deployment
                if not skip_update:
                    await self._update_services()

                self.logger.info(f"Undeployment of artifact '{artifact_id}' completed")

    async def initialize(self, server) -> None:
        """
        Initialize the deployment manager with a Hypha server connection.

        Establishes connection to the Hypha server and artifact manager service
        for deployment operations.

        Args:
            server: Hypha server connection instance

        Raises:
            Exception: If server connection or artifact manager initialization fails
        """
        try:
            # Store server connection
            self.server = server

            # Get artifact manager service
            self.artifact_manager = await self.server.get_service(
                "public/artifact-manager"
            )
            self.logger.info("Successfully connected to artifact manager")

        except Exception as e:
            self.logger.error(f"Error initializing Ray Deployment Manager: {e}")
            self.server = None
            self.artifact_manager = None
            raise e

    async def deploy_artifact(
        self,
        artifact_id: str,
        mode: str = None,
        version: str = None,
        context: Optional[Dict[str, Any]] = None,
        _skip_update=False,
    ) -> bool:
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
            context: Optional context information from Hypha request containing user info
            _skip_update: Skip updating Hypha services after deployment (for batch operations)

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
        async with self._deployment_semaphore:
            self._check_initialized()

            # Get the full artifact ID
            artifact_id = self._get_full_artifact_id(artifact_id)

            # Check user permissions
            self._check_permissions(
                context=context,
                authorized_users=self.admin_users,
                resource_name=f"deployment of artifact '{artifact_id}'",
            )
            user_id = context["user"]["id"]

            # Verify Ray is initialized
            await self.ray_cluster.check_connection()

            # Check if the artifact is already in a deployment process
            deployment_info = self._deployed_artifacts.get(artifact_id)
            if deployment_info:
                if deployment_info["is_deployed"]:
                    # If already deployed, cancel the existing deployment task to update deployment in a new task
                    deployment_info["remove_on_exit"] = False
                    deployment_info["deployment_task"].cancel()
                    timeout = 10
                    try:
                        await asyncio.wait_for(deployment_info["deployment_task"], timeout=timeout)
                    except asyncio.TimeoutError:
                        self.logger.warning(
                            f"Cancellation of existing deployment task for artifact '{artifact_id}' "
                            f"did not finish in time ({timeout} seconds). Proceeding with new deployment."
                        )
                else:
                    self.logger.debug(
                        f"Artifact '{artifact_id}' is in an unfinished deployment process. Skipping new deployment."
                    )
                    return

            # Initialize a new deployment tracking entry
            self._deployed_artifacts[artifact_id] = {
                "display_name": "",
                "description": "",
                "deployment_name": "",
                "class_config": {},
                "resources": {},
                "async_init": False,
                "deployment_task": None,
                "is_deployed": False,  # Track if the deployment has been started
                "remove_on_exit": True,  # Default to remove on exit
            }

            # Create and start the deployment task
            deployment_task = asyncio.create_task(
                self._deploy_artifact(
                    artifact_id=artifact_id,
                    version=version,
                    mode=mode,
                    user_id=user_id,
                    skip_update=_skip_update,
                ),
                name=f"Deployment_{artifact_id}",
            )
            self._deployed_artifacts[artifact_id]["deployment_task"] = deployment_task

    async def deploy_artifacts(self, artifact_ids: List[str], context: dict) -> None:
        """
        Deploy all startup deployments defined in the manager.

        Automatically deploys all artifacts specified in the startup_deployments
        list during initialization. Uses admin user context for authentication.

        Raises:
            RuntimeError: If server or artifact manager is not initialized
            Exception: If deployment of any startup artifact fails
        """
        if not artifact_ids:
            return

        for artifact_id in artifact_ids:
            await self.deploy_artifact(
                artifact_id=artifact_id,
                mode=None,  # Use default mode for startup deployments
                version=None,  # Use latest version for startup deployments
                context=context,
                _skip_update=True,
            )

        if self.ray_cluster.mode == "slurm":
            # Notify the autoscaling system of the new deployment(s)
            await self.ray_cluster.notify()

        # Wait for all deployments to complete
        deployment_tasks = [
            self._deployed_artifacts[artifact_id]["deployment_task"]
            for artifact_id in artifact_ids
        ]
        await asyncio.gather(*deployment_tasks)

        # Update services after startup deployments
        await self._update_services()

    async def deploy_collection(
        self,
        deployment_collection_id: str = "bioengine-apps",
        context: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Deploy all artifacts in the deployment collection to Ray Serve with batch processing.

        Iterates through all artifacts in the specified collection and deploys
        each one to Ray Serve using individual deployment tasks. Updates Hypha
        services once after all deployments are initiated to improve efficiency.

        Each artifact is deployed independently with proper state tracking, allowing
        for partial success scenarios where some deployments succeed while others fail.

        Args:
            deployment_collection_id: Artifact collection ID for deployments
            context: Optional context information from Hypha request containing user info

        Returns:
            List of artifact IDs that were successfully initiated for deployment

        Raises:
            ValueError: If artifact manager is not initialized
            Exception: If deployment of any artifact fails to initiate

        Note:
            The method returns after initiating all deployments. Use get_status()
            to monitor individual deployment progress and success.
        """
        self.logger.info(
            f"Deploying all artifacts in collection '{deployment_collection_id}'..."
        )
        # Ensure artifact manager is available
        if not self.artifact_manager:
            raise ValueError("Artifact manager not initialized")

        # Get all artifacts in the collection
        artifacts = await self.artifact_manager.list(parent_id=deployment_collection_id)

        # Deploy each artifact
        await self.deploy_artifacts(artifact_ids=artifacts, context=context)

    async def undeploy_artifact(
        self,
        artifact_id: str,
        context: Dict[str, Any],
    ) -> None:
        """
        Remove a deployment from Ray Serve with proper task management.

        Gracefully undeploys an artifact by canceling the active deployment task,
        which triggers automatic cleanup including removal from Ray Serve and
        deletion from internal state tracking. Validates that the artifact is
        currently deployed before attempting undeployment.

        The method cancels the deployment task, which causes the _deploy_artifact
        method to exit and perform cleanup in its finally block. This ensures
        consistent cleanup regardless of deployment state.

        Args:
            artifact_id: ID of the artifact to undeploy
            context: Context information from Hypha request containing user info
            _skip_update: Skip updating Hypha services after undeployment (for batch operations)

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
        self._check_permissions(
            context=context,
            authorized_users=self.admin_users,
            resource_name=f"undeployment of artifact '{artifact_id}'",
        )
        user_id = context["user"]["id"]

        # Get the full artifact ID
        artifact_id = self._get_full_artifact_id(artifact_id)

        # Check if artifact is currently deployed
        if artifact_id in self._deployed_artifacts:
            self.logger.info(
                f"User '{user_id}' is starting undeployment of artifact '{artifact_id}'"
            )
            self._deployed_artifacts[artifact_id]["deployment_task"].cancel()
        else:
            self.logger.debug(
                f"Artifact '{artifact_id}' is not currently deployed. No action taken."
            )

    async def cleanup_deployments(self, context: Dict[str, Any]) -> None:
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
            context: Context information from Hypha request containing user info

        Raises:
            RuntimeError: If Ray cluster is not running or connections unavailable
            PermissionError: If user lacks admin permissions for cleanup

        Note:
            Failed cleanup attempts are logged but don't prevent the method from
            completing. Service registration is updated after cleanup regardless
            of individual task failures.
        """
        self._check_initialized()

        # Check if any deployments exist
        if not self._deployed_artifacts:
            self.logger.info("No applications are currently deployed.")
            return

        # Check user permissions
        self._check_permissions(
            context=context,
            authorized_users=self.admin_users,
            resource_name="cleanup of all deployments",
        )
        user_id = context["user"]["id"]

        self.logger.info(f"User '{user_id}' is starting cleanup of all deployments...")

        # Cancel all deployment tasks
        for artifact_id in list(self._deployed_artifacts.keys()):
            await self.undeploy_artifact(artifact_id, context)

        # Wait for all undeployment tasks to complete
        failed_attempts = 0
        for artifact_id in list(self._deployed_artifacts.keys()):
            deployment_info = self._deployed_artifacts.get(artifact_id)
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

    async def get_status(self) -> Dict[str, Any]:
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
        if self.service_info:
            output["service_id"] = self.service_info.id
        else:
            output["service_id"] = None

        # Get status of actively running deployments
        serve_status = await asyncio.to_thread(serve.status)

        if not self._deployed_artifacts:
            output["note"] = "Currently no artifacts are deployed."
            return output

        for artifact_id in list(self._deployed_artifacts.keys()):
            deployment_info = self._deployed_artifacts[artifact_id]
            deployment_name = deployment_info["deployment_name"]
            application = serve_status.applications.get(deployment_name)
            if not application:
                if deployment_info["is_deployed"]:
                    # If the deployment is marked as deployed but not found in Ray Serve,
                    # it may have failed or been removed unexpectedly.
                    self.logger.warning(
                        f"Deployment '{deployment_name}' for artifact '{artifact_id}' "
                        "is marked as deployed but not found in Ray Serve status."
                    )
                    # TODO: This can happen if deploy_artifact and undeploy_artifact are called at the same time
                continue
            if len(application.deployments) > 1:
                raise NotImplementedError

            class_config = deployment_info["class_config"]
            class_methods = class_config.get("exposed_methods", {})
            class_name = class_config["class_name"]
            deployment = application.deployments.get(class_name)
            output[artifact_id] = {
                "display_name": deployment_info["display_name"],
                "description": deployment_info["description"],
                "deployment_name": deployment_name,
                "available_methods": list(class_methods.keys()),
                "start_time": application.last_deployed_time_s,
                "status": application.status.value,
                "replica_states": deployment.replica_states if deployment else None,
                "resources": deployment_info["resources"],
            }

        return output

    async def create_artifact(
        self, files: List[dict], artifact_id: str = None, context: Optional[dict] = None
    ) -> str:
        """
        Create a deployment artifact

        Args:
            files: List of file dictionaries with 'name', 'content', and 'type' keys
                   type can be 'text' or 'base64'
            artifact_id: Optional artifact ID. If provided, will edit existing artifact.
                        If not provided, will create new artifact using alias from manifest.

        Returns:
            str: The artifact ID of the created/updated artifact
        """
        self._check_initialized()

        self._check_permissions(
            context=context,
            authorized_users=self.admin_users,
            resource_name=f"creation of artifact '{artifact_id}'",
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

        if artifact_id is not None:
            # If artifact_id is provided, we expect an existing artifact and will edit it
            workspace = self.server.config.workspace
            full_artifact_id = (
                artifact_id if "/" in artifact_id else f"{workspace}/{artifact_id}"
            )

            try:
                # Try to edit existing artifact
                self.logger.info(f"Editing existing artifact '{full_artifact_id}'")
                artifact = await self.artifact_manager.edit(
                    artifact_id=full_artifact_id,
                    manifest=deployment_manifest,
                    type="application",
                    stage=True,
                )
                self.logger.info(
                    f"Successfully edited existing artifact '{full_artifact_id}'"
                )
            except Exception as e:
                # If edit fails, throw an error since we expected an existing artifact
                raise ValueError(
                    f"Failed to edit existing artifact '{full_artifact_id}': {e}"
                )
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
            workspace = self.server.config.workspace
            collection_id = f"{workspace}/bioengine-apps"
            try:
                await self.artifact_manager.read(collection_id)
            except Exception as collection_error:
                expected_error = (
                    f"KeyError: \"Artifact with ID '{collection_id}' does not exist.\""
                )
                if str(collection_error).strip().endswith(expected_error):
                    self.logger.info(
                        f"Collection '{collection_id}' does not exist. Creating it."
                    )

                    collection_manifest = {
                        "name": "BioEngine Apps",
                        "description": "A collection of Ray deployments for the BioEngine.",
                    }
                    collection = await self.artifact_manager.create(
                        alias=collection_id,
                        type="collection",
                        manifest=collection_manifest,
                        config={"permissions": {"*": "r", "@": "r+"}},
                    )
                    self.logger.info(
                        f"Bioengine Apps collection created with ID: {collection.id}"
                    )

            # Create new artifact using alias
            self.logger.info(f"Creating new artifact with alias '{alias}'")
            artifact = await self.artifact_manager.create(
                alias=alias,
                parent_id=collection_id,
                manifest=deployment_manifest,
                type=deployment_manifest.get("type", "application"),
                stage=True,
            )
            self.logger.info(f"Artifact created with ID: {artifact.id}")

        # Upload all files
        for file in files:
            file_name = file["name"]
            file_content = file["content"]
            file_type = file["type"]

            self.logger.info(f"Uploading file '{file_name}' to artifact")

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

            # Upload the file
            async with httpx.AsyncClient(timeout=30) as client:
                if file_type == "text":
                    response = await client.put(upload_url, data=upload_data)
                else:
                    response = await client.put(upload_url, content=upload_data)
                response.raise_for_status()
                self.logger.info(f"Successfully uploaded '{file_name}' to artifact")

        # Commit the artifact
        await self.artifact_manager.commit(
            artifact_id=artifact.id,
        )
        self.logger.info(f"Committed artifact with ID: {artifact.id}")

        return artifact.id


async def create_demo_artifact(apps_manager, artifact_id=None):
    """Helper function to create a demo artifact from demo deployment files

    Args:
        apps_manager: AppsManager instance (must be initialized)
        artifact_id: Optional custom artifact ID

    Returns:
        str: The created artifact ID
    """
    # Read demo deployment files
    demo_deployment_dir = Path(__file__).parent / "deployments" / "demo_deployment"

    # Read manifest.yaml
    with open(demo_deployment_dir / "manifest.yaml", "r") as f:
        manifest_content = f.read()

    # Read main.py
    with open(demo_deployment_dir / "main.py", "r") as f:
        main_py_content = f.read()

    # Prepare files for create_artifact
    files = [
        {"name": "manifest.yaml", "content": manifest_content, "type": "text"},
        {"name": "main.py", "content": main_py_content, "type": "text"},
    ]

    # Create the artifact
    created_artifact_id = await apps_manager.create_artifact(
        files, artifact_id=artifact_id
    )
    return created_artifact_id


if __name__ == "__main__":
    """Test the AppsManager functionality with a real Ray cluster and deployment."""

    from hypha_rpc import connect_to_server, login

    async def test_create_artifact(
        apps_manager=None, server_url="https://hypha.aicell.io"
    ):
        """Test the create_artifact function with demo deployment files

        Args:
            apps_manager: Optional existing deployment manager (must be initialized)
            server_url: Server URL if creating new connection

        Returns:
            str: The artifact ID of the last created artifact (for use in other tests)
        """
        print("\n===== Testing create_artifact function =====\n")

        # Use existing deployment manager or create new one
        if apps_manager is None:
            try:
                # Create deployment manager (no Ray cluster needed for artifact creation)
                apps_manager = AppsManager(debug=True)

                # Connect to Hypha server using token from environment
                token = os.environ.get("HYPHA_TOKEN") or await login(
                    {"server_url": server_url}
                )
                server = await connect_to_server(
                    {"server_url": server_url, "token": token}
                )

                # Initialize deployment manager
                await apps_manager.initialize(server)

            except Exception as e:
                print(f"❌ Failed to initialize deployment manager: {e}")
                raise e

        try:
            # Test creating artifact without specifying artifact_id (should use ID from manifest)
            print("Testing create_artifact without specifying artifact_id...")
            created_artifact_id = await create_demo_artifact(apps_manager)
            print(f"Successfully created artifact: {created_artifact_id}")

            # Test updating the same artifact
            print(f"\nTesting update of existing artifact: {created_artifact_id}")
            updated_artifact_id = await create_demo_artifact(
                apps_manager, artifact_id=created_artifact_id
            )
            print(f"Successfully updated artifact: {updated_artifact_id}")

            # Test creating artifact with custom artifact_id
            print("\nTesting create_artifact with custom artifact_id...")
            custom_artifact_id = "test-demo-deployment"
            custom_created_id = await create_demo_artifact(
                apps_manager, artifact_id=custom_artifact_id
            )
            print(f"Successfully created custom artifact: {custom_created_id}")

            print("\n✅ All create_artifact tests passed!")

            # Return the last created artifact ID for use in other tests
            return custom_created_id

        except Exception as e:
            print(f"❌ create_artifact test failed: {e}")
            raise e

    async def test_apps_manager(
        server_url="https://hypha.aicell.io", keep_running=False
    ):
        try:
            print("\n===== Testing AppsManager in single-machine mode =====\n")

            # Connect to Hypha server using token from environment
            token = os.environ.get("HYPHA_TOKEN") or await login(
                {"server_url": server_url}
            )
            server = await connect_to_server({"server_url": server_url, "token": token})

            # Start Ray cluster in single-machine mode
            bioengine_cache_dir = Path(os.environ["HOME"]) / ".bioengine"
            ray_cluster = RayCluster(
                mode="single-machine",
                head_num_cpus=1,
                head_num_gpus=0,
                ray_temp_dir=bioengine_cache_dir / "ray",
                status_interval_seconds=3,
                debug=True,
            )
            await ray_cluster.start()

            print("\n=== Cluster status ===\n", ray_cluster.status, end="\n\n")

            # Create deployment manager
            apps_manager = AppsManager(
                ray_cluster=ray_cluster,
                admin_users=[server.config.user["email"]],
                apps_cache_dir=bioengine_cache_dir / "apps",
                debug=True,
            )

            # Initialize deployment manager
            await apps_manager.initialize(server)

            # Test create_artifact function
            created_artifact_id = await test_create_artifact(apps_manager)

            # Test deploying the newly created artifact
            print(
                f"\n--- Testing deployment of created artifact: {created_artifact_id} ---"
            )
            await apps_manager.deploy_artifact(created_artifact_id)

            # Test the deployed artifact
            deployment_status = await apps_manager.get_status()
            if created_artifact_id in deployment_status:
                print(f"Successfully deployed created artifact: {created_artifact_id}")

                # Test the service
                deployment_service_id = deployment_status["service_id"]
                deployment_service = await server.get_service(deployment_service_id)
                deployment_name = deployment_status[created_artifact_id][
                    "deployment_name"
                ]

                # Test ping method
                response = await deployment_service[deployment_name]["ping"]()
                print(f"Ping response from created artifact: {response}")

                # Test get_time method
                response = await deployment_service[deployment_name]["get_time"](
                    "Stockholm"
                )
                print(f"Time response from created artifact: {response}")
            else:
                print(f"Failed to deploy created artifact: {created_artifact_id}")

            # Deploy the example deployment
            artifact_id = "example-deployment"
            await apps_manager.deploy_artifact(artifact_id)

            deployment_status = await apps_manager.get_status()
            assert artifact_id in deployment_status

            # Test registered Hypha service
            deployment_service_id = deployment_status["service_id"]
            deployment_service = await server.get_service(deployment_service_id)

            # Call the deployed application
            deployment_name = deployment_status[artifact_id]["deployment_name"]
            response = await deployment_service[deployment_name]["ping"]()
            apps_manager.logger.info(f"Response from deployed application: {response}")

            response = await deployment_service[deployment_name]["train"]()
            apps_manager.logger.info(f"Response from deployed application: {response}")

            # Keep server running if requested
            if keep_running:
                print("Server running. Press Ctrl+C to stop.")
                await server.serve()

            # Undeploy the test artifact
            await apps_manager.undeploy_artifact(artifact_id, server.context)

            # Deploy again
            await apps_manager.deploy_artifact(artifact_id)

            # Clean up deployments
            await apps_manager.cleanup_deployments()

        except Exception as e:
            print(f"An error occurred: {e}")
            raise e
        finally:
            await ray_cluster.stop()

    # Run the test
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test_create_artifact":
        # Run only the create_artifact test (no Ray cluster needed)
        asyncio.run(test_create_artifact())
    else:
        # Run the full deployment manager test
        asyncio.run(test_apps_manager(keep_running=True))
