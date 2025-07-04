import asyncio
import base64
import json
import logging
import os
import pickle
from functools import partial, wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import httpx
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
        self.ray_cluster = ray_cluster

        if self.ray_cluster.mode == "slurm":
            # SLURM workers always mount to /tmp/bioengine and /data
            self.apps_cache_dir = Path("/tmp/bioengine/apps")
            self.apps_data_dir = Path("/data")
        elif self.ray_cluster.mode == "single-machine":
            # Resolve local paths to ensure they are absolute
            self.apps_cache_dir = Path(apps_cache_dir).resolve()
            self.apps_data_dir = Path(apps_data_dir).resolve()
        elif self.ray_cluster.mode == "external-cluster":
            # For external clusters, use the provided paths directly
            self.apps_cache_dir = Path(apps_cache_dir)
            self.apps_data_dir = Path(apps_data_dir)
        else:
            raise ValueError(
                f"Unsupported Ray cluster mode: {self.ray_cluster.mode}. "
                "Supported modes are 'slurm', 'single-machine', and 'external-cluster'."
            )

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
    ) -> None:
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

    async def _ray_serve_get_request(
        self,
        route: str,
    ) -> Any:
        """
        Make a simple HTTP GET request to a Ray Serve route.

        Args:
            route: The route to call on the Ray Serve HTTP API

        Returns:
            Any: The response from the Ray Serve route
        """
        serve_base_url = self.ray_cluster.serve_http_url
        route = route.lstrip("/")  # Ensure no leading slash
        endpoint_url = f"{serve_base_url}/{route}"

        # Use appropriate timeouts for internal Ray Serve requests
        # Connect: 10s, Read: 600s (10 minutes for long-running _async_init and _test_deployment methods)
        # Write: 30s (sufficient for simple GET requests), Pool: 30s (connection pool timeout)
        timeout = httpx.Timeout(connect=10.0, read=600.0, write=30.0, pool=30.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(endpoint_url)

        if response.status_code != 200:
            raise RuntimeError(
                f"Error accessing Ray Serve URL '{endpoint_url}': "
                f"HTTP {response.status_code} - {response.text}"
            )

        return response.json()

    async def _get_serve_applications(self) -> Dict[str, str]:
        """
        Get all currently deployed Ray Serve applications.

        Returns:
        """
        available_routes = await self._ray_serve_get_request(route="-/routes")

        return list(available_routes.values())

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

    def _add_init_wrapper(self, deployment_class: Any) -> Any:
        orig_init = getattr(deployment_class, "__init__")

        @wraps(orig_init)
        def wrapped_init(self, *args, **kwargs):
            import os
            from pathlib import Path

            workdir = Path(os.environ["BIOENGINE_WORKDIR"])
            workdir.mkdir(parents=True, exist_ok=True)
            os.chdir(workdir)
            return orig_init(self, *args, **kwargs)

        deployment_class.__init__ = wrapped_init
        return deployment_class

    def _add_multiplexed_method(self, deployment_class: Any, class_config: dict) -> Any:
        """
        Decorate the `_get_model` method with @serve.multiplexed if it exists.

        This allows the method to handle multiple models per replica efficiently.

        Args:
            deployment_class: The class containing the `_get_model` method
            class_config: Configuration dictionary for the deployment class
        """
        orig_method = getattr(deployment_class, "_get_model", None)
        if orig_method:
            max_num_models_per_replica = int(
                class_config.get("max_num_models_per_replica", 3)
            )
            decorated_method = serve.multiplexed(
                orig_method, max_num_models_per_replica=max_num_models_per_replica
            )
            deployment_class._get_model = decorated_method
            self.logger.debug(
                f"Added @serve.multiplexed decorator to method '_get_model' in class '{class_config['class_name']}' "
                f"(max_num_models_per_replica={max_num_models_per_replica})."
            )
        return deployment_class

    def _add_http_handler(self, deployment_class: Any, class_config: dict) -> Any:
        """
        Create a single HTTP handler that routes to exposed methods.

        The __call__ method is reserved for the HTTP handler and must not be defined
        by the deployment class or listed in exposed_methods.

        Internal methods like _async_init and _test_deployment are never exposed via HTTP.
        """

        streaming_chunk_size = class_config.get("streaming_chunk_size", 1024 * 1024)
        exposed_methods = class_config.get("exposed_methods", {})

        # Validate that __call__ is not defined by the user (check if it's a custom method)
        if "__call__" in deployment_class.__dict__:
            raise ValueError(
                f"Class {deployment_class.__name__} must not define '__call__' method. "
                "The '__call__' method is reserved for the HTTP handler."
            )

        # Validate that __call__, _async_init and _test_deployment are not in exposed_methods
        for method_name in ["__call__", "_async_init", "_test_deployment"]:
            if method_name in exposed_methods:
                raise ValueError(
                    f"Method '{method_name}' cannot be listed in exposed_methods for class {deployment_class.__name__}. "
                    "This method is reserved for internal use and cannot be exposed."
                )

        # Only methods in exposed_methods are accessible via HTTP
        if exposed_methods:
            for method_name in exposed_methods.keys():
                if not hasattr(deployment_class, method_name):
                    raise ValueError(
                        f"Method '{method_name}' specified in 'exposed_methods' "
                        f"does not exist in class {deployment_class.__name__}."
                    )
        else:
            raise ValueError(
                f"Class {deployment_class.__name__} must define 'exposed_methods' "
                "to specify which methods are accessible via HTTP."
            )

        # Store HTTP handler configuration on the class to avoid closure references
        deployment_class._bioengine_http_config = {
            "exposed_methods": list(exposed_methods.keys()),
            "streaming_chunk_size": streaming_chunk_size,
        }

        # Create completely self-contained HTTP handler that gets config from class attributes
        def create_http_handler():
            async def http_handler(self, request):
                import asyncio
                import json
                import pickle

                from starlette.responses import JSONResponse, StreamingResponse

                try:
                    # Get configuration from class attribute (no closure references)
                    config = getattr(self.__class__, "_bioengine_http_config", {})
                    exposed_method_names = config.get("exposed_methods", [])
                    streaming_chunk_size = config.get(
                        "streaming_chunk_size", 1024 * 1024
                    )

                    # Parse the request path to get the method name
                    path = request.url.path
                    # Remove deployment prefix and get method name
                    path_parts = [p for p in path.split("/") if p]

                    # Determine which method to call
                    method_name = None

                    if len(path_parts) > 1:
                        # Method specified in path: /deployment/method_name
                        potential_method = path_parts[-1]
                        if potential_method in exposed_method_names + [
                            "_async_init",
                            "_test_deployment",
                        ]:
                            method_name = potential_method
                        else:
                            return JSONResponse(
                                {
                                    "success": False,
                                    "error": f"Method '{potential_method}' not found in exposed methods: {exposed_method_names}",
                                    "error_type": "MethodNotFound",
                                },
                                status_code=404,
                            )
                    else:
                        # No method specified in path: /deployment
                        # Check if this is a ping request (GET with no data)
                        if request.method == "GET" and not dict(request.query_params):
                            # Return ping response with basic deployment info
                            return JSONResponse(
                                {
                                    "success": True,
                                    "data_type": "ping",
                                }
                            )
                        elif len(exposed_method_names) == 1:
                            # If only one method is exposed, use it as default
                            method_name = exposed_method_names[0]
                        else:
                            # Multiple methods available, must specify one
                            return JSONResponse(
                                {
                                    "success": False,
                                    "error": f"No method specified in path. Available methods: {exposed_method_names}",
                                    "error_type": "MethodNotFound",
                                },
                                status_code=404,
                            )

                    print(f"Handling HTTP request for method '{method_name}'")

                    # Parse request data
                    if request.method == "POST":
                        content_type = request.headers.get("content-type", "")
                        if "application/json" in content_type:
                            request_data = await request.json()
                        elif "application/octet-stream" in content_type:
                            # Handle streamed/chunked data
                            body = await request.body()
                            # Deserialize the body using pickle
                            request_data = await asyncio.to_thread(pickle.loads, body)
                        else:
                            request_data = {}
                    else:
                        request_data = dict(request.query_params)

                    # Extract args and kwargs from request
                    args = request_data.get("args", [])
                    kwargs = request_data.get("kwargs", {})

                    print(
                        f"Calling method '{method_name}' with {len(args)} args and {len(kwargs)} kwargs"
                    )

                    # Get and call the target method from the instance
                    target_method = getattr(self, method_name)

                    # Call the method
                    if asyncio.iscoroutinefunction(target_method):
                        result = await target_method(*args, **kwargs)
                    else:
                        result = await asyncio.to_thread(target_method, *args, **kwargs)

                    # Try JSON serialization first, fall back to streaming if it fails
                    try:
                        # Try to JSON serialize the result
                        json.dumps(result)

                        print(f"Returning JSON result from method '{method_name}'")
                        return JSONResponse(
                            {"success": True, "result": result, "data_type": "json"}
                        )

                    except (TypeError, ValueError):
                        # JSON serialization failed, stream it instead
                        print(
                            f"Streaming non-JSON-serializable result from method '{method_name}'"
                        )

                        # Handle memoryview objects by converting to bytes first
                        if isinstance(result, memoryview):
                            result = bytes(result)

                        # Serialize and chunk the result for streaming
                        serialized_data = await asyncio.to_thread(pickle.dumps, result)

                        async def stream_chunks():
                            # Split large data into chunks for streaming
                            chunks = [
                                serialized_data[i : i + streaming_chunk_size]
                                for i in range(
                                    0, len(serialized_data), streaming_chunk_size
                                )
                            ]
                            for chunk in chunks:
                                yield chunk

                        return StreamingResponse(
                            stream_chunks(),
                            media_type="application/octet-stream",
                            headers={"X-Data-Type": "chunked-pickle"},
                        )

                except Exception as e:
                    print(f"Error in HTTP request: {e}")
                    return JSONResponse(
                        {
                            "success": False,
                            "error": str(e),
                            "error_type": type(e).__name__,
                        },
                        status_code=500,
                    )

            return http_handler

        # Add the HTTP handler as __call__ method
        deployment_class.__call__ = create_http_handler()

        return deployment_class

    async def _load_deployment_code(
        self,
        class_config: dict,
        artifact_id: str,
        version: str,
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

                # Download the file content with timeout (30s for all operations)
                download_timeout = httpx.Timeout(30.0)
                async with httpx.AsyncClient(timeout=download_timeout) as client:
                    response = await client.get(download_url)

                if response.status_code != 200:
                    raise RuntimeError(
                        f"Failed to download deployment code from {download_url}: "
                        f"HTTP {response.status_code} - {response.text}"
                    )

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

            # Wrap the class `__init__` method to ensure it uses the workdir
            deployment_class = self._add_init_wrapper(deployment_class)

            # Add @serve.multiplexed decorator to the `_get_model`` method if it exists
            deployment_class = self._add_multiplexed_method(
                deployment_class, class_config
            )

            # Add HTTP handler for streaming support as the `__call__` method
            deployment_class = self._add_http_handler(deployment_class, class_config)

            self.logger.debug(
                f"Loaded class '{class_config['class_name']}' from artifact '{artifact_id}'."
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

            self.logger.info(f"Using mode '{mode_key}' for deployment '{artifact_id}'.")

        # Add default ray_actor_options if not present
        ray_actor_options = deployment_config.setdefault("ray_actor_options", {})
        ray_actor_options.setdefault("num_cpus", 1)
        ray_actor_options.setdefault("num_gpus", 0)
        ray_actor_options.setdefault("memory", 0)

        # Check if the required resources are available
        insufficient_resources = True
        while not self.ray_cluster.status["nodes"]:
            # Wait for Ray cluster to be ready
            await asyncio.sleep(1)
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

        # Add extra environment variables to the deployment
        runtime_env = ray_actor_options.setdefault("runtime_env", {})
        env_vars = runtime_env.setdefault("env_vars", {})

        # Set standard directories to ensure it only uses the specified workdir
        deployment_workdir = self.apps_cache_dir / deployment_name
        env_vars["BIOENGINE_WORKDIR"] = str(deployment_workdir)
        env_vars["HOME"] = str(deployment_workdir)
        env_vars["TMPDIR"] = str(deployment_workdir / "tmp")

        # Pass the data directory to the deployment
        env_vars["BIOENGINE_DATA_DIR"] = str(self.apps_data_dir)

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
        deployment_info["async_init"] = hasattr(deployment_class, "_async_init")
        deployment_info["test_deployment"] = hasattr(
            deployment_class, "_test_deployment"
        )

        # Create the Ray Serve deployment
        deployment = serve.deployment(**deployment_config)(deployment_class)

        # Bind the arguments to the deployment and return an Application
        app = deployment.bind()

        return app

    async def _ping_deployment(self, deployment_name: str) -> None:
        ping_result = await self._ray_serve_get_request(route=f"/{deployment_name}")
        # Verify this is a proper ping response
        if not (
            isinstance(ping_result, dict)
            and ping_result.get("success") == True
            and ping_result.get("data_type") == "ping"
        ):
            raise RuntimeError(
                f"Deployment '{deployment_name}' failed to start or is not accessible. Ping response: {ping_result}"
            )
        else:
            self.logger.debug(
                f"Deployment '{deployment_name}' is accessible and ready."
            )

    async def _run_deployment_async_init(self, deployment_name: str):
        init_result = await self._ray_serve_get_request(
            route=f"/{deployment_name}/_async_init"
        )
        if not (
            isinstance(init_result, dict)
            and init_result.get("success") == True
            and init_result.get("data_type") == "json"
            and init_result.get("result", "") is None
        ):
            raise RuntimeError(
                f"Deployment '{deployment_name}' failed async initialization. Response: {init_result}"
            )
        else:
            self.logger.debug(
                f"Deployment '{deployment_name}' async initialization completed successfully."
            )

    async def _run_deployment_test(self, deployment_name: str):
        test_result = await self._ray_serve_get_request(
            route=f"/{deployment_name}/_test_deployment"
        )
        if not (
            isinstance(test_result, dict)
            and test_result.get("success") == True
            and test_result.get("data_type") == "json"
            and test_result.get("result", False) is True
        ):
            raise RuntimeError(
                f"Deployment '{deployment_name}' failed its test check. Response: {test_result}"
            )
        else:
            self.logger.debug(f"Deployment '{deployment_name}' passed its test check.")

    async def _ray_serve_post_request(
        self,
        deployment_name: str,
        method_name: str,
        user_id: str,
        args: list,
        kwargs: dict,
    ) -> Any:
        """
        Make an HTTP request to a Ray Serve deployment.
        This method handles both JSON and binary streaming requests to the Ray Serve
        HTTP API, allowing for flexible data handling based on the request content.

        Args:
            deployment_name: Name of the Ray Serve deployment
            method_name: Method to call on the deployment
            args: Positional arguments for the method
            kwargs: Keyword arguments for the method

        Returns:
            Any: The result of the method call, either as JSON or binary data

        Raises:
            RuntimeError: If the HTTP request fails or returns an error
            ValueError: If the response cannot be parsed as JSON
        """
        try:
            self.logger.info(
                f"User '{user_id}' is calling method '{method_name}' on deployment '{deployment_name}'."
            )
            # Prepare request data
            request_data = {"args": args, "kwargs": kwargs}

            # Determine if we need to stream the request
            # Check if request_data contains any non-JSON-serializable data
            try:
                json.dumps(request_data)
                needs_streaming = False
            except (TypeError, ValueError):
                # JSON serialization failed, so it contains non-serializable data
                needs_streaming = True

            # Get Ray Serve HTTP URL
            serve_base_url = self.ray_cluster.serve_http_url
            if method_name == "__call__":
                endpoint_url = f"{serve_base_url}/{deployment_name}"
            else:
                endpoint_url = f"{serve_base_url}/{deployment_name}/{method_name}"

            # Make HTTP request to Ray Serve
            # Connect: 10s (consistent with other methods), Read: 300s (5 minutes for user methods),
            # Write: 300s (for large data uploads), Pool: 30s (connection pool timeout)
            timeout = httpx.Timeout(connect=10.0, read=300.0, write=300.0, pool=30.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                if needs_streaming:
                    # Handle memoryview objects by converting to bytes first
                    if isinstance(request_data, memoryview):
                        request_data = bytes(request_data)
                    serialized_data = await asyncio.to_thread(
                        pickle.dumps, request_data
                    )

                    # Send data as binary stream
                    response = await client.post(
                        endpoint_url,
                        content=serialized_data,
                        headers={"Content-Type": "application/octet-stream"},
                    )
                else:
                    # Send data as JSON
                    response = await client.post(
                        endpoint_url,
                        json=request_data,
                        headers={"Content-Type": "application/json"},
                    )

            # Handle response
            if response.status_code != 200:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                raise RuntimeError(error_msg)

            content_type = response.headers.get("content-type", "")
            if "application/octet-stream" in content_type:
                # Handle streamed response
                if response.headers.get("X-Data-Type") == "chunked-pickle":
                    # Reassemble chunks
                    chunks = []
                    async for chunk in response.aiter_bytes():
                        chunks.append(chunk)
                    reassembled_data = b"".join(chunks)
                    # Deserialize the reassembled data
                    return await asyncio.to_thread(pickle.loads, reassembled_data)
                else:
                    # Return raw bytes
                    return await response.aread()
            else:
                # Handle JSON response
                try:
                    response_data = response.json()
                    if not response_data.get("success", True):
                        error_type = response_data.get("error_type", "Unknown")
                        error_msg = response_data.get("error", "Unknown error")
                        raise RuntimeError(f"{error_type}: {error_msg}")
                    return response_data.get("result")
                except (json.JSONDecodeError, ValueError) as e:
                    # Handle non-JSON responses
                    self.logger.warning(
                        f"Non-JSON response received: {response.text[:200]}"
                    )
                    return response.text

        except Exception as e:
            raise RuntimeError(
                f"Failed to call '{method_name}' on deployment '{deployment_name}': {e}"
            )

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
                self.logger.debug("No deployments to register as services.")
                self.service_info = None
                return

            async def create_deployment_function(
                *args,
                context,
                deployment_name,
                method_name,
                authorized_users,
                **kwargs,
            ):
                self._check_permissions(
                    context=context,
                    authorized_users=authorized_users,
                    resource_name=f"deployment '{deployment_name}' method '{method_name}'",
                )

                # Call the deployment method via Ray Serve HTTP API
                return await self._ray_serve_post_request(
                    deployment_name=deployment_name,
                    method_name=method_name,
                    user_id=context["user"]["id"],
                    args=args,
                    kwargs=kwargs,
                )

            # Create service functions for each deployment
            service_functions = {}
            for deployment_info in self._deployed_artifacts.values():
                if deployment_info["is_deployed"] is False:
                    continue
                deployment_name = deployment_info["deployment_name"]
                service_functions[deployment_name] = {}
                class_config = deployment_info["class_config"]
                for method_name, method_config in class_config[
                    "exposed_methods"
                ].items():
                    service_functions[deployment_name][method_name] = partial(
                        create_deployment_function,
                        deployment_name=deployment_name,
                        method_name=method_name,
                        authorized_users=method_config.get("authorized_users", "*"),
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
            self.logger.debug("Successfully registered deployment service.")
            server_url = self.server.config.public_base_url
            workspace, sid = service_info.id.split("/")
            service_url = f"{server_url}/{workspace}/services/{sid}"
            for deployment_name in service_functions.keys():
                self.logger.debug(
                    f"Access the deployment '{deployment_name}' at: {service_url}/{deployment_name}"
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
            applications = await self._get_serve_applications()
            if deployment_name in applications:
                self.logger.info(
                    f"User '{user_id}' is updating existing deployment for artifact '{artifact_id}'..."
                )
            else:
                self.logger.info(
                    f"User '{user_id}' is starting a new deployment for artifact '{artifact_id}'..."
                )

            # Create the deployment from the artifact
            app = await self._create_deployment(
                artifact_id=artifact_id,
                mode=mode,
                version=version,
                deployment_name=deployment_name,
            )

            # Run the deployment in Ray Serve with unique route prefix
            deployment_coroutine = asyncio.to_thread(
                serve.run,
                target=app,
                name=deployment_name,
                route_prefix=f"/{deployment_name}",
                blocking=False,
            )

            if not skip_update and self.ray_cluster.mode == "slurm":
                # Notify the autoscaling system of the new deployment
                await self.ray_cluster.notify()

            # Await the coroutine to start the deployment
            await deployment_coroutine

            # Validate the deployment is accessible via HTTP (using ping endpoint)
            await self._ping_deployment(deployment_name)

            # Run async init if provided
            if self._deployed_artifacts[artifact_id]["async_init"]:
                await self._run_deployment_async_init(deployment_name)

            # Run deployment test if provided
            if self._deployed_artifacts[artifact_id]["test_deployment"]:
                await self._run_deployment_test(deployment_name)

            self._deployed_artifacts[artifact_id]["is_deployed"] = True

            # Update services with the new deployment
            if not skip_update:
                await self._update_services()

            # Track the deployment in the internal state
            self.logger.info(
                f"Successfully completed deployment of artifact '{artifact_id}'."
            )

            # Keep the deployment task running until cancelled
            while True:
                await asyncio.sleep(1)

        except asyncio.CancelledError:
            self.logger.info(
                f"Deployment task for artifact '{artifact_id}' was cancelled."
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
                    self.logger.debug(
                        f"Deleted Ray Serve deployment '{deployment_name}'."
                    )
                except Exception as delete_err:
                    self.logger.error(
                        f"Error deleting Ray Serve deployment {deployment_name}: {delete_err}"
                    )

                # Remove from deployment tracking
                del self._deployed_artifacts[artifact_id]
                self.logger.debug(
                    f"Removed artifact '{artifact_id}' from deployment tracking."
                )

                # Update services with removed deployment
                if not skip_update:
                    await self._update_services()

                self.logger.info(f"Undeployment of artifact '{artifact_id}' completed.")

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
            self.logger.debug("Successfully connected to artifact manager.")

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
                        await asyncio.wait_for(
                            deployment_info["deployment_task"], timeout=timeout
                        )
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
                "test_deployment": False,
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
        while not all(
            self._deployed_artifacts.get(artifact_id)
            and self._deployed_artifacts.get(artifact_id)["is_deployed"]
            for artifact_id in artifact_ids
        ):
            await asyncio.sleep(1)

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
                f"User '{user_id}' is starting undeployment of artifact '{artifact_id}'..."
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

        await self.ray_cluster.check_connection()

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
                    self._deployed_artifacts.pop(artifact_id, None)
                    # Note: This can happen if deploy_artifact and undeploy_artifact are called at the same time
                continue
            if len(application.deployments) > 1:
                raise NotImplementedError

            class_config = deployment_info["class_config"]
            class_methods = class_config["exposed_methods"]
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
                self.logger.debug(f"Editing existing artifact '{full_artifact_id}'...")
                artifact = await self.artifact_manager.edit(
                    artifact_id=full_artifact_id,
                    manifest=deployment_manifest,
                    type="application",
                    stage=True,
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
                        f"Bioengine Apps collection created with ID: {collection.id}."
                    )

            # Create new artifact using alias
            self.logger.debug(f"Creating new artifact with alias '{alias}'...")
            artifact = await self.artifact_manager.create(
                alias=alias,
                parent_id=collection_id,
                manifest=deployment_manifest,
                type=deployment_manifest.get("type", "application"),
                stage=True,
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
        self.logger.info(f"Committed artifact with ID: {artifact.id}.")

        return artifact.id

    async def delete_artifact(
        self, artifact_id: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Delete a deployment artifact from Hypha.

        Args:
            artifact_id: ID of the artifact to delete
            context: Optional context information from Hypha request containing user info

        Returns:
            str: The ID of the deleted artifact

        Raises:
            ValueError: If the artifact does not exist or cannot be deleted
        """
        self._check_initialized()

        # Check user permissions
        self._check_permissions(
            context=context,
            authorized_users=self.admin_users,
            resource_name=f"deletion of artifact '{artifact_id}'",
        )

        # Get the full artifact ID
        self.logger.debug(f"Deleting artifact '{artifact_id}'...")
        artifact_id = self._get_full_artifact_id(artifact_id)

        # Delete the artifact
        await self.artifact_manager.delete(artifact_id)
        self.logger.info(f"Successfully deleted artifact '{artifact_id}'.")
