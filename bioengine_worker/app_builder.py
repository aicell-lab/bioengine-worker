import inspect
import logging
import os
import time
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, Union

import httpx
import yaml
from hypha_rpc.rpc import RemoteService
from hypha_rpc.utils import ObjectProxy
from ray import serve
from ray.serve.handle import DeploymentHandle

from bioengine_worker.proxy_deployment import BioEngineProxyDeployment
from bioengine_worker.utils import create_logger, update_requirements


class AppManifest(TypedDict):
    """Type definition for application manifest structure."""

    name: str
    id: str
    id_emoji: str
    description: str
    type: str
    deployments: List[str]
    authorized_users: List[str]


class AppBuilder:
    """
    A builder class for creating and managing BioEngine applications from deployment artifacts.

    The AppBuilder handles the complete lifecycle of application deployment including:
    - Loading deployment artifacts and manifests
    - Setting up Ray Serve deployments with proper environment configuration
    - Managing resource allocation and dependencies
    - Creating proxy deployments for RTC communication
    - Validating and testing deployments during initialization

    Attributes:
        logger: Logger instance for the AppBuilder
        apps_cache_dir: Directory for caching deployment artifacts
        apps_data_dir: Directory accessible to deployments for data storage
        server: Remote service connection to Hypha server
        artifact_manager: Service for managing deployment artifacts
        serve_http_url: HTTP URL for Ray Serve endpoints
    """

    def __init__(
        self,
        token: str,
        apps_cache_dir: Path,
        apps_data_dir: Path,
        log_file: Optional[str] = None,
        debug: bool = False,
    ) -> None:
        """
        Initialize the AppBuilder with configuration parameters.

        Args:
            token: Authentication token for Hypha server access
            apps_cache_dir: Cache directory for deployment artifacts and workspaces
            apps_data_dir: Data directory accessible to all deployments
            log_file: Optional path to log file for output (defaults to console logging)
            debug: Enable debug-level logging for detailed output

        Raises:
            ValueError: If required directories cannot be created or accessed
        """
        # Set up logging
        self.logger = create_logger(
            name="AppBuilder",
            level=logging.DEBUG if debug else logging.INFO,
            log_file=log_file,
        )

        # Store parameters
        self._token = token
        self.apps_cache_dir = apps_cache_dir
        self.apps_data_dir = apps_data_dir
        self.server: Optional[RemoteService] = None
        self.artifact_manager: Optional[ObjectProxy] = None
        self.serve_http_url: Optional[str] = None

    def initialize(
        self, server: RemoteService, artifact_manager: ObjectProxy, serve_http_url: str
    ) -> None:
        """
        Initialize the AppBuilder with required external services.

        This method must be called after construction to set up the connection
        to the Hypha server and artifact manager service.

        Args:
            server: Connected RemoteService instance to Hypha server
            artifact_manager: ObjectProxy to the artifact management service
            serve_http_url: Base HTTP URL for Ray Serve deployment endpoints

        Raises:
            ValueError: If any of the provided services are invalid or not properly connected
        """
        # Store server connection and artifact manager
        if not server or not isinstance(server, RemoteService):
            raise ValueError("Invalid server connection provided.")
        if not artifact_manager or not isinstance(artifact_manager, ObjectProxy):
            raise ValueError("Invalid artifact manager provided.")
        if (
            not serve_http_url
            or not isinstance(serve_http_url, str)
            or not serve_http_url.startswith("http")
        ):
            raise ValueError("Invalid serve HTTP URL provided.")
        self.server = server
        self.artifact_manager = artifact_manager
        self.serve_http_url = serve_http_url

    async def _load_manifest(
        self, artifact_id: str, version: Optional[str] = None
    ) -> AppManifest:
        """
        Load and validate the artifact manifest from the artifact manager or local filesystem.

        Args:
            artifact_id: The ID of the artifact to load (format: 'workspace/artifact_alias')
            version: The version of the artifact to load (defaults to latest)

        Returns:
            The loaded and validated manifest dictionary containing deployment configuration

        Raises:
            FileNotFoundError: If local manifest file is not found when using local artifact path
            ValueError: If the manifest is invalid, missing required fields, or has incorrect format
            Exception: If there is an error loading the manifest from the artifact manager
        """
        if os.environ.get("BIOENGINE_WORKER_LOCAL_ARTIFACT_PATH"):
            # Load the file content from local path
            artifact_folder = artifact_id.split("/")[1].replace("-", "_")
            local_deployments_dir = Path(
                os.environ["BIOENGINE_WORKER_LOCAL_ARTIFACT_PATH"]
            )
            local_path = local_deployments_dir / artifact_folder / "manifest.yaml"
            if not local_path.exists():
                raise FileNotFoundError(f"Local manifest file not found: {local_path}")
            with open(local_path, "r") as f:
                manifest = yaml.safe_load(f)
        else:
            artifact = await self.artifact_manager.read(artifact_id, version=version)
            manifest = artifact.get("manifest")
            if manifest is None:
                raise ValueError(f"Manifest not found in artifact {artifact_id}.")

        required_fields = [
            "name",
            "id",
            "id_emoji",
            "description",
            "type",
            "deployments",
            "authorized_users",
        ]
        for field in required_fields:
            if field not in manifest:
                raise ValueError(f"Manifest is missing required field: {field}")

        if manifest["type"] != "application":
            raise ValueError(
                f"Invalid manifest type: {manifest['type']}. Expected 'application'."
            )

        deployments = manifest["deployments"]
        if not isinstance(deployments, list) and len(deployments) > 0:
            raise ValueError(
                f"Invalid deployments format in artifact. "
                "Expected a list of deployment descriptions in the format 'python_file:class_name'."
            )
        authorized_users = manifest["authorized_users"]
        if not isinstance(authorized_users, list) and len(authorized_users) > 0:
            raise ValueError(
                f"Invalid authorized users format in artifact. "
                "Expected a list of user IDs or '*' for all users."
            )

        return manifest

    def _update_actor_options(
        self,
        deployment: serve.Deployment,
        application_id: str,
        enable_gpu: bool,
        token: str,
    ) -> serve.Deployment:
        """
        Update Ray actor options for a deployment with BioEngine-specific configuration.

        Adds required dependencies, environment variables, and directory structure
        needed for BioEngine deployments to function properly.

        Args:
            deployment: The Ray Serve deployment to update
            application_id: Unique identifier for the application instance
            enable_gpu: Flag indicating whether GPU support is enabled
            token: Authentication token for Hypha server access

        Returns:
            Updated deployment with configured actor options

        Note:
            This method modifies the deployment's runtime environment to include:
            - BioEngine pip requirements
            - Working directory setup
            - Hypha server connection parameters
            - Data directory access paths
        """
        ray_actor_options = deployment.ray_actor_options.copy()

        # Disable GPU if not enabled
        if not enable_gpu:
            ray_actor_options["num_gpus"] = 0

        # Update runtime environment with BioEngine requirements
        runtime_env = ray_actor_options.setdefault("runtime_env", {})
        pip_requirements = runtime_env.setdefault("pip", [])
        env_vars = runtime_env.setdefault("env_vars", {})

        # Update with BioEngine requirements
        pip_requirements = update_requirements(pip_requirements)

        # Set BioEngine environment variables
        app_work_dir = self.apps_cache_dir / application_id
        env_vars["BIOENGINE_WORKDIR"] = str(app_work_dir)
        env_vars["HOME"] = str(app_work_dir)
        env_vars["TMPDIR"] = str(app_work_dir / "tmp")

        # Pass the data directory to the deployment
        env_vars["BIOENGINE_DATA_DIR"] = str(self.apps_data_dir)

        env_vars["HYPHA_SERVER_URL"] = self.server.config.public_base_url
        env_vars["HYPHA_WORKSPACE"] = self.server.config.workspace
        env_vars["HYPHA_TOKEN"] = token

        updated_deployment = deployment.options(ray_actor_options=ray_actor_options)
        return updated_deployment

    def _update_init(self, deployment: serve.Deployment) -> serve.Deployment:
        """
        Update the __init__ method of the deployment class to set up BioEngine environment.

        Wraps the original __init__ method to ensure proper working directory setup
        and deployment state initialization.

        Args:
            deployment: The Ray Serve deployment to update

        Returns:
            Updated deployment with wrapped __init__ method
        """

        orig_init = getattr(deployment.func_or_class, "__init__")

        @wraps(orig_init)
        def wrapped_init(self, *args, **kwargs):
            import asyncio
            import os
            from pathlib import Path

            # Get replica identifier for logging
            try:
                self.replica_id = serve.get_replica_context().replica_tag
            except Exception:
                self.replica_id = "unknown"

            # Ensure the workdir is set to the BIOENGINE_WORKDIR environment variable
            workdir = Path(os.environ["BIOENGINE_WORKDIR"]).resolve()
            os.environ["BIOENGINE_WORKDIR"] = str(workdir)
            workdir.mkdir(parents=True, exist_ok=True)
            os.chdir(workdir)
            print(f"📁 [{self.replica_id}] Working directory: {workdir}/")

            # Log data directory
            data_dir = Path(os.environ["BIOENGINE_DATA_DIR"]).resolve()
            if data_dir.exists() and data_dir.is_dir():
                os.environ["BIOENGINE_DATA_DIR"] = str(data_dir)
                print(f"📂 [{self.replica_id}] Data directory: {data_dir}/")
            else:
                del os.environ["BIOENGINE_DATA_DIR"]
                print(f"📂 [{self.replica_id}] Data directory {data_dir}/ not found.")

            # Initialize deployment states
            self._deployment_initialized = False
            self._deployment_tested = False

            # Create a health check lock to synchronize health checks
            self._health_check_lock = asyncio.Lock()

            # Call the original __init__ method
            orig_init(self, *args, **kwargs)

        setattr(deployment.func_or_class, "__init__", wrapped_init)
        return deployment

    def _update_async_init(self, deployment: serve.Deployment) -> serve.Deployment:
        """
        Update the async_init method of the deployment class for post-initialization setup.

        Wraps the original async_init method to handle both sync and async implementations
        and track initialization state.

        Args:
            deployment: The Ray Serve deployment to update

        Returns:
            Updated deployment with wrapped async_init method
        """
        orig_async_init = getattr(
            deployment.func_or_class, "async_init", lambda self: None
        )

        @wraps(orig_async_init)
        async def wrapped_async_init(self):
            import asyncio

            start_time = time.time()
            print(
                f"⚡ [{self.replica_id}] Running async initialization for '{self.__class__.__name__}'..."
            )

            try:
                # Check if the original async_init method is async
                if inspect.iscoroutinefunction(orig_async_init):
                    await orig_async_init(self)
                else:
                    # If it's a regular function, call it directly
                    await asyncio.to_thread(orig_async_init, self)

                elapsed_time = time.time() - start_time
                self._deployment_initialized = True
                print(
                    f"✅ [{self.replica_id}] Deployment '{self.__class__.__name__}' async initialized successfully in {elapsed_time:.2f}s"
                )
            except Exception as e:
                elapsed_time = time.time() - start_time
                print(
                    f"❌ [{self.replica_id}] Async initialization failed for '{self.__class__.__name__}' after {elapsed_time:.2f}s: {e}"
                )
                raise

        setattr(deployment.func_or_class, "async_init", wrapped_async_init)
        return deployment

    def _update_test_deployment(self, deployment: serve.Deployment) -> serve.Deployment:
        """
        Update the test_deployment method of the deployment class for testing functionality.

        Wraps the original test_deployment method to handle both sync and async implementations
        and track testing state.

        Args:
            deployment: The Ray Serve deployment to update

        Returns:
            Updated deployment with wrapped test_deployment method
        """
        orig_test_deployment = getattr(
            deployment.func_or_class, "test_deployment", lambda self: None
        )

        @wraps(orig_test_deployment)
        async def wrapped_test_deployment(self):
            import asyncio

            start_time = time.time()
            print(
                f"🧪 [{self.replica_id}] Running deployment test for '{self.__class__.__name__}'..."
            )

            try:
                # Check if the original test_deployment method is async
                if inspect.iscoroutinefunction(orig_test_deployment):
                    await orig_test_deployment(self)
                else:
                    # If it's a regular function, call it directly
                    await asyncio.to_thread(orig_test_deployment, self)

                # Mark the deployment as tested
                elapsed_time = time.time() - start_time
                self._deployment_tested = True
                print(
                    f"✅ [{self.replica_id}] Deployment '{self.__class__.__name__}' tested successfully in {elapsed_time:.2f}s"
                )

            except Exception as e:
                elapsed_time = time.time() - start_time
                print(
                    f"❌ [{self.replica_id}] Deployment test failed for '{self.__class__.__name__}' after {elapsed_time:.2f}s: {e}"
                )
                raise RuntimeError(
                    f"Deployment test failed for '{self.__class__.__name__}': {e}"
                )

        setattr(deployment.func_or_class, "test_deployment", wrapped_test_deployment)
        return deployment

    def _update_health_check(self, deployment: serve.Deployment) -> serve.Deployment:
        """
        Add a comprehensive health check method to the deployment class.

        This method is called by Ray Serve during actor initialization and keeps the
        deployment in "DEPLOYING" state until all initialization and testing passes.

        Args:
            deployment: The Ray Serve deployment to update

        Returns:
            Updated deployment with health check method

        Note:
            The health check ensures both async_init and test_deployment complete
            successfully before marking the deployment as healthy.
        """
        orig_health_check = getattr(
            deployment.func_or_class, "check_health", lambda self: None
        )

        @wraps(orig_health_check)
        async def check_health(self):
            import asyncio

            async with self._health_check_lock:
                # Ensure async initialization has completed
                if not getattr(self, "_deployment_initialized", False):
                    await self.async_init()

                # Ensure deployment testing has completed
                if not getattr(self, "_deployment_tested", False):
                    await self.test_deployment()

                try:
                    # Check if the original health check method is async
                    if inspect.iscoroutinefunction(orig_health_check):
                        result = await orig_health_check(self)
                    else:
                        # If it's a regular function, call it directly
                        result = await asyncio.to_thread(orig_health_check, self)

                    return result

                except Exception as e:
                    print(
                        f"❌ [{self.replica_id}] Health check failed for '{self.__class__.__name__}': {e}"
                    )
                    raise

        # Add the updated health check method to the deployment class
        setattr(deployment.func_or_class, "check_health", check_health)
        return deployment

    def _get_init_param_info(
        self, deployment: serve.Deployment
    ) -> Dict[str, Dict[str, Any]]:
        """
        Extract parameter information from a deployment class's __init__ method.

        Args:
            deployment: The Ray Serve deployment to inspect

        Returns:
            Dictionary mapping parameter names to their type and default value information

        Note:
            Excludes 'self', '*args', and '**kwargs' parameters from the returned dictionary
        """
        sig = inspect.signature(deployment.func_or_class.__init__)
        params = {}
        has_var_positional = False
        has_var_keyword = False

        for name, param in sig.parameters.items():
            if name == "self":
                continue
            # Skip *args and **kwargs parameters as they have special handling
            if param.kind == inspect.Parameter.VAR_POSITIONAL:
                has_var_positional = True
                continue
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                has_var_keyword = True
                continue

            param_type = (
                param.annotation
                if param.annotation is not inspect.Parameter.empty
                else None
            )
            default = (
                param.default if param.default is not inspect.Parameter.empty else None
            )
            params[name] = {
                "type": param_type,
                "default": default,
            }

        # Store information about *args and **kwargs for validation
        params["__has_var_positional__"] = {"type": None, "default": has_var_positional}
        params["__has_var_keyword__"] = {"type": None, "default": has_var_keyword}
        return params

    def _check_params(
        self, init_params: Dict[str, Dict[str, Any]], kwargs: Dict[str, Any]
    ) -> None:
        """
        Validate that provided kwargs match the expected init parameters.

        Args:
            init_params: Parameter information from _get_init_param_info
            kwargs: Keyword arguments to validate

        Raises:
            ValueError: If there are unexpected parameters, missing required parameters,
                       type mismatches, or *args parameters are detected
        """
        # Extract special flags for **kwargs handling (*args is not supported)
        has_var_keyword = init_params.get("__has_var_keyword__", {}).get(
            "default", False
        )

        # Remove special flags from init_params for normal processing
        filtered_init_params = {
            k: v for k, v in init_params.items() if not k.startswith("__has_var_")
        }

        # Check if all provided parameters are expected
        for key in kwargs:
            if key not in filtered_init_params:
                if not has_var_keyword:
                    raise ValueError(
                        f"Unexpected parameter '{key}' provided. "
                        f"Expected one of {list(filtered_init_params.keys())}."
                    )
                # If **kwargs is present, allow any additional parameters
            else:
                expected_type = filtered_init_params[key]["type"]
                if expected_type and not isinstance(kwargs[key], expected_type):
                    raise ValueError(
                        f"Parameter '{key}' must be of type {expected_type.__name__}, "
                        f"but got {type(kwargs[key]).__name__}."
                    )

        # Check if all required parameters are provided
        for key, param_info in filtered_init_params.items():
            if param_info["type"] == DeploymentHandle:
                # DeploymentHandle parameters are handled separately
                continue
            if param_info["default"] is None and key not in kwargs:
                raise ValueError(
                    f"Missing required parameter '{key}' of type {param_info['type'].__name__}."
                )

    async def _load_deployment(
        self,
        application_id: str,
        artifact_id: str,
        version: Optional[str],
        import_path: str,
        enable_gpu: bool,
        token: str,
    ) -> serve.Deployment:
        """
        Load and execute deployment code from an artifact directly in memory.

        Downloads and executes Python code from an artifact to create deployable classes.
        Supports both remote artifact loading and local file loading for development.

        Args:
            application_id: Unique identifier for the application instance
            artifact_id: ID of the artifact containing the deployment code
            version: Version of the artifact to load (defaults to latest)
            import_path: Import path in format 'python_file:class_name'
            enable_gpu: Whether to enable GPU support for the deployment
            token: Authentication token for Hypha server access

        Returns:
            Configured Ray Serve deployment ready for use

        Raises:
            FileNotFoundError: If local deployment file is not found
            ValueError: If import path format is invalid or class name is not found
            RuntimeError: If class loading or configuration fails
            Exception: If code execution or download fails
        """
        try:
            python_file, class_name = import_path.split(":")
            python_file = f"{python_file}.py"  # Add .py extension
        except ValueError:
            raise ValueError(
                f"Invalid import path format: {import_path}. "
                "Expected format is 'python_file:class_name'."
            )

        if os.environ.get("BIOENGINE_WORKER_LOCAL_ARTIFACT_PATH"):
            # Load the file content from local path
            artifact_folder = artifact_id.split("/")[1].replace("-", "_")
            self.logger.debug(
                f"Loading deployment code from local path: {python_file} in folder {artifact_folder}/"
            )
            local_deployments_dir = Path(
                os.environ["BIOENGINE_WORKER_LOCAL_ARTIFACT_PATH"]
            )
            local_path = local_deployments_dir / artifact_folder / python_file
            if not local_path.exists():
                raise FileNotFoundError(
                    f"Local deployment file not found: {local_path}"
                )
            with open(local_path, "r") as f:
                code_content = f.read()
        else:
            local_path = None
            try:
                # Get download URL for the file
                download_url = await self.artifact_manager.get_file(
                    artifact_id=artifact_id,
                    version=version,
                    file_path=python_file,
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
            except Exception as e:
                self.logger.error(
                    f"Error downloading deployment code from {artifact_id}: {e}"
                )
                raise e

        # Create a restricted globals dictionary for sandboxed execution - pass some deployment options
        try:
            # Execute the code in a sandboxed environment
            safe_globals = {}
            exec(code_content, safe_globals)
            if class_name not in safe_globals:
                raise ValueError(f"{class_name} not found in {artifact_id}")
            deployment = safe_globals[class_name]
            if not deployment:
                raise RuntimeError(f"Error loading {class_name} from {artifact_id}")

            # Update environment variables and requirements
            deployment = self._update_actor_options(
                deployment, application_id, enable_gpu, token
            )

            # Update the deployment class methods
            deployment = self._update_init(deployment)
            deployment = self._update_async_init(deployment)
            deployment = self._update_test_deployment(deployment)
            deployment = self._update_health_check(deployment)

            if local_path:
                self.logger.info(
                    f"Successfully loaded and configured deployment class '{class_name}' from local path '{local_path}/'"
                )
            else:
                self.logger.info(
                    f"Successfully loaded and configured deployment class '{class_name}' from artifact '{artifact_id}'"
                )
            return deployment
        except Exception as e:
            self.logger.error(
                f"Error creating deployment class from {artifact_id}: {e}"
            )
            raise e

    def _calculate_required_resources(
        self, deployments: List[serve.Deployment]
    ) -> Dict[str, Union[int, float]]:
        """
        Calculate the total resource requirements for all deployments.

        Args:
            deployments: List of Ray Serve deployments to analyze

        Returns:
            Dictionary containing total CPU, GPU, and memory requirements
        """
        required_resources = {
            "num_cpus": 0,
            "num_gpus": 0,
            "memory": 0,
        }
        for deployment in deployments:
            ray_actor_options = deployment.ray_actor_options
            required_resources["num_cpus"] += ray_actor_options.get("num_cpus", 0)
            required_resources["num_gpus"] += ray_actor_options.get("num_gpus", 0)
            required_resources["memory"] += ray_actor_options.get("memory", 0)

        return required_resources

    async def build(
        self,
        application_id: str,
        artifact_id: str,
        version: Optional[str] = None,
        deployment_kwargs: Optional[Dict[str, Any]] = None,
        enable_gpu: bool = True,
    ) -> serve.Application:
        """
        Build a complete BioEngine application from a deployment artifact.

        This method orchestrates the entire application building process including:
        - Loading and validating the artifact manifest
        - Creating and configuring all deployments
        - Setting up the RTC proxy for communication
        - Validating deployment parameters and dependencies

        Args:
            application_id: Unique identifier for the application instance
            artifact_id: ID of the artifact containing deployment code (format: 'workspace/artifact_alias')
            version: Version of the artifact to load (defaults to latest)
            deployment_options: Resource allocation options for deployments
            deployment_kwargs: Initialization parameters for each deployment class

        Returns:
            Fully configured Ray Serve application ready for deployment

        Raises:
            ValueError: If application_id is empty, artifact_id format is invalid,
                       or deployment configuration is incorrect
            FileNotFoundError: If local artifact files are not found
            Exception: If manifest loading, deployment creation, or configuration fails
        """
        # Validate application_id and artifact_id
        if not application_id:
            raise ValueError("Application ID cannot be empty.")

        if not artifact_id or "/" not in artifact_id:
            raise ValueError(
                f"Invalid artifact ID format: {artifact_id}. "
                "Expected format is 'workspace/artifact_alias'."
            )

        self.logger.info(
            f"Building application '{application_id}' from artifact {artifact_id} (version: {version or 'latest'})"
        )

        # Load the artifact manifest
        manifest = await self._load_manifest(artifact_id, version)

        # Load all deployments defined in the manifest
        deployments = [
            await self._load_deployment(
                application_id=application_id,
                artifact_id=artifact_id,
                version=version,
                import_path=import_path,
                enable_gpu=enable_gpu,
                token=self._token,
            )
            for import_path in manifest["deployments"]
        ]
        deployment_kwargs = deployment_kwargs or {}

        # Calculate the total number of required resources
        rtc_proxy_deployment = BioEngineProxyDeployment
        required_resources = self._calculate_required_resources(
            deployments + [rtc_proxy_deployment]
        )

        # Get all schema_methods from the entry deployment class
        entry_deployment = deployments[0]
        class_name = entry_deployment.func_or_class.__name__
        method_schemas = []
        for method_name in dir(entry_deployment.func_or_class):
            method = getattr(entry_deployment.func_or_class, method_name)
            if callable(method) and hasattr(method, "__schema__"):
                method_schemas.append(method.__schema__)

        if not method_schemas:
            raise ValueError(
                f"No schema methods found in the entry deployment class: {class_name}."
            )

        # Get kwargs for the entry deployment
        entry_deployment_kwargs = deployment_kwargs.get(class_name, {}).copy()
        entry_init_params = self._get_init_param_info(entry_deployment)
        self._check_params(entry_init_params, entry_deployment_kwargs)

        # If multiple deployment classes are found, create a composition deployment
        if len(deployments) > 1:
            self.logger.debug(
                f"Creating a composition deployment with {len(deployments)} classes."
            )

            # Add the composition deployment class(es) to the entry deployment kwargs
            # TODO: use kwargs instead of args to pass deployment handles
            deployment_handle_params = [
                param_name
                for param_name, param_info in entry_init_params.items()
                if param_info["type"] == DeploymentHandle
            ]
            if len(deployment_handle_params) != len(deployments) - 1:
                raise ValueError(
                    f"Mismatch between number of deployment handle parameters "
                    f"({len(deployment_handle_params)}) and number of deployment "
                    f"classes defined in artifact manifest ({len(deployments) - 1})."
                )
            for handle_name, deployment in zip(
                deployment_handle_params, deployments[1:]
            ):
                class_name = deployment.func_or_class.__name__
                init_params = self._get_init_param_info(deployment)
                kwargs = deployment_kwargs.get(class_name, {})
                self._check_params(init_params, kwargs)
                entry_deployment_kwargs[handle_name] = deployment.bind(**kwargs)

        # Create the entry deployment handle
        entry_deployment_handle = entry_deployment.bind(**entry_deployment_kwargs)

        # Create the application
        app = rtc_proxy_deployment.bind(
            application_id=application_id,
            application_name=manifest["name"],
            application_description=manifest["description"],
            entry_deployment_handle=entry_deployment_handle,
            method_schemas=method_schemas,
            server_url=self.server.config.public_base_url,
            workspace=self.server.config.workspace,
            token=self._token,
            worker_client_id=self.server.config.client_id,
            authorized_users=manifest["authorized_users"],
            serve_http_url=self.serve_http_url,
        )

        # Create application metadata
        app.metadata = {
            "name": manifest["name"],
            "description": manifest["description"],
            "resources": required_resources,
            "authorized_users": manifest["authorized_users"],
            "available_methods": [
                method_schema["name"] for method_schema in method_schemas
            ],
        }

        self.logger.info(
            f"Successfully built application '{application_id}' with "
            f"available methods: {app.metadata['available_methods']}"
        )
        return app


if __name__ == "__main__":
    import asyncio

    from hypha_rpc import connect_to_server

    server_url = "https://hypha.aicell.io"
    token = os.environ["HYPHA_TOKEN"]

    base_dir = Path(__file__).parent.parent
    apps_cache_dir = base_dir / ".bioengine" / "apps"
    apps_data_dir = base_dir / "data"
    os.environ["BIOENGINE_WORKER_LOCAL_ARTIFACT_PATH"] = str(base_dir / "tests")

    async def test_app_builder():
        server = await connect_to_server({"server_url": server_url, "token": token})
        artifact_manager = await server.get_service("public/artifact-manager")

        app_builder = AppBuilder(
            token=token,
            apps_cache_dir=apps_cache_dir,
            apps_data_dir=apps_data_dir,
            log_file=None,
            debug=True,
        )
        app_builder.initialize(
            server, artifact_manager, serve_http_url="https://test-url"
        )

        app = await app_builder.build(
            application_id="test-application-1234",
            artifact_id="test-workspace/composition-app",
            version=None,
            deployment_options={
                "num_cpus": 1,
                "num_gpus": 0,
                "memory": 1024 * 1024 * 1024,  # 1 GB
            },
            deployment_kwargs={
                "CompositionDeployment": {"demo_input": "Hello World!"},
                "Deployment2": {"start_number": 10},
            },
        )

    asyncio.run(test_app_builder())
