import inspect
import logging
import os
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import yaml
from hypha_rpc.rpc import RemoteService
from hypha_rpc.utils import ObjectProxy
from ray import serve
from ray.serve.handle import DeploymentHandle

from bioengine_worker.rtc_proxy_deployment import RtcProxyDeployment
from bioengine_worker.utils import create_logger, update_requirements


class AppBuilder:
    """
    A class to build and manage the application.
    """

    def __init__(
        self,
        token: str,
        apps_cache_dir: Path,
        apps_data_dir: Path,
        log_file: str = None,
        debug: bool = False,
    ):
        """
        Initialize the AppBuilder with configuration parameters.

        Args:
            token: Authentication token for service access
            apps_cache_dir: Cache directory for deployment artifacts
            apps_data_dir: Data directory accessible to deployments
            log_file: Optional log file path for output
            debug: Enable debug logging
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
        self.server = None
        self.artifact_manager = None

    def initialize(self, server: RemoteService, artifact_manager: ObjectProxy) -> None:
        # Store server connection and artifact manager
        self.server = server
        self.artifact_manager = artifact_manager

    async def _load_manifest(self, artifact_id: str, version: str = None) -> dict:
        """
        Load the artifact manifest from the artifact manager.

        Args:
            artifact_id (str): The ID of the artifact to load.
            version (str, optional): The version of the artifact to load. Defaults to None.

        Returns:
            dict: The loaded manifest dictionary.

        Raises:
            ValueError: If the manifest is invalid or missing required fields.
            Exception: If there is an error loading the manifest.
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

    def _update_init(self, deployment_class: serve.Deployment) -> serve.Deployment:
        orig_init = getattr(deployment_class, "__init__")

        @wraps(orig_init)
        def wrapped_init(self, *args, **kwargs):
            import os
            from pathlib import Path

            # Ensure the workdir is set to the BIOENGINE_WORKDIR environment variable
            workdir = Path(os.environ["BIOENGINE_WORKDIR"])
            workdir.mkdir(parents=True, exist_ok=True)
            os.chdir(workdir)

            # Initialize deployment states
            self._deployment_initialized = False
            self._deployment_tested = False

            # Call the original __init__ method
            return orig_init(self, *args, **kwargs)

        setattr(deployment_class, "__init__", wrapped_init)
        return deployment_class

    def _update_async_init(
        self, deployment_class: serve.Deployment
    ) -> serve.Deployment:
        orig_async_init = getattr(deployment_class, "async_init", lambda self: None)

        @wraps(orig_async_init)
        async def wrapped_async_init(self):
            # Check if the original health check method is async
            if inspect.iscoroutinefunction(orig_async_init):
                await orig_async_init(self)
            else:
                # If it's a regular function, call it directly
                orig_async_init(self)

            self._deployment_initialized = True

        setattr(deployment_class, "async_init", wrapped_async_init)
        return deployment_class

    def _update_test_deployment(
        self, deployment_class: serve.Deployment
    ) -> serve.Deployment:
        orig_test_deployment = getattr(
            deployment_class, "test_deployment", lambda self: True
        )

        @wraps(orig_test_deployment)
        async def wrapped_test_deployment(self):
            # Check if the original health check method is async
            if inspect.iscoroutinefunction(orig_test_deployment):
                test_result = await orig_test_deployment(self)
            else:
                # If it's a regular function, call it directly
                test_result = orig_test_deployment(self)

            if test_result is not True:
                raise RuntimeError(
                    f"Deployment test failed for {deployment_class.func_or_class.__name__}"
                )

            self._deployment_tested = True

        setattr(deployment_class, "test_deployment", wrapped_test_deployment)
        return deployment_class

    def _update_health_check(
        self, deployment_class: serve.Deployment
    ) -> serve.Deployment:
        """
        Add a health check method to the deployment class.
        This method is called by Ray Serve during the actor initialization and keeps the
        deployment in stage "DEPLOYING" until the health check passes.
        """
        orig_health_check = getattr(deployment_class, "health_check", lambda self: None)

        @wraps(orig_health_check)
        async def health_check(self):
            if not self._deployment_initialized:
                await self.async_init()
            if not self._deployment_tested:
                await self.test_deployment()

            # Check if the original health check method is async
            if inspect.iscoroutinefunction(orig_health_check):
                return await orig_health_check(self)
            else:
                # If it's a regular function, call it directly
                return orig_health_check(self)

        # Add the updated health check method to the deployment class
        setattr(deployment_class, "health_check", health_check)
        return deployment_class

    def _get_init_param_info(self, deployment_class: serve.Deployment) -> dict:
        sig = inspect.signature(deployment_class.func_or_class.__init__)
        params = {}
        for name, param in sig.parameters.items():
            if name == "self":
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
        return params

    def _check_params(self, init_params: dict, kwargs: dict) -> None:
        """
        Check if the provided kwargs match the expected init parameters.
        Raises ValueError if there are unexpected parameters, missing required parameters,
        or type mismatches.
        """
        # Check if all provided parameters are expected
        for key in kwargs:
            if key not in init_params:
                raise ValueError(
                    f"Unexpected parameter '{key}' provided. "
                    f"Expected one of {list(init_params.keys())}."
                )
            expected_type = init_params[key]["type"]
            if expected_type and not isinstance(kwargs[key], expected_type):
                raise ValueError(
                    f"Parameter '{key}' must be of type {expected_type.__name__}, "
                    f"but got {type(kwargs[key]).__name__}."
                )

        # Check if all required parameters are provided
        for key, param_info in init_params.items():
            if param_info["type"] == DeploymentHandle:
                # DeploymentHandle parameters are handled separately
                continue
            if param_info["default"] is None and key not in kwargs:
                raise ValueError(
                    f"Missing required parameter '{key}' of type {param_info['type'].__name__}."
                )

    async def _update_actor_options(
        self, deployment_class: serve.Deployment, token: str
    ) -> serve.Deployment:
        """
        Add any missing BioEngine requirements to the deployment class.
        Add BioEngine and Hypha specific environment variables to the deployment
        """
        ray_actor_options = deployment_class.ray_actor_options
        runtime_env = ray_actor_options.setdefault("runtime_env", {})
        pip_requirements = runtime_env.setdefault("pip", [])
        env_vars = runtime_env.setdefault("env_vars", {})

        # Update pip requirements with BioEngine requirements
        pip_requirements = update_requirements(pip_requirements)

        # Set standard directories to ensure it only uses the specified workdir
        env_vars["BIOENGINE_WORKDIR"] = str(self.apps_cache_dir)
        env_vars["HOME"] = str(self.apps_cache_dir)
        env_vars["TMPDIR"] = str(self.apps_cache_dir / "tmp")

        # Pass the data directory to the deployment
        env_vars["BIOENGINE_DATA_DIR"] = str(self.apps_data_dir)

        env_vars["HYPHA_SERVER_URL"] = self.server.config.public_base_url
        env_vars["HYPHA_WORKSPACE"] = self.server.config.workspace
        env_vars["HYPHA_TOKEN"] = token

        return deployment_class.options(ray_actor_options=ray_actor_options)

    async def _load_deployment(
        self,
        artifact_id: str,
        version: str,
        import_path: str,
        deployment_options: Dict[str, int],
        token: str,
    ) -> serve.Deployment:
        """
        Load and execute deployment code from an artifact directly in memory.

        Downloads and executes Python code from an artifact to create deployable classes.
        Supports both remote artifact loading and local file loading for development.

        Args:
            class_config: Configuration for the class to load including class name and file path
            artifact_id: ID of the artifact containing the deployment code
            version: Optional version of the artifact to load

        Returns:
            Any: The loaded class ready for Ray Serve deployment

        Raises:
            FileNotFoundError: If local deployment file is not found
            ValueError: If class name is not found in the code
            RuntimeError: If class loading fails
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
            self.logger.debug(
                f"Loading deployment code from local path: {python_file} in artifact {artifact_id}"
            )
            artifact_folder = artifact_id.split("/")[1].replace("-", "_")
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
        safe_globals = {}
        if deployment_options["num_cpus"] is not None:
            safe_globals["NUM_CPUS"] = deployment_options["num_cpus"]
        if deployment_options["num_gpus"] is not None:
            safe_globals["NUM_GPUS"] = deployment_options["num_gpus"]
        if deployment_options["memory"] is not None:
            safe_globals["MEMORY"] = deployment_options["memory"]

        try:
            # Execute the code in a sandboxed environment
            exec(code_content, safe_globals)
            if class_name not in safe_globals:
                raise ValueError(f"{class_name} not found in {artifact_id}")
            deployment_class = safe_globals[class_name]
            if not deployment_class:
                raise RuntimeError(f"Error loading {class_name} from {artifact_id}")

            # Update the deployment class methods
            deployment_class = self._update_init(deployment_class)
            deployment_class = self._update_async_init(deployment_class)
            deployment_class = self._update_test_deployment(deployment_class)
            deployment_class = self._update_health_check(deployment_class)

            # Update environment variables and requirements
            deployment_class = await self._update_actor_options(deployment_class, token)

            self.logger.debug(
                f"Loaded class '{class_name}' from artifact '{artifact_id}'."
            )

            return deployment_class
        except Exception as e:
            self.logger.error(
                f"Error creating deployment class from {artifact_id}: {e}"
            )
            raise e

    def _calculate_required_resources(
        self, deployments: List[serve.Deployment]
    ) -> Dict[str, int]:
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
        deployment_options: Optional[Dict[str, int]] = None,
        deployment_kwargs: Optional[Dict[str, Any]] = None,
    ) -> serve.Application:
        """
        Build the application.

        :return: The built application instance.
        """
        # Validate application_id and artifact_id
        if not application_id:
            raise ValueError("Application ID cannot be empty.")

        if not artifact_id or "/" not in artifact_id:
            raise ValueError(
                f"Invalid artifact ID format: {artifact_id}. "
                "Expected format is 'workspace/artifact_alias'."
            )

        # Load the artifact manifest
        manifest = await self._load_manifest(artifact_id, version)

        # Load all deployments defined in the manifest
        deployments = [
            await self._load_deployment(
                artifact_id=artifact_id,
                version=version,
                import_path=import_path,
                deployment_options=deployment_options or {},
                token=self._token,
            )
            for import_path in manifest["deployments"]
        ]
        deployment_kwargs = deployment_kwargs or {}

        # Calculate the total number of required resources
        rtc_proxy_deployment = RtcProxyDeployment
        required_resources = self._calculate_required_resources(
            deployments + [rtc_proxy_deployment]
        )

        # Get all schema_methods from the entry deployment class
        entry_deployment = deployments[0]
        method_schemas = []
        for method_name in dir(entry_deployment.func_or_class):
            method = getattr(entry_deployment.func_or_class, method_name)
            if callable(method) and hasattr(method, "__schema__"):
                method_schemas.append(method.__schema__)

        if not method_schemas:
            raise ValueError(
                f"No schema methods found in the entry deployment class: "
                f"{entry_deployment.func_or_class.__name__}."
            )

        # Get kwargs for the entry deployment
        class_name = entry_deployment.func_or_class.__name__
        entry_deployment_kwargs = deployment_kwargs.get(class_name, {}).copy()
        entry_init_params = self._get_init_param_info(entry_deployment)
        self._check_params(entry_init_params, entry_deployment_kwargs)

        # If multiple deployment classes are found, create a composition deployment
        if len(deployments) > 1:
            self.logger.debug(
                "Creating a composition deployment with multiple classes."
            )

            # Add the composition deployment class(es) to the entry deployment kwargs
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
            authorized_users=manifest["authorized_users"],
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
        app_builder.initialize(server, artifact_manager)

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
