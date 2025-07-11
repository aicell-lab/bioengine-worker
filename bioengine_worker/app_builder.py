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
from bioengine_worker.utils import create_logger


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
        Initialize the AppBuilder with the given app instance.

        :param app: The application instance to be managed.
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

    def _add_init_wrapper(self, deployment_class: serve.Deployment) -> serve.Deployment:
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

    async def _update_env_vars(
        self, deployment_class: serve.Deployment, token: str
    ) -> serve.Deployment:
        """
        Add BioEngine and Hypha specific environment variables to the deployment
        """
        ray_actor_options = deployment_class.ray_actor_options
        runtime_env = ray_actor_options.setdefault("runtime_env", {})
        env_vars = runtime_env.setdefault("env_vars", {})

        # Set standard directories to ensure it only uses the specified workdir
        env_vars["BIOENGINE_WORKDIR"] = str(self.apps_cache_dir)
        env_vars["HOME"] = str(self.apps_cache_dir)
        env_vars["TMPDIR"] = str(self.apps_cache_dir / "tmp")

        # Pass the data directory to the deployment
        env_vars["BIOENGINE_DATA_DIR"] = str(self.apps_data_dir)

        env_vars["HYPHA_SERVER_URL"] = self.server.config.public_base_url
        env_vars["HYPHA_WORKSPACE"] = self.server.config.workspace
        # env_vars["HYPHA_CLIENT_ID"] = self.server.config.client_id
        # env_vars["HYPHA_SERVICE_ID"] = self.service_id
        env_vars["HYPHA_TOKEN"] = token

        return deployment_class.options(ray_actor_options=ray_actor_options)

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

            # Wrap the class `__init__` method to ensure it uses the workdir
            deployment_class = self._add_init_wrapper(deployment_class)

            self.logger.debug(
                f"Loaded class '{class_name}' from artifact '{artifact_id}'."
            )

            deployment_class = await self._update_env_vars(deployment_class, token)

            return deployment_class
        except Exception as e:
            self.logger.error(
                f"Error creating deployment class from {artifact_id}: {e}"
            )
            raise e

    def _calculate_required_resources(self, deployments: List[serve.Deployment]) -> Dict[str, int]:
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
        version: str = None,
        deployment_options: Optional[Dict[str, int]] = None,
        deployment_kwargs: Optional[Dict[str, Any]] = None,
    ) -> serve.Application:
        """
        Build the application.

        :return: The built application instance.
        """
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

        # Get kwargs for the entry deployment
        entry_deployment = deployments[0]
        entry_init_params = self._get_init_param_info(entry_deployment)
        deployment_kwargs = deployment_kwargs or {}
        entry_deployment_kwargs = {
            p_name: value
            for p_name, value in deployment_kwargs.items()
            if p_name in entry_init_params
        }

        # If multiple deployment classes are found, create a composition deployment
        if len(deployments) > 1:
            self.logger.debug(
                "Creating a composition deployment with multiple classes."
            )

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
                init_params = self._get_init_param_info(deployment)
                entry_deployment_kwargs[handle_name] = deployment.bind(
                    **{
                        param_name: value
                        for param_name, value in deployment_kwargs.items()
                        if param_name in init_params
                    }
                )

        # Create the entry deployment handle
        entry_deployment_handle = entry_deployment.bind(**entry_deployment_kwargs)

        # Get all schema_methods from the entry deployment class
        method_schemas = []
        for method_name in dir(entry_deployment.func_or_class):
            method = getattr(entry_deployment.func_or_class, method_name)
            if callable(method) and hasattr(method, "__schema__"):
                method_schemas.append(method.__schema__)

        # Create the application
        rtc_proxy_deployment = RtcProxyDeployment
        app = rtc_proxy_deployment.bind(
            application_id=application_id,
            application_name=manifest["name"],
            application_description=manifest["description"],
            entry_deployment=entry_deployment_handle,
            method_schemas=method_schemas,
            server_url=self.server.config.public_base_url,
            workspace=self.server.config.workspace,
            token=self._token,
            authorized_users=manifest["authorized_users"],
        )

        # Calculate the total number of required resources
        required_resources = self._calculate_required_resources(deployments + [rtc_proxy_deployment])

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
