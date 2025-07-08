import asyncio
import base64
import inspect
import logging
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import yaml
from hypha_rpc.rpc import RemoteService
from ray import serve
from ray.serve.handle import DeploymentHandle

from bioengine_worker.rtc_proxy_deployment import RtcProxyDeployment
from bioengine_worker.utils import check_permissions, create_logger


class AppBuilder:
    """
    A class to build and manage the application.
    """

    def __init__(
        self,
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

        self.apps_cache_dir = apps_cache_dir
        self.apps_data_dir = apps_data_dir
        self.server = None
        self.artifact_manager = None

    async def initialize(self, server: RemoteService) -> None:
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

    def _validate_manifest(self, manifest: dict) -> None:
        """
        Validate the manifest structure and required fields.

        Args:
            manifest (dict): The manifest dictionary to validate.

        Raises:
            ValueError: If the manifest is missing required fields or has an invalid structure.
        """
        if manifest is None:
            raise ValueError("Manifest cannot be None")

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

    # TODO
    def _create_deployment_metadata(self, manifest: dict, artifact_id: str) -> dict:
        pass
        # Update the deployment tracking information
        # deployment_info = self._deployed_artifacts[artifact_id]
        # artifact_emoji = manifest["id_emoji"] + " " if manifest.get("id_emoji") else ""
        # artifact_name = manifest.get("name", manifest["id"])
        # deployment_info["display_name"] = artifact_emoji + artifact_name
        # deployment_info["description"] = manifest.get("description", "")
        # deployment_info["deployment_name"] = deployment_name
        # deployment_info["class_config"] = class_config
        # deployment_info["resources"] = {
        #     "num_cpus": deployment_config["ray_actor_options"]["num_cpus"],
        #     "num_gpus": deployment_config["ray_actor_options"]["num_gpus"],
        #     "memory": deployment_config["ray_actor_options"].get("memory"),
        # }
        # deployment_info["async_init"] = hasattr(deployment_class, "_async_init")
        # deployment_info["test_deployment"] = hasattr(
        #     deployment_class, "_test_deployment"
        # )

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
        self, deployment_class: serve.Deployment
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

        env_vars["HYPHA_SERVER_URL"] = self.server.config.workspace
        env_vars["HYPHA_WORKSPACE"] = self.server.config.workspace
        # env_vars["HYPHA_CLIENT_ID"] = self.server.config.client_id
        # env_vars["HYPHA_SERVICE_ID"] = self.service_id
        env_vars["HYPHA_TOKEN"] = await self.server.generate_token()

        return deployment_class.options(
            route_prefix=None,
            ray_actor_options=ray_actor_options,
        )

    async def _load_deployment(
        self,
        artifact_id: str,
        version: str,
        import_path: str,
        num_cpus: int,
        num_gpus: int,
        memory: int,
        object_store_memory: int,
        _local: bool = False,  # Used for development
    ) -> serve.Deployment:
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
            try:
                python_file, class_name = import_path.split(":")
            except ValueError:
                raise ValueError(
                    f"Invalid import path format: {import_path}. "
                    "Expected format is 'python_file:class_name'."
                )
            if _local:
                # Load the file content from local path
                deployment = artifact_id.split("/")[1].replace("-", "_")
                local_deployments_dir = (
                    Path(__file__).parent.parent.resolve() / "deployments"
                )
                local_path = local_deployments_dir / deployment / python_file
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

            # Create a restricted globals dictionary for sandboxed execution
            safe_globals = {}
            if num_cpus is not None:
                safe_globals["NUM_CPUS"] = num_cpus
            if num_gpus is not None:
                safe_globals["NUM_GPUS"] = num_gpus
            if memory is not None:
                safe_globals["MEMORY"] = memory
            if object_store_memory is not None:
                safe_globals["OBJECT_STORE_MEMORY"] = object_store_memory
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

            deployment_class = await self._update_env_vars(deployment_class)

            return deployment_class

        except Exception as e:
            self.logger.error(f"Error loading deployment code for {artifact_id}: {e}")
            raise e

    # TODO
    def _check_resources(self, deployments):
        pass

        # # Check if the required resources are available
        # insufficient_resources = True
        # while not self.ray_cluster.status["nodes"]:
        #     # Wait for Ray cluster to be ready
        #     await asyncio.sleep(1)
        # for node_resource in self.ray_cluster.status["nodes"].values():
        #     if (
        #         node_resource["available_cpu"] >= ray_actor_options["num_cpus"]
        #         and node_resource["available_gpu"] >= ray_actor_options["num_gpus"]
        #         and node_resource["available_memory"] >= ray_actor_options["memory"]
        #     ):
        #         insufficient_resources = False

        # if self.ray_cluster.mode == "slurm" and insufficient_resources:
        #     # Check if additional SLURM workers can be created that meet the resource requirements
        #     # TODO: Remove resource check when SLURM workers can adjust resources dynamically
        #     num_worker_jobs = await self.ray_cluster.slurm_workers.get_num_worker_jobs()
        #     default_num_cpus = self.ray_cluster.slurm_workers.default_num_cpus
        #     default_num_gpus = self.ray_cluster.slurm_workers.default_num_gpus
        #     default_memory = (
        #         self.ray_cluster.slurm_workers.default_mem_per_cpu * default_num_cpus
        #     )
        #     if (
        #         num_worker_jobs < self.ray_cluster.slurm_workers.max_workers
        #         and default_num_cpus >= ray_actor_options["num_cpus"]
        #         and default_num_gpus >= ray_actor_options["num_gpus"]
        #         and default_memory >= ray_actor_options["memory"]
        #     ):
        #         insufficient_resources = False

        # if insufficient_resources:
        #     if self.ray_cluster.mode != "external-cluster":
        #         raise ValueError(
        #             f"Insufficient resources for deployment '{deployment_name}'. "
        #             f"Requested: {ray_actor_options}"
        #         )
        #     else:
        #         self.logger.warning(
        #             f"Currently insufficient resources for deployment '{deployment_name}'. "
        #             "Assuming Ray autoscaling is available. "
        #             f"Requested resources: {ray_actor_options}"
        #         )

    async def build(
        self,
        artifact_id: str,
        version: str = None,
        num_cpus: int = None,
        num_gpus: int = None,
        memory: int = None,
        object_store_memory: int = None,
        **deployment_kwargs: Dict[str, Any],
    ) -> serve.Application:
        """
        Build the application.

        :return: The built application instance.
        """
        # Load the artifact manifest
        artifact = await self.artifact_manager.read(artifact_id, version=version)
        manifest = artifact.get("manifest")
        self._validate_manifest(manifest)

        # Create deployment metadata
        deployment_metadata = self._create_deployment_metadata(
            manifest=manifest, artifact_id=artifact_id
        )

        # Load all deployments defined in the manifest
        deployments = [
            await self._load_deployment(
                artifact_id=artifact_id,
                version=version,
                import_path=import_path,
                num_cpus=num_cpus,
                num_gpus=num_gpus,
                memory=memory,
                object_store_memory=object_store_memory,
            )
            for import_path in manifest["deployments"]
        ]

        # Check if the total number of required resources is available
        self._check_resources(deployments)

        # Get kwargs for the entry deployment
        entry_deployment = deployments[0]
        entry_init_params = self._get_init_param_info(entry_deployment)
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

        # Get public methods (no underscore prefix) of entry deployment class
        exposed_methods = [
            method_name
            for method_name in dir(entry_deployment.func_or_class)
            if not method_name.startswith("_")
            and callable(getattr(entry_deployment.func_or_class, method_name))
        ]

        # Create the application
        rtc_proxy_deployment = await self._update_env_vars(RtcProxyDeployment)
        app = rtc_proxy_deployment.bind(
            application_id=artifact_id,
            application_name=artifact["manifest"]["name"],
            entry_deployment=entry_deployment_handle,
            exposed_methods=exposed_methods,
            authorized_users=artifact["manifest"]["authorized_users"],
        )

        # TODO: create application info

        return app

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

        check_permissions(
            context=context,
            authorized_users=self.admin_users,
            resource_name=f"create or modify an artifact",
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
