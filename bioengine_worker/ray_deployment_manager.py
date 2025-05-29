import asyncio
import base64
import logging
import os
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import httpx
import numpy as np
import yaml

import ray
from bioengine_worker import __version__
from bioengine_worker.ray_autoscaler import RayAutoscaler
from bioengine_worker.utils import create_logger, format_time
from ray import serve


class RayDeploymentManager:
    """Manages Ray Serve deployments using Hypha artifact manager

    This class integrates with Hypha artifact manager to deploy
    artifacts from Hypha as Ray Serve deployments.
    """

    def __init__(
        self,
        service_id: str = "bioengine-apps",
        admin_users: Optional[List[str]] = None,
        deployment_cache_dir: str = "/tmp",
        startup_deployments: Optional[List[str]] = None,
        autoscaler: Optional[RayAutoscaler] = None,
        # Logger
        logger: Optional[logging.Logger] = None,
        log_file: Optional[str] = None,
        _debug: bool = False,
    ):
        """Initialize the Ray Deployment Manager

        Args:
            service_id: ID to use for the Hypha service exposing deployed models
            admin_users: List of user IDs or emails with admin permissions
            deployment_cache_dir: Caching directory used in Ray Serve deployments
            startup_deployments: List of artifact IDs to start on initialization
            autoscaler: Optional RayAutoscaler instance
            logger: Optional logger instance
            _debug: Enable debug logging
        """
        # Set up logging
        self.logger = logger or create_logger(
            name="RayDeploymentManager",
            level=logging.DEBUG if _debug else logging.INFO,
            log_file=log_file,
        )

        # Store parameters
        self._service_id = service_id
        self.autoscaler = autoscaler
        self.deployment_cache_dir = deployment_cache_dir
        self.admin_users = admin_users or []

        # Initialize state variables
        self.server = None
        self.artifact_manager = None
        self.service_info = None
        self.startup_deployments = startup_deployments or []
        self._deployed_artifacts = {}
        self._deployment_tasks = {}
        self._undeploying_artifacts = set()

    async def _get_full_artifact_id(self, artifact_id: str) -> str:
        """Convert artifact ID to a full artifact ID

        Args:
            artifact_id: The artifact ID to convert

        Returns:
            str: The converted full artifact ID
        """
        if "/" not in artifact_id:
            return f"{self.server.config.workspace}/{artifact_id}"
        return artifact_id

    async def _load_deployment_code(
        self,
        class_config: dict,
        artifact_id: str,
        version=None,
        timeout: int = 30,
        _local: bool = False,
    ) -> Any:
        """Load and execute deployment code from an artifact directly in memory

        Args:
            class_config: Configuration for the class to load
            artifact_id: ID of the artifact
            version: Optional version of the artifact
            timeout: Timeout in seconds for network requests (default: 30)

        Returns:
            Any: The loaded class or None if not found
        """
        try:
            if _local:
                # Load the file content from local path
                deployment = artifact_id.split("/")[1].replace("-", "_")
                local_deployments_dir = Path(__file__).parent.resolve() / "deployments"
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
            model = safe_globals[class_config["class_name"]]
            if not model:
                raise RuntimeError(
                    f"Error loading {class_config['class_name']} from {artifact_id}"
                )

            if "multiplexed" in class_config:
                # Add @serve.multiplexed decorator to specified class method
                method_name = class_config["multiplexed"]["method_name"]
                max_num_models_per_replica = class_config["multiplexed"][
                    "max_num_models_per_replica"
                ]

                orig_method = getattr(model, method_name)
                decorated_method = serve.multiplexed(
                    orig_method, max_num_models_per_replica=max_num_models_per_replica
                )
                setattr(model, method_name, decorated_method)

            self.logger.info(
                f"Loaded class '{class_config['class_name']}' from {artifact_id}"
            )
            return model

        except Exception as e:
            self.logger.error(f"Error loading deployment code for {artifact_id}: {e}")
            raise e

    async def _create_deployment_name(self, artifact_id: str) -> str:
        deployment_name = artifact_id.lower()
        for char in ["|", "/", "-", "."]:
            deployment_name = deployment_name.replace(char, "_")
        if not deployment_name.isidentifier():
            raise ValueError(
                f"Artifact ID '{artifact_id}' can not be automatically converted to a "
                f"valid deployment name ('{deployment_name}' is not a valid identifier)."
            )
        return deployment_name

    async def _check_permissions(
        self,
        context: Dict[str, Any],
        authorized_users: Union[List[str], str],
        resource_name: str,
    ) -> bool:
        """Check if the user in the context is authorized to access the deployment"""
        user = context["user"]
        if isinstance(authorized_users, str):
            authorized_users = [authorized_users]
        if (
            "*" not in authorized_users
            and user["id"] not in authorized_users
            and user.get("email", "no-email") not in authorized_users
        ):
            msg = f"User {user['id']} is not authorized to access {resource_name}"
            self.logger.warning(msg)
            raise PermissionError(msg)

    async def _update_services(self) -> None:
        """Update Hypha services based on currently deployed models"""
        try:
            # Ensure server connection
            if not self.server:
                raise RuntimeError("Hypha server connection not available")

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
                    await self._check_permissions(
                        context,
                        authorized_users,
                        resource_name=f"deployment '{deployment_name}' method '{method_name}'",
                    )
                    user_id = context["user"]["id"]

                    self.logger.info(
                        f"User '{user_id}' is calling deployment '{deployment_name}' with method '{method_name}'"
                    )
                    app_handle = serve.get_app_handle(name=deployment_name)

                    # Recursively put args and kwargs into ray object storage
                    args = [
                        ray.put(arg) if isinstance(arg, np.ndarray) else arg
                        for arg in args
                    ]
                    kwargs = {
                        k: ray.put(v) if isinstance(v, np.ndarray) else v
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
                class_config = deployment_info["deployment_class"]
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

            # Register all model functions as a single service
            service_info = await self.server.register_service(
                {
                    "id": self._service_id,
                    "name": "BioEngine Worker Deployments",
                    "type": "bioengine-apps",
                    "description": "Deployed Ray Serve models",
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

    async def initialize(self, server) -> None:
        """Initialize the deployment manager with a Hypha server connection

        Args:
            server: Hypha server connection
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

    async def initialize_deployments(self) -> None:
        """Deploy all startup deployments defined in the manager"""
        if not self.server:
            raise RuntimeError(
                "Hypha server connection not available. Call initialize() first."
            )
        if not self.artifact_manager:
            raise RuntimeError(
                "Artifact manager not initialized. Call initialize() first."
            )

        if self.startup_deployments:
            self.logger.info(
                f"Starting deployments for artifacts: {', '.join(self.startup_deployments)}"
            )
            context = {
                "user": {
                    "id": "startup",
                    "email": (self.admin_users[0] if self.admin_users else "anonymous"),
                }
            }
            deployment_tasks = [
                self.deploy_artifact(
                    artifact_id,
                    context=context,
                    _skip_update=True,
                )
                for artifact_id in self.startup_deployments
            ]
            await asyncio.gather(*deployment_tasks)

            # Update services after startup deployments
            await self._update_services()

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
        if not self.server:
            raise RuntimeError(
                "Hypha server connection not available. Call initialize() first."
            )

        # Ensure artifact manager is available
        if not self.artifact_manager:
            raise RuntimeError(
                "Artifact manager not initialized. Call initialize() first."
            )

        await self._check_permissions(
            context,
            self.admin_users,
            resource_name=f"creation of artifact '{artifact_id}'",
        )
        user_id = context["user"]["id"]

        # Find the manifest file to extract metadata
        manifest_file = None
        for file in files:
            if file["name"].lower() in ["manifest.yaml"]:
                manifest_file = file
                break

        if not manifest_file:
            raise ValueError(
                "No manifest file found in files list. Expected 'manifest.yaml'"
            )

        # Parse the manifest
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

        if artifact_id is None:
            deployment_manifest["created_by"] = user_id
        assert (
            deployment_manifest.get("type") == "application"
        ), f"type must be 'application', got '{deployment_manifest.get('type')}'"
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
                    type=deployment_manifest.get("type", "application"),
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

    async def deploy_artifact(
        self,
        artifact_id: str,
        mode: str = None,
        version: str = None,
        context: Optional[Dict[str, Any]] = None,
        _skip_update=False,
    ) -> str:
        """
        Deploy a single artifact to Ray Serve

        Args:
            artifact_id: ID of the artifact to deploy
            version: Optional version of the artifact
            context: Context for Hypha service
            _skip_update: Skip updating services after deployment

        Returns:
            str: Deployment name
        """
        # Verify client is connected to Hypha server
        if not self.server:
            raise RuntimeError(
                "Hypha server connection not available. Call initialize() first."
            )

        # Ensure artifact manager is available
        if not self.artifact_manager:
            raise RuntimeError(
                "Artifact manager not initialized. Call initialize() first."
            )

        # Verify Ray is initialized
        if not ray.is_initialized():
            raise RuntimeError("Ray cluster is not running. Call initialize() first.")

        # Check user permissions
        await self._check_permissions(
            context,
            self.admin_users,
            resource_name=f"deployment of artifact '{artifact_id}'",
        )
        user_id = context["user"]["id"]
        artifact_id = await self._get_full_artifact_id(artifact_id)

        # Check if the artifact is already deployed
        deployment_name = await self._create_deployment_name(artifact_id)
        serve_status = serve.status()
        if deployment_name not in serve_status.applications.keys():
            self.logger.info(f"User '{user_id}' is starting a new deployment for artifact '{artifact_id}'")
        else:
            application = serve_status.applications[deployment_name]
            if application.status.value == "DEPLOYING":
                self.logger.info(
                    f"Artifact '{artifact_id}' is currently being deployed. Skipping deployment."
                )
                return
            else:
                self.logger.info(
                    f"Updating existing deployment for artifact '{artifact_id}'"
                )

        # Read the manifest to get deployment configuration
        artifact = await self.artifact_manager.read(artifact_id, version=version)
        manifest = artifact["manifest"]

        # Get the deployment configuration
        deployment_name = await self._create_deployment_name(artifact_id)
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

        # Add cache path to deployment config environment
        runtime_env = ray_actor_options.setdefault("runtime_env", {})
        env_vars = runtime_env.setdefault("env_vars", {})
        env_vars["BIOENGINE_CACHE_PATH"] = str(self.deployment_cache_dir)

        # Load the deployment code
        model = await self._load_deployment_code(
            class_config,
            artifact_id,
            version=version,
        )

        # Create the Ray Serve deployment
        model_deployment = serve.deployment(**deployment_config)(model)

        # Bind the arguments to the deployment and return an Application
        kwargs = class_config.get("kwargs", {})
        app = model_deployment.bind(**kwargs)

        # Store the deployment information first so it's available to other tasks
        self._deployed_artifacts[artifact_id] = {
            "deployment_name": deployment_name,
            "deployment_class": class_config,
            "resources": {
                "num_cpus": deployment_config["ray_actor_options"]["num_cpus"],
                "num_gpus": deployment_config["ray_actor_options"]["num_gpus"],
            },
        }

        # Deploy the application in a separate task to avoid blocking
        async def deploy_task():
            self.logger.info(f"Started deployment process for artifact '{artifact_id}'")
            await asyncio.to_thread(
                serve.run, app, name=deployment_name, route_prefix=None
            )

            # Check if the deployment was successful
            serve_status = serve.status()
            if deployment_name in serve_status.applications.keys():
                self.logger.info(
                    f"Successfully completed deployment of artifact '{artifact_id}'"
                )
                # Update services if not skipped
                if not _skip_update:
                    await self._update_services()
            else:
                self.logger.error(
                    f"Deployment of artifact '{artifact_id}' failed. Deployment name '{deployment_name}' not found in serve status."
                )
                self._deployed_artifacts.pop(artifact_id, None)

        try:
            # Create and start the deployment task
            task = asyncio.create_task(deploy_task(), name=f"deploy-{artifact_id}")

            # Store the task with a strong reference to prevent garbage collection
            self._deployment_tasks[artifact_id] = task

            # Notify the autoscaler of the new deployment after a short delay
            if self.autoscaler:
                # Wait a moment for the deployment to start initializing
                await asyncio.sleep(1)
                # Notify the autoscaler
                await self.autoscaler.notify()

            # Wait for the deployment task to complete
            await task
        except asyncio.CancelledError as cancel_err:
            self.logger.info(f"Deployment of artifact '{artifact_id}' was cancelled")
            raise cancel_err
        except Exception as e:
            self.logger.error(
                f"Error during deployment of artifact '{artifact_id}': {e}"
            )
            raise e
        finally:
            # Clean up the deployment task reference
            self._deployment_tasks.pop(artifact_id, None)

    async def undeploy_artifact(
        self,
        artifact_id: str,
        context: Optional[Dict[str, Any]] = None,
        _skip_update=False,
    ) -> None:
        """Remove a deployment from Ray Serve

        Args:
            artifact_id: ID of the artifact to undeploy
            context: Context for Hypha service
            _skip_update: Skip updating services after undeployment
        """
        # Verify client is connected to Hypha server
        if not self.server:
            raise RuntimeError(
                "Hypha server connection not available. Call initialize() first."
            )

        # Ensure artifact manager is available
        if not self.artifact_manager:
            raise RuntimeError(
                "Artifact manager not initialized. Call initialize() first."
            )

        # Verify Ray is initialized
        if not ray.is_initialized():
            raise RuntimeError("Ray cluster is not running. Call initialize() first.")

        # Check user permissions
        await self._check_permissions(
            context,
            self.admin_users,
            resource_name=f"undeployment of artifact '{artifact_id}'",
        )
        user_id = context["user"]["id"]
        artifact_id = await self._get_full_artifact_id(artifact_id)

        # Check if artifact exists in deployed artifacts
        deployment_name = await self._create_deployment_name(artifact_id)
        serve_status = serve.status()
        if deployment_name not in serve_status.applications.keys():
            self.logger.warning(
                f"Artifact '{artifact_id}' is not deployed. Skipping undeployment."
            )
            return

        # Check if artifact is already being undeployed
        if artifact_id in self._undeploying_artifacts:
            self.logger.info(
                f"Artifact '{artifact_id}' is already being undeployed. Skipping duplicate request."
            )
            return

        try:
            self.logger.info(
                f"User '{user_id}' is undeploying artifact '{artifact_id}'..."
            )
            self._undeploying_artifacts.add(artifact_id)

            # Check if there's an ongoing deployment task for this artifact
            deployment_task = self._deployment_tasks.get(artifact_id)
            if deployment_task and not deployment_task.done():
                # Cancel the task
                self.logger.info(f"Cancelling ongoing deployment for '{artifact_id}'")
                deployment_task.cancel()

                # Wait for the task to finish or be cancelled
                timeout = 30
                try:
                    await asyncio.wait_for(deployment_task, timeout=timeout)
                except asyncio.TimeoutError:
                    self.logger.warning(
                        f"Deployment task for '{artifact_id}' did not finish in time ({timeout} seconds)"
                    )
                except Exception as e:
                    self.logger.error(
                        f"Error waiting for deployment task cancellation: {e}"
                    )

            # Delete the deployment asynchronously
            try:
                await asyncio.to_thread(serve.delete, deployment_name)
            except Exception as delete_err:
                self.logger.error(
                    f"Error deleting deployment {deployment_name}: {delete_err}"
                )
                raise delete_err

            # Remove the artifact from deployed artifacts
            self._deployed_artifacts.pop(artifact_id, None)

            if not _skip_update:
                await self._update_services()

            self.logger.info(f"Successfully undeployed {artifact_id}")

        except Exception as e:
            self.logger.error(f"Error undeploying '{artifact_id}': {e}")
            raise e
        finally:
            # Clean up the undeploying artifacts reference
            self._undeploying_artifacts.discard(artifact_id)

    async def get_status(self) -> Dict[str, Any]:
        """Get a dictionary of currently deployed models"""
        if not ray.is_initialized():
            self.logger.error("Can not get deployments - Ray cluster is not running")
            raise RuntimeError("Ray cluster is not running")

        output = {}
        if self.service_info:
            output["service_id"] = self.service_info.id
        else:
            output["service_id"] = None

        # Get status of actively running deployments
        serve_status = serve.status()
        self.logger.debug(
            f"Current deployments: {list(serve_status.applications.keys())}"
        )

        if not serve_status.applications:
            output["note"] = "Currently no artifacts are deployed."
            return output

        for artifact_id in list(self._deployed_artifacts.keys()):
            deployment_name = self._deployed_artifacts[artifact_id]["deployment_name"]
            if deployment_name not in serve_status.applications:
                # ? Clean up if deployment is not found
                self.logger.warning(
                    f"Deployment '{deployment_name}' for artifact '{artifact_id}' not found in Ray Serve status."
                )
                # del self._deployed_artifacts[artifact_id]
                continue

            application = serve_status.applications[deployment_name]
            formatted_time = format_time(application.last_deployed_time_s)
            if len(application.deployments) > 1:
                raise NotImplementedError

            class_config = self._deployed_artifacts[artifact_id]["deployment_class"]
            class_methods = class_config.get("exposed_methods", {})
            class_name = class_config["class_name"]
            deployment = application.deployments.get(class_name)
            resources = self._deployed_artifacts[artifact_id]["resources"]
            output[artifact_id] = {
                "deployment_name": deployment_name,
                "available_methods": list(class_methods.keys()),
                "start_time_s": application.last_deployed_time_s,
                "start_time": formatted_time["start_time"],
                "uptime": formatted_time["uptime"],
                "status": application.status.value,
                "replica_states": deployment.replica_states if deployment else None,
                "resources": resources,
            }

        return output

    async def deploy_all_artifacts(
        self,
        deployment_collection_id: str = "bioengine-apps",
        context: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Deploy all artifacts in the deployment collection to Ray Serve

        Args:
            deployment_collection_id: Artifact collection ID for deployments
            context: Context for Hypha service

        Returns:
            list: List of artifact IDs that were successfully deployed
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
        for artifact in artifacts:
            try:
                artifact_id = artifact["id"]
                await self.deploy_artifact(artifact_id, _skip_update=True)
            except Exception as e:
                self.logger.error(f"Failed to deploy {artifact_id}: {e}")
                raise e

        # Update services after all deployments
        await self._update_services()

    async def cleanup_deployments(
        self, context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Cleanup Ray Serve deployments"""
        self.logger.info("Cleaning up all deployments...")

        # Ensure Ray is initialized
        if not ray.is_initialized():
            raise RuntimeError("Ray cluster is not running")

        artifact_ids = list(self._deployed_artifacts.keys())
        failed_attempts = 0
        for artifact_id in artifact_ids:
            try:
                await self.undeploy_artifact(artifact_id)
            except Exception as e:
                failed_attempts += 1
                self.logger.error(f"Failed to undeploy {artifact_id}: {e}")

        # Cancel any remaining deployment tasks
        pending_tasks = []
        for artifact_id, task in list(self._deployment_tasks.items()):
            if not task.done():
                pending_tasks.append(task)
                task.cancel()

        # Wait for all tasks to complete or be cancelled
        if pending_tasks:
            self.logger.warning(
                f"Cancelling {len(pending_tasks)} remaining deployment tasks"
            )
            try:
                await asyncio.wait(pending_tasks, timeout=5)
            except Exception as e:
                self.logger.error(f"Error waiting for tasks to cancel: {e}")

        if failed_attempts != 0:
            self.logger.warning(
                f"Failed to clean up all deployments, {failed_attempts} remaining."
            )


async def create_demo_artifact(deployment_manager, artifact_id=None):
    """Helper function to create a demo artifact from demo deployment files

    Args:
        deployment_manager: RayDeploymentManager instance (must be initialized)
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
    created_artifact_id = await deployment_manager.create_artifact(
        files, artifact_id=artifact_id
    )
    return created_artifact_id


if __name__ == "__main__":
    """Test the RayDeploymentManager functionality with a real Ray cluster and model deployment."""

    from hypha_rpc import connect_to_server, login

    from bioengine_worker.ray_cluster_manager import RayClusterManager

    print("\n===== Testing RayDeploymentManager =====\n")

    # Create and start the autoscaler with shorter thresholds for quicker testing
    cluster_manager = RayClusterManager(
        head_num_cpus=4,
        head_num_gpus=1,
        ray_temp_dir=f"/tmp/ray/{os.environ['USER']}",
        image=str(
            Path(__file__).parent.parent
            / f"apptainer_images/bioengine-worker_{__version__}.sif"
        ),
        worker_data_dir=str(Path(__file__).parent.parent / "data"),
        slurm_log_dir=str(Path(__file__).parent.parent / "logs"),
        _debug=True,
    )
    cluster_manager.start_cluster(force_clean_up=True)

    if cluster_manager.ray_cluster_config["mode"] == "slurm":
        autoscaler = RayAutoscaler(
            cluster_manager=cluster_manager,
            # Use shorter times for faster testing
            default_time_limit="00:10:00",
            max_workers=1,
            metrics_interval_seconds=10,
            _debug=True,
        )
    else:
        autoscaler = None

    async def test_create_artifact(
        deployment_manager=None, server_url="https://hypha.aicell.io"
    ):
        """Test the create_artifact function with demo deployment files

        Args:
            deployment_manager: Optional existing deployment manager (must be initialized)
            server_url: Server URL if creating new connection

        Returns:
            str: The artifact ID of the last created artifact (for use in other tests)
        """
        print("\n===== Testing create_artifact function =====\n")

        # Use existing deployment manager or create new one
        if deployment_manager is None:
            try:
                # Create deployment manager (no Ray cluster needed for artifact creation)
                deployment_manager = RayDeploymentManager(_debug=True)

                # Connect to Hypha server using token from environment
                token = os.environ.get("HYPHA_TOKEN") or await login(
                    {"server_url": server_url}
                )
                server = await connect_to_server(
                    {"server_url": server_url, "token": token}
                )

                # Initialize deployment manager
                await deployment_manager.initialize(server)

            except Exception as e:
                print(f"❌ Failed to initialize deployment manager: {e}")
                raise e

        try:
            # Test creating artifact without specifying artifact_id (should use ID from manifest)
            print("Testing create_artifact without specifying artifact_id...")
            created_artifact_id = await create_demo_artifact(deployment_manager)
            print(f"Successfully created artifact: {created_artifact_id}")

            # Test updating the same artifact
            print(f"\nTesting update of existing artifact: {created_artifact_id}")
            updated_artifact_id = await create_demo_artifact(
                deployment_manager, artifact_id=created_artifact_id
            )
            print(f"Successfully updated artifact: {updated_artifact_id}")

            # Test creating artifact with custom artifact_id
            print("\nTesting create_artifact with custom artifact_id...")
            custom_artifact_id = "test-demo-deployment"
            custom_created_id = await create_demo_artifact(
                deployment_manager, artifact_id=custom_artifact_id
            )
            print(f"Successfully created custom artifact: {custom_created_id}")

            print("\n✅ All create_artifact tests passed!")

            # Return the last created artifact ID for use in other tests
            return custom_created_id

        except Exception as e:
            print(f"❌ create_artifact test failed: {e}")
            raise e

    async def test_deployment_manager(
        server_url="https://hypha.aicell.io", keep_running=False
    ):
        try:
            if autoscaler:
                # Start autoscaler
                await autoscaler.start()

            # Create deployment manager
            deployment_manager = RayDeploymentManager(
                autoscaler=autoscaler, _debug=True
            )

            # Connect to Hypha server using token from environment
            token = os.environ.get("HYPHA_TOKEN") or await login(
                {"server_url": server_url}
            )
            server = await connect_to_server({"server_url": server_url, "token": token})

            # Initialize deployment manager
            await deployment_manager.initialize(server)

            # Test create_artifact function
            created_artifact_id = await test_create_artifact(deployment_manager)

            # Test deploying the newly created artifact
            print(
                f"\n--- Testing deployment of created artifact: {created_artifact_id} ---"
            )
            await deployment_manager.deploy_artifact(created_artifact_id)

            # Test the deployed artifact
            deployment_status = await deployment_manager.get_status()
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
            await deployment_manager.deploy_artifact(artifact_id)

            deployment_status = await deployment_manager.get_status()
            assert artifact_id in deployment_status

            # Test registered Hypha service
            deployment_service_id = deployment_status["service_id"]
            deployment_service = await server.get_service(deployment_service_id)

            # Call the deployed model
            deployment_name = deployment_status[artifact_id]["deployment_name"]
            response = await deployment_service[deployment_name]["ping"]()
            deployment_manager.logger.info(f"Response from deployed model: {response}")

            response = await deployment_service[deployment_name]["train"]()
            deployment_manager.logger.info(f"Response from deployed model: {response}")

            # Keep server running if requested
            if keep_running:
                print("Server running. Press Ctrl+C to stop.")
                await server.serve()

            # Undeploy the test artifact
            await deployment_manager.undeploy_artifact(artifact_id)

            # Deploy again
            await deployment_manager.deploy_artifact(artifact_id)

            # Clean up deployments
            await deployment_manager.cleanup_deployments()

        except Exception as e:
            print(f"An error occurred: {e}")
            raise e
        finally:
            if autoscaler:
                await autoscaler.shutdown_cluster()
            else:
                cluster_manager.shutdown_cluster()

    # Run the test
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test_create_artifact":
        # Run only the create_artifact test (no Ray cluster needed)
        asyncio.run(test_create_artifact())
    else:
        # Run the full deployment manager test
        asyncio.run(test_deployment_manager(keep_running=True))
