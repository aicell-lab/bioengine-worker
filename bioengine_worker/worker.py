import asyncio
import inspect
import logging
import os
import textwrap
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import cloudpickle
import ray
from hypha_rpc import connect_to_server
from hypha_rpc.sync import login

from bioengine_worker import __version__
from bioengine_worker.dataset_manager import DatasetManager
from bioengine_worker.ray_autoscaler import RayAutoscaler
from bioengine_worker.ray_cluster_manager import RayClusterManager
from bioengine_worker.ray_deployment_manager import RayDeploymentManager
from bioengine_worker.utils import create_logger, format_time, stream_logging_format


class BioEngineWorker:
    """Manages Ray cluster lifecycle and model deployments on HPC systems or pre-existing Ray environments.

    Provides a Hypha service interface for controlling Ray cluster operations,
    autoscaling, and model deployments through Ray Serve.
    """

    def __init__(
        self,
        server_url: str = "https://hypha.aicell.io",
        workspace: str = None,
        token: Optional[str] = None,
        client_id: Optional[str] = None,
        service_id: str = "bioengine-worker",
        admin_users: Optional[List[str]] = None,
        mode: Literal["slurm", "single-machine", "connect"] = "slurm",
        ray_cluster_config: Optional[Dict] = None,
        clean_up_previous_cluster: bool = True,
        ray_autoscaler_config: Optional[Dict] = None,
        ray_connection_config: Optional[Dict] = None,
        ray_deployment_config: Optional[Dict] = None,
        dataset_config: Optional[Dict] = None,
        cache_dir: str = "/tmp/bioengine",
        logger: Optional[logging.Logger] = None,
        log_file: Optional[str] = None,
        _debug: bool = False,
    ):
        """Initialize BioEngine worker with component managers.

        Args:
            server_url: URL of the Hypha server to register the worker with.
            workspace: Hypha workspace to connect to. Defaults to user's workspace.
            token: Optional authentication token for the Hypha server.
            client_id: Optional client ID for the worker. If not provided, a new one will be generated.
            service_id: Service ID used when registering with the Hypha server.
            admin_users: List of user IDs or emails with admin permissions. If not set, defaults to the logged-in user.
            ray_cluster_config: Configuration for RayClusterManager component.
            clean_up_previous_cluster: Flag to indicate whether to cleanup of previous Ray cluster.
            ray_autoscaler_config: Configuration for the RayAutoscaler component.
            ray_connection_config: Optional arguments passed to `ray.init()` to connect to an existing ray cluster. If provided, disables cluster management.
            ray_deployment_config: Configuration for the RayDeploymentManager component.
            dataset_config: Optional configuration for data management.
            cache_dir: Directory for temporary files and Ray data. Defaults to `/tmp/bioengine`.
            logger: Optional custom logger. If not provided, a default logger will be created.
            log_file: File for logging output.
            _debug: Enable debug logging.
        """
        self.logger = logger or create_logger(
            name="BioEngineWorker",
            level=logging.DEBUG if _debug else logging.INFO,
            log_file=log_file,
        )
        try:
            self.server_url = server_url
            self.workspace = workspace
            self._token = token or os.environ.get("HYPHA_TOKEN")
            self.client_id = client_id
            self.service_id = service_id
            self.admin_users = admin_users or []
            self.start_time = None
            self.mode = mode
            self.cluster_manager = None
            self.autoscaler = None
            self._debug = _debug
            self.server = None
            self.serve_event = None
            cache_dir = Path(cache_dir).resolve()

            # If token is not provided, attempt to login
            if not self._token:
                print("\n" + "=" * 60)
                print("NO HYPHA TOKEN FOUND - USER LOGIN REQUIRED")
                print("-" * 60, end="\n\n")
                self._token = login({"server_url": self.server_url})
                print("\n" + "-" * 60)
                print("Login completed successfully!")
                print("=" * 60, end="\n\n")

            # Initialize component managers depending on the mode
            ray_cluster_config = ray_cluster_config or {}
            ray_autoscaler_config = ray_autoscaler_config or {}
            self.ray_connection_config = ray_connection_config or {}
            # TODO: integrate autoscaler and ray connection in ray cluster manager
            dataset_config = dataset_config or {}
            ray_deployment_config = ray_deployment_config or {}

            if self.mode in ["slurm", "single-machine"]:
                # os.environ["TMPDIR"] = str(cache_dir)

                # Set parameters for RayClusterManager

                # Overwrite existing 'mode', 'log_file', and '_debug' parameters if provided
                self._set_parameter(ray_cluster_config, "mode", self.mode)
                self._set_parameter(ray_cluster_config, "log_file", log_file)
                self._set_parameter(ray_cluster_config, "_debug", _debug)
                # Set default 'ray_temp_dir', 'worker_data_dir', and 'slurm_log_dir' if not provided
                self._set_parameter(
                    ray_cluster_config,
                    "ray_temp_dir",
                    cache_dir / "ray",
                    overwrite=False,
                )
                self._set_parameter(
                    ray_cluster_config,
                    "worker_data_dir",
                    dataset_config.get("data_dir"),
                    overwrite=False,
                )
                self._set_parameter(
                    ray_cluster_config,
                    "slurm_log_dir",
                    cache_dir / "slurm_logs",
                    overwrite=False,
                )

                # Initialize RayClusterManager and update mode
                self.cluster_manager = RayClusterManager(**ray_cluster_config)
                self.mode = self.cluster_manager.ray_cluster_config["mode"]

                # Set cluster cleanup flag
                self._clean_up = clean_up_previous_cluster

                if self.mode == "slurm":
                    # Set parameters for RayAutoscaler
                    self._set_parameter(
                        ray_autoscaler_config, "cluster_manager", self.cluster_manager
                    )
                    self._set_parameter(ray_autoscaler_config, "log_file", log_file)
                    self._set_parameter(ray_autoscaler_config, "_debug", _debug)
                    # Initialize RayAutoscaler
                    self.autoscaler = RayAutoscaler(**ray_autoscaler_config)
                else:
                    self.autoscaler = None

            elif self.mode == "connect":
                # Set parameters for connecting to an existing Ray cluster
                self._set_parameter(
                    self.ray_connection_config, "logging_format", stream_logging_format
                )

                if not "address" in self.ray_connection_config:
                    raise ValueError("Ray connection mode requires a provided address!")

                self.logger.info(
                    "Connecting to existing Ray cluster. Skipping cluster management."
                )
            else:
                raise ValueError(
                    f"Invalid mode '{self.mode}'. Choose from 'slurm', 'single-machine', or 'connect'."
                )

            # Set parameters for RayDeploymentManager
            self._set_parameter(ray_deployment_config, "admin_users", self.admin_users)
            self._set_parameter(
                ray_deployment_config,
                "deployment_cache_dir",
                cache_dir,
                overwrite=False,
            )
            self._set_parameter(ray_deployment_config, "autoscaler", self.autoscaler)
            self._set_parameter(ray_deployment_config, "log_file", log_file)
            self._set_parameter(ray_deployment_config, "_debug", _debug)
            # Initialize RayDeploymentManager
            self.deployment_manager = RayDeploymentManager(**ray_deployment_config)

            # Set parameters for DatasetManager
            self._set_parameter(dataset_config, "admin_users", self.admin_users)
            self._set_parameter(dataset_config, "log_file", log_file)
            self._set_parameter(dataset_config, "_debug", _debug)
            # Initialize DatasetManager
            self.dataset_manager = DatasetManager(**dataset_config)
        except Exception as e:
            self.logger.error(f"Error initializing BioEngineWorker: {e}")
            raise

    def _set_parameter(
        self,
        kwargs: Dict[str, Any],
        key: str,
        value: Any,
        overwrite: bool = True,
    ) -> dict:
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
        """Connect to Hypha server using provided URL.

        Args:
            server_url: URL of the Hypha server
            token: Token for authentication
            workspace: Workspace to connect to
        """
        self.logger.info(f"Connecting to Hypha server at {self.server_url}...")
        self.server = await connect_to_server(
            {
                "server_url": self.server_url,
                "token": self._token,
                "workspace": self.workspace,
                "client_id": self.client_id,
            }
        )
        if not self.server:
            raise ValueError("Failed to connect to Hypha server")

        user_id = self.server.config.user["id"]
        user_email = self.server.config.user["email"]

        self.workspace = self.server.config.workspace
        self.logger.info(
            f"User {user_id} connected to workspace '{self.workspace}' with client ID '{self.server.config.client_id}'"
        )

        # Add the logged-in user to admin users if not already present
        if user_email not in self.admin_users:
            self.admin_users.append(user_email)
            self.deployment_manager.admin_users = self.admin_users
            self.dataset_manager.admin_users = self.admin_users

        self.logger.info(f"Admin users for this worker: {', '.join(self.admin_users)}")

    async def _register(self) -> None:
        """Initialize connection to Hypha and register service interface.

        Args:
            server: Hypha server connection

        Returns:
            bool: True if initialization successful
        """
        # Register service interface
        description = "Manages BioEngine Apps and Datasets"
        if self.mode == "slurm":
            description += (
                " on a HPC system with Ray Autoscaler support for dynamic resource management."
            )
        elif self.mode == "single-machine":
            description += " on a single machine Ray instance."
        else:
            description += " in a pre-existing Ray environment."

        service_info = await self.server.register_service(
            {
                "id": self.service_id,
                "name": "BioEngine worker",
                "type": "bioengine-worker",
                "description": description,
                "config": {
                    "visibility": "public",
                    "require_context": True,
                },
                "get_status": self.get_status,
                "load_dataset": self.dataset_manager.load_dataset,
                "close_dataset": self.dataset_manager.close_dataset,
                "execute_python_code": self.execute_python_code,
                "create_artifact": self.deployment_manager.create_artifact,
                "deploy_artifact": self.deployment_manager.deploy_artifact,
                "undeploy_artifact": self.deployment_manager.undeploy_artifact,
                "deploy_all_artifacts": self.deployment_manager.deploy_all_artifacts,
                "cleanup_deployments": self.deployment_manager.cleanup_deployments,
                "cleanup": self.cleanup,
            },
            {"overwrite": True},
        )
        sid = service_info.id

        self.logger.info(f"BioEngine worker service registered with ID '{sid}'")
        self.logger.info(
            f"Manage BioEngine worker at: https://dev.bioimage.io/#/bioengine?service_id={sid}"
        )

        return sid

    async def start(self) -> str:
        """Start the BioEngine worker by initializing the Ray cluster or attaching to an existing one,
        connecting to the Hypha server, and initializing the deployment manager.

        Returns:
            The service ID assigned after successful registration with Hypha.
        """
        self.start_time = time.time()

        if self.mode == "connect":
            # Connect to an existing Ray cluster
            if ray.is_initialized():
                raise RuntimeError(
                    "Ray is already initialized. Please stop the existing Ray instance before starting the worker."
                )
            try:
                ray.init(**self.ray_connection_config)
            except Exception as e:
                self.logger.error(f"Failed to connect to existing Ray cluster: {e}")
                raise
            self.logger.info("Connected to existing Ray cluster.")
        else:
            # Start the Ray cluster
            self.cluster_manager.start_cluster(force_clean_up=self._clean_up)

            # If running on a HPC system, use the RayAutoscaler to manage the Ray cluster
            if self.mode == "slurm":
                await self.autoscaler.start()

        # Connect to the Hypha server and register the service
        await self._connect_to_server()
        await self.deployment_manager.initialize(self.server)
        await self.dataset_manager.initialize(self.server)
        sid = await self._register()
        await self.deployment_manager.initialize_deployments()

        return sid

    async def serve(self) -> None:
        """Keep the BioEngine worker running and serving requests."""
        if not self.server or not ray.is_initialized():
            raise RuntimeError("Server not initialized. Call start() first.")

        self.serve_event = asyncio.Event()
        await self.serve_event.wait()

    async def cleanup(self, context: Optional[Dict[str, Any]] = None) -> None:
        """Clean up resources and stop the Ray cluster if managed by this worker."""
        if ray.is_initialized():
            self.logger.info("Cleaning up resources...")

            await self.deployment_manager.cleanup_deployments()

            if self.mode == "slurm":
                await self.autoscaler.shutdown_cluster()
            elif self.mode == "single-machine":
                self.cluster_manager.shutdown_cluster()
        else:
            self.logger.warning("Ray is not initialized. No cleanup needed.")

        # Signal the serve loop to exit
        self.serve_event.set()

    async def get_status(self, context: Optional[Dict[str, Any]] = None) -> Dict:
        """Get comprehensive status of the Ray cluster or connected Ray instance.

        Returns:
            Dict containing service, cluster, autoscaler and deployment status.
        """
        if not ray.is_initialized():
            raise RuntimeError("Ray is not initialized. Call start() first.")

        formatted_service_time = format_time(self.start_time)
        status = {
            "service": {
                "start_time_s": self.start_time,
                "start_time": formatted_service_time["start_time"],
                "uptime": formatted_service_time["uptime"],
            }
        }

        if self.mode == "connect":
            # TODO: move ray cluster status completely to RayClusterManager
            # Ray connected externally
            head_address = ray._private.services.get_node_ip_address()
            status["ray_cluster"] = {
                "head_address": head_address,
                "start_time_s": "N/A",
                "start_time": "N/A",
                "uptime": "N/A",
                "worker_nodes": "N/A",
                "autoscaler": None,
                "note": "Connected to existing Ray cluster; no autoscaler info available.",
            }
        else:
            # Ray started internally
            status["ray_cluster"] = self.cluster_manager.get_status()
            if self.mode == "slurm":
                # Get autoscaler status if in SLURM mode
                status["ray_cluster"]["autoscaler"] = await self.autoscaler.get_status()
            else:
                status["ray_cluster"]["autoscaler"] = None
                status["ray_cluster"][
                    "note"
                ] = "Autoscaler is only available in 'slurm' mode."
        status["bioengine_apps"] = await self.deployment_manager.get_status()
        status["bioengine_datasets"] = await self.dataset_manager.get_status()

        return status

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
    ) -> dict:
        """Execute Python code in a Ray task."""
        self.logger.info("Executing Python code in Ray task...")

        args = args or []
        kwargs = kwargs or {}
        remote_options = remote_options or {}

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

        RemoteRayTask = ray.remote(**remote_options)(ray_task)
        future = RemoteRayTask.remote(user_func, args, kwargs)
        if self.mode == "slurm":
            await self.autoscaler.notify()
        result = await asyncio.get_event_loop().run_in_executor(None, ray.get, future)

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
                    "service_id": "bioengine-datasets",
                },
                ray_cluster_config={
                    "head_num_cpus": 4,
                    "ray_temp_dir": f"/tmp/ray/{os.environ['USER']}",
                    "image": str(
                        Path(__file__).parent.parent
                        / "apptainer_images"
                        / f"bioengine-worker_{__version__}.sif"
                    ),
                },
                ray_autoscaler_config={
                    "metrics_interval_seconds": 10,
                },
                ray_deployment_config={
                    "service_id": "bioengine-apps",
                },
                _debug=True,
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
            await bioengine_worker.cleanup()

    # Run the test
    asyncio.run(test_bioengine_worker())
