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
from bioengine_worker.apps_manager import AppsManager
from bioengine_worker.datasets_manager import DatasetsManager
from bioengine_worker.ray_cluster import RayCluster
from bioengine_worker.utils import create_logger, format_time, stream_logging_format


class BioEngineWorker:
    """Manages Ray cluster lifecycle and model deployments on HPC systems or pre-existing Ray environments.

    Provides a Hypha service interface for controlling Ray cluster operations,
    autoscaling, and model deployments through Ray Serve.
    """

    def __init__(
        self,
        mode: Literal["slurm", "single-machine", "connect"] = "slurm",
        admin_users: Optional[List[str]] = None,
        cache_dir: str = "/tmp/bioengine",
        data_dir: str = "/data",
        startup_deployments: Optional[List[str]] = None,
        # Hypha server connection configuration
        server_url: str = "https://hypha.aicell.io",
        workspace: Optional[str] = None,
        token: Optional[str] = None,
        client_id: Optional[str] = None,
        # Ray cluster configuration
        ray_cluster_config: Optional[Dict] = None,
        # Logger configuration
        log_file: Optional[str] = None,
        debug: bool = False,
    ):
        """Initialize BioEngine worker with component managers.

        Args:
            server_url: URL of the Hypha server to register the worker with.
            workspace: Hypha workspace to connect to. Defaults to user's workspace.
            token: Optional authentication token for the Hypha server.
            client_id: Optional client ID for the worker. If not provided, a new one will be generated.
            service_id: Service ID used when registering with the Hypha server.
            admin_users: List of user IDs or emails with admin permissions. If not set, defaults to the logged-in user.
            mode: Mode of operation for the worker. Can be 'slurm', 'single-machine', or 'connect'.
            ray_cluster_config: Configuration for RayCluster component.
            clean_up_previous_cluster: Flag to indicate whether to cleanup of previous Ray cluster.
            ray_autoscaler_config: Configuration for the RayAutoscaler component.
            ray_connection_config: Optional arguments passed to `ray.init()` to connect to an existing ray cluster. If provided, disables cluster management.
            ray_deployment_config: Configuration for the AppsManager component.
            dataset_config: Optional configuration for data management.
            cache_dir: Directory for temporary files and Ray data. Defaults to `/tmp/bioengine`.
            logger: Optional custom logger. If not provided, a default logger will be created.
            log_file: File for logging output.
            debug: Enable debug logging.
        """
        self.mode = mode
        self.admin_users = admin_users or []
        self.cache_dir = Path(cache_dir).resolve()
        self.data_dir = Path(data_dir).resolve()
        self.startup_deployments = startup_deployments or []

        self.server_url = server_url
        self.workspace = workspace
        self._token = token or os.environ.get("HYPHA_TOKEN")
        self.client_id = client_id
        self.service_id = "bioengine-worker"

        self.start_time = None
        self.server = None
        self.serve_event = None

        if not log_file:
            log_dir = self.cache_dir / "logs"
            log_file = (
                log_dir / f"bioengine_worker_{time.strftime('%Y%m%d_%H%M%S')}.log"
            )

        self.logger = create_logger(
            name="BioEngineWorker",
            level=logging.DEBUG if debug else logging.INFO,
            log_file=log_file,
        )
        try:
            # If token is not provided, attempt to login
            if not self._token:
                print("\n" + "=" * 60)
                print("NO HYPHA TOKEN FOUND - USER LOGIN REQUIRED")
                print("-" * 60, end="\n\n")
                self._token = login({"server_url": self.server_url})
                print("\n" + "-" * 60)
                print("Login completed successfully!")
                print("=" * 60, end="\n\n")

            # Set temporary directory for Ray runtime installation
            os.environ["TMPDIR"] = str(self.cache_dir)
            # os.environ["HOME"] = str(self.cache_dir)

            # Set parameters for RayCluster
            ray_cluster_config = ray_cluster_config or {}

            # Overwrite existing 'mode', 'log_file', and 'debug' parameters if provided
            self._set_parameter(ray_cluster_config, "mode", self.mode)
            self._set_parameter(ray_cluster_config, "log_file", log_file)
            self._set_parameter(ray_cluster_config, "debug", debug)
            self._set_parameter(ray_cluster_config, "ray_temp_dir", self.cache_dir / "ray")
            force_clean_up = not ray_cluster_config.pop("skip_cleanup", False)
            self._set_parameter(ray_cluster_config, "force_clean_up", force_clean_up)

            # Initialize RayCluster and update mode
            self.ray_cluster = RayCluster(**ray_cluster_config)

            # Update mode in case SLURM is not available
            self.mode = self.ray_cluster.mode

            # Initialize AppsManager
            self.deployment_manager = AppsManager(
                mode=self.mode,
                admin_users=self.admin_users,
                cache_dir=self.cache_dir / "apps",
                data_dir=self.data_dir,
                startup_deployments=self.startup_deployments,
                autoscaler=self.ray_cluster.autoscaler,
                log_file=log_file,
                debug=debug,
            )

            # Initialize DatasetsManager
            self.dataset_manager = DatasetsManager(
                data_dir=self.data_dir,
                admin_users=self.admin_users,
                log_file=log_file,
                debug=debug,
            )
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
            description += " on a HPC system with Ray Autoscaler support for dynamic resource management."
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
            await self.ray_cluster.start()

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
            await self.dataset_manager.cleanup_datasets()
            await self.deployment_manager.cleanup_deployments()
            await self.ray_cluster.shutdown()
        else:
            self.logger.warning("Ray is not initialized. No cleanup needed.")

        # Signal the serve loop to exit
        self.serve_event.set()

    async def get_status(self, context: Optional[Dict[str, Any]] = None) -> dict:
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
            },
            "ray_cluster": self.ray_cluster.status,
            "bioengine_apps": await self.deployment_manager.get_status(),
            "bioengine_datasets": await self.dataset_manager.get_status(),
        }

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
            await self.ray_cluster.autoscaler.notify()
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
                    "service_id": "bioengine-dataset",
                },
                ray_cluster_config={
                    "head_num_cpus": 4,
                    "ray_temp_dir": str(
                        Path(__file__).parent.parent / ".bioengine" / "ray"
                    ),
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
                debug=True,
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
