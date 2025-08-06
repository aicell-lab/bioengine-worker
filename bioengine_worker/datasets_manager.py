import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from hypha_rpc.rpc import RemoteService
from hypha_rpc.utils.schema import schema_method

from bioengine_worker.utils import check_permissions, create_logger


# TODO: Move all dataset interaction to Ray actors (or ray serve?)
# TODO: for each loaded dataset, register a Hypha ASGI service in a separate Ray actor
# TODO: Update status, only return loaded datasets and their status
# TODO: Update list datasets, return available datasets and some information about them

class DatasetsManager:
    """
    Manages dataset loading, access control, and HTTP streaming services for BioEngine datasets.

    This class provides comprehensive dataset management by integrating with the Hypha
    server to expose datasets as HTTP streaming services. It handles the complete lifecycle
    from dataset discovery through service registration to cleanup, with robust permission
    control and file streaming capabilities.

    The DatasetsManager orchestrates:
    - Dataset discovery and manifest validation from filesystem
    - Permission-based access control for dataset operations and file access
    - HTTP streaming service registration with Hypha server
    - ASGI-based file serving with range request support
    - Real-time dataset monitoring and status reporting
    - Graceful service cleanup and resource management

    Key Features:
    - Automatic dataset discovery from directory structure with manifest validation
    - Admin-level permission control for dataset loading/unloading operations
    - User-level authorization for file access based on dataset configuration
    - HTTP range request support for efficient large file streaming
    - CORS-enabled endpoints for cross-origin access
    - Comprehensive error handling and logging with proper state management
    - Real-time service monitoring and status reporting

    Dataset Structure:
    Each dataset must contain a manifest.yml file with the following structure:
    ```yaml
    description: "Dataset description"
    authorized_users: ["user@example.com", "*"]  # "*" for public access
    files:
      data_file.zarr:
        description: "File description"
        version: "1.0.0"
        n_samples: 1000
        n_vars: 500
    ```

    Attributes:
        server: Hypha server connection instance
        admin_users (List[str]): List of user emails with admin permissions for dataset operations
        loaded_datasets (Dict): Tracking of currently loaded dataset services
        service_id_base (str): Base service ID prefix for dataset services
        logger: Logger instance for dataset operations
        _datasets (Dict): Internal dataset registry with manifest data and file paths
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        # Logger
        log_file: Optional[Union[str, Path]] = None,
        debug: bool = False,
    ):
        """
        Initialize the DatasetsManager with dataset discovery and configuration.

        Scans the specified data directory for dataset manifests and validates
        dataset structure. Sets up logging and initializes state variables for
        dataset service management.

        Args:
            data_dir: Root directory containing dataset subdirectories with manifest.yml files
            log_file: Optional log file path for output
            debug: Enable debug logging

        Raises:
            Exception: If dataset discovery or manifest validation fails

        Note:
            The admin_users parameter is passed during initialize() call,
            not during construction. Server connection is also established
            during initialization.
        """

        # Set up logging
        self.logger = create_logger(
            name="DatasetsManager",
            level=logging.DEBUG if debug else logging.INFO,
            log_file=log_file,
        )

        # Load dataset info
        self._datasets = self._load_dataset_info(data_dir)

        # Initialize state variables
        self.service_id_base = "bioengine-dataset"
        self.loaded_datasets = {}
        self.server = None
        self.admin_users = None

    def _check_initialized(self) -> None:
        """
        Check if the server connection is initialized.

        Raises:
            RuntimeError: If server connection is not available
        """
        if not self.server:
            raise RuntimeError(
                "Hypha server connection not available. Call initialize() first."
            )

    @property
    def datasets(self) -> Dict[str, Dict]:
        """
        Return the public dataset information without internal attributes.

        Returns:
            Dict containing dataset metadata excluding internal file paths
        """
        if not self._datasets:
            return {}
        return {
            dataset_id: {
                key: value for key, value in dataset_info.items() if key != "_path"
            }
            for dataset_id, dataset_info in self._datasets.items()
        }

    def _load_dataset_info(self, data_dir: str) -> Dict[str, Dict]:
        """
        Read and parse dataset manifest files from the data directory.

        Scans the data directory for subdirectories containing manifest.yml files,
        validates dataset structure, and builds the internal dataset registry.

        Args:
            data_dir: Root directory path containing dataset subdirectories

        Returns:
            Dict mapping dataset IDs to their manifest data and file paths

        Raises:
            Exception: If directory access fails or manifest parsing errors occur
        """
        try:
            data_dir = Path(data_dir).resolve()

            # Check if data directory exists
            if not data_dir.exists():
                self.logger.warning(
                    f"Data directory {data_dir} does not exist. Skipping dataset loading."
                )
                return {}

            # Check if path is actually a directory
            if not data_dir.is_dir():
                self.logger.warning(
                    f"Data directory {data_dir} is not a directory. Skipping dataset loading."
                )
                return {}

            # Check if path is readable
            if not os.access(data_dir, os.R_OK):
                self.logger.warning(
                    f"Data directory {data_dir} is not readable. Skipping dataset loading."
                )
                return {}

            datasets = {}

            # Try to access the directory - this will catch permission errors
            try:
                directory_contents = list(data_dir.iterdir())
            except PermissionError:
                self.logger.warning(
                    f"Permission denied accessing data directory {data_dir}. Skipping dataset loading."
                )
                return {}
            except OSError as e:
                self.logger.warning(
                    f"Cannot access data directory {data_dir}: {e}. Skipping dataset loading."
                )
                return {}

            for dataset_path in directory_contents:
                if not dataset_path.is_dir():
                    continue

                manifest_file = dataset_path / "manifest.yml"
                if not manifest_file.exists():
                    self.logger.warning(
                        f"No manifest file found in {dataset_path}. Skipping dataset."
                    )
                    continue

                with open(manifest_file, "r") as f:
                    manifest = yaml.safe_load(f)

                # TODO: Update required manifest fields
                # Validate required manifest fields
                if "files" not in manifest:
                    self.logger.warning(
                        f"Manifest file {manifest_file} missing 'files' field. Skipping dataset."
                    )
                    continue

                # Add default authorized_users if not specified
                if "authorized_users" not in manifest:
                    manifest["authorized_users"] = ["*"]  # Default to public access
                    self.logger.warning(
                        f"Dataset '{dataset_path.name}' missing 'authorized_users', defaulting to public access."
                    )

                dataset_id = dataset_path.name
                datasets[dataset_id] = {
                    **manifest,
                    "_path": dataset_path,
                }

                # Check if all data files exist
                for data_file, attributes in manifest["files"].items():
                    data_file_path = dataset_path / data_file
                    if not data_file_path.suffix == ".zarr":
                        self.logger.warning(
                            f"Data file {data_file_path} is not a .zarr file. Skipping dataset."
                        )
                    elif not data_file_path.exists() or not data_file_path.is_dir():
                        self.logger.warning(
                            f"Data file {data_file_path} does not exist or is not a directory. Skipping dataset."
                        )
                    elif "n_samples" not in attributes.keys():
                        self.logger.warning(
                            f"Data file {data_file_path} does not have 'n_samples' in the manifest. Skipping dataset."
                        )
                    else:
                        continue
                    datasets.pop(dataset_id)
                    break

            return datasets

        except Exception as e:
            self.logger.error(f"Error loading dataset info: {e}")
            raise e

    def _define_app(self, dataset_id: str) -> FastAPI:
        """
        Define the FastAPI app for serving dataset files with permission control.

        Creates a FastAPI application that serves dataset files with proper CORS
        configuration, range request support, and user authorization checks.

        Args:
            dataset_id: ID of the dataset to serve

        Returns:
            FastAPI application instance configured for file serving
        """
        app = FastAPI()

        # Enable CORS for all origins — you can restrict this to specific domains
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Or restrict to your frontend domain
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=[
                "Content-Range",
                "Content-Length",
                "Accept-Ranges",
                "Accept-encoding",
                "Connection",
            ],  # <- important
        )

        @app.get("/")
        async def root():
            return {dataset_id: self.datasets[dataset_id]}

        @app.get("/files/{path:path}")
        async def serve_file(path: str, request: Request):
            # Extract context from request (provided by Hypha ASGI wrapper)
            context = getattr(request.state, "context", None)

            # Check user permissions for dataset file access using standardized check_permissions
            try:
                authorized_users = self._datasets[dataset_id]["authorized_users"]
                check_permissions(
                    context=context,
                    authorized_users=authorized_users,
                    resource_name=f"access files in dataset '{dataset_id}'",
                )
            except PermissionError as e:
                self.logger.warning(
                    f"File access denied for dataset '{dataset_id}': {e}"
                )
                raise HTTPException(status_code=403, detail=str(e))

            file_path = self._datasets[dataset_id]["_path"] / path
            if not file_path.exists() or not file_path.is_file():
                raise HTTPException(status_code=404, detail="File not found")

            file_size = file_path.stat().st_size
            range_header = request.headers.get("range")

            def file_stream(start: int = 0, end: Optional[int] = None):
                with open(file_path, "rb") as f:
                    f.seek(start)
                    remaining = (end - start + 1) if end is not None else None
                    while True:
                        chunk_size = 1024 * 1024  # 1MB
                        if remaining is not None:
                            chunk_size = min(chunk_size, remaining)
                            if chunk_size <= 0:
                                break
                        data = f.read(chunk_size)
                        if not data:
                            break
                        if remaining is not None:
                            remaining -= len(data)
                        yield data

            if range_header:
                try:
                    range_value = range_header.replace("bytes=", "").split("-")
                    start = int(range_value[0])
                    end = int(range_value[1]) if range_value[1] else file_size - 1
                    if start > end or end >= file_size:
                        raise ValueError()
                except Exception:
                    raise HTTPException(status_code=416, detail="Invalid Range header")
                content_length = end - start + 1
                return StreamingResponse(
                    file_stream(start, end),
                    status_code=206,
                    headers={
                        "Content-Range": f"bytes {start}-{end}/{file_size}",
                        "Accept-Ranges": "bytes",
                        "Content-Length": str(content_length),
                        "Content-Type": "application/octet-stream",
                    },
                )

            # Full file
            return StreamingResponse(
                file_stream(),
                headers={
                    "Content-Length": str(file_size),
                    "Content-Type": "application/octet-stream",
                    "Accept-Ranges": "bytes",
                },
            )

        return app

    def _get_service_url(self, sid) -> str:
        server_url = self.server.config.public_base_url
        workspace, sid = sid.split("/")
        service_url = f"{server_url}/{workspace}/apps/{sid}"
        return service_url

    async def _register_service(self, dataset_id) -> None:
        """Register the data service to the Hypha server."""
        # Define the app for streaming files from the dataset
        dataset_app = self._define_app(dataset_id)

        # Hypha ASGI service integration
        async def serve_fastapi(args, context=None):
            scope = args["scope"]
            if context and "user" in context:
                self.logger.debug(
                    f'{context["user"]["id"]} - {scope["client"]} - {scope["method"]} - {scope["path"]}'
                )

            # Attach context to request state for access in FastAPI endpoints
            if "state" not in scope:
                scope["state"] = {}
            scope["state"]["context"] = context

            await dataset_app(args["scope"], args["receive"], args["send"])

        dataset_service_id = f"{self.service_id_base}-{dataset_id}"
        service_info = await self.server.register_service(
            {
                "id": dataset_service_id,
                "name": "BioEngine Worker Datasets",
                "description": "Streaming files from BioEngine Worker",
                "type": "asgi",
                "serve": serve_fastapi,
                "config": {"visibility": "public", "require_context": True},
            }
        )
        service_info["url"] = self._get_service_url(service_info.id)
        self.loaded_datasets[dataset_id] = service_info

        self.logger.info(
            f"Successfully registered data service for dataset '{dataset_id}'"
        )
        self.logger.info(f"Access the app at: {service_info['url']}")

    async def initialize(self, server: RemoteService, admin_users: List[str]) -> None:
        """
        Initialize the dataset manager with a Hypha server connection.

        Establishes connection to the Hypha server for dataset service registration
        and stores admin user permissions for dataset operations.

        Args:
            server: Hypha server connection instance
            admin_users: List of user IDs or emails with admin permissions for dataset operations

        Raises:
            Exception: If server connection initialization fails
        """
        # Store server connection and list of admin users
        self.server = server
        self.admin_users = admin_users

    async def get_status(self) -> Dict[str, dict]:
        """
        Get the comprehensive status of the dataset manager.

        Returns:
            Dict containing available datasets and currently loaded dataset services
        """
        return self.loaded_datasets

    async def monitor_datasets(self) -> None:
        """
        Monitor the datasets and log their status.

        Placeholder for future dataset monitoring functionality such as
        service health checks, usage statistics, and performance metrics.
        """
        try:
            # TODO: Implement monitoring
            pass
        except Exception as e:
            self.logger.error(f"Error monitoring datasets: {e}")
            raise e
        
    async def list_datasets(self, context: Dict[str, Any]) -> Dict[str, Dict]:
        """
        List all available datasets with their metadata.

        Returns:
            Dict containing dataset IDs and their metadata

        Raises:
            RuntimeError: If server connection is not initialized
        """
        return self.datasets

    @schema_method
    async def load_dataset(self, dataset_id: str, context: Dict[str, Any]) -> str:
        """
        Load a dataset by ID and register it as an HTTP streaming service.

        Creates and registers a new Hypha ASGI service for the specified dataset,
        enabling HTTP access to dataset files with proper authorization checks.
        Requires admin permissions to perform dataset loading operations.

        Args:
            dataset_id: ID of the dataset to load
            context: User context information automatically injected by Hypha.

        Returns:
            str: URL of the registered dataset service

        Raises:
            RuntimeError: If server connection is not initialized
            PermissionError: If user lacks admin permissions
            ValueError: If dataset is not available or already loaded
            Exception: If service registration fails
        """
        self._check_initialized()

        # Check admin permissions for dataset operations
        check_permissions(
            context=context,
            authorized_users=self.admin_users,
            resource_name=f"load dataset '{dataset_id}'",
        )

        user_id = context["user"]["id"] if context and "user" in context else "unknown"

        try:
            if dataset_id not in self._datasets.keys():
                raise ValueError(f"Dataset '{dataset_id}' not available.")
            if dataset_id in self.loaded_datasets:
                self.logger.info(f"Dataset {dataset_id} is already open.")
                return self.loaded_datasets[dataset_id]["url"]

            self.logger.info(f"User '{user_id}' is loading dataset '{dataset_id}'...")
            await self._register_service(dataset_id)

            return self.loaded_datasets[dataset_id]["url"]
        except Exception as e:
            self.logger.error(f"Error loading dataset {dataset_id}: {e}")
            raise e

    @schema_method
    async def close_dataset(self, dataset_id: str, context: Dict[str, Any]) -> str:
        """
        Close a dataset by ID and unregister its HTTP streaming service.

        Unregisters the Hypha ASGI service for the specified dataset and removes
        it from the loaded datasets tracking. Requires admin permissions to
        perform dataset closing operations.

        Args:
            dataset_id: ID of the dataset to close
            context: User context information automatically injected by Hypha.

        Returns:
            str: Confirmation message of successful dataset closure

        Raises:
            RuntimeError: If server connection is not initialized
            PermissionError: If user lacks admin permissions
            ValueError: If dataset is not available or not currently loaded
            Exception: If service unregistration fails
        """
        self._check_initialized()

        # Check admin permissions for dataset operations
        check_permissions(
            context=context,
            authorized_users=self.admin_users,
            resource_name=f"close dataset '{dataset_id}'",
        )

        user_id = context["user"]["id"] if context and "user" in context else "unknown"

        try:
            if dataset_id not in self._datasets.keys():
                raise ValueError(f"Dataset '{dataset_id}' not available.")
            if dataset_id not in self.loaded_datasets:
                self.logger.info(f"Dataset {dataset_id} is not loaded.")
                raise ValueError(f"Dataset '{dataset_id}' is not loaded.")

            self.logger.info(f"User '{user_id}' is closing dataset '{dataset_id}'...")
            service_info = self.loaded_datasets[dataset_id]
            await self.server.unregister_service(service_info["id"])
            self.loaded_datasets.pop(dataset_id)
            self.logger.info(f"Successfully unregistered dataset '{dataset_id}'")
            return f"Dataset '{dataset_id}' closed."
        except Exception as e:
            self.logger.error(f"Error closing dataset {dataset_id}: {e}")
            raise e

    @schema_method
    async def cleanup(self, context: Dict[str, Any]) -> str:
        """
        Close all loaded datasets and clean up all HTTP streaming services.

        Unregisters all currently loaded dataset services and clears the
        loaded datasets tracking. Requires admin permissions to perform
        cleanup operations.

        Args:
            context: User context information automatically injected by Hypha.

        Returns:
            str: Confirmation message of successful cleanup

        Raises:
            RuntimeError: If server connection is not initialized
            PermissionError: If user lacks admin permissions
            Exception: If cleanup operations fail
        """
        if not self.loaded_datasets:
            self.logger.info("No datasets are currently loaded.")
            return

        self._check_initialized()

        # Check admin permissions for cleanup operations
        check_permissions(
            context=context,
            authorized_users=self.admin_users,
            resource_name="cleanup all datasets",
        )

        user_id = context["user"]["id"] if context and "user" in context else "unknown"

        try:
            self.logger.info(f"User '{user_id}' is starting cleanup of all datasets...")
            for dataset_id in list(self.loaded_datasets.keys()):
                await self.close_dataset(dataset_id=dataset_id, context=context)
        except Exception as e:
            self.logger.error(f"Error closing all datasets: {e}")
            raise e
