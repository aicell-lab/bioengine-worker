import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from hypha_rpc.rpc import RemoteService
from hypha_rpc.utils.schema import schema_method

from bioengine_worker.utils import create_logger


# TODO: Add user authorization checks
class DatasetsManager:
    def __init__(
        self,
        data_dir: str,
        # Logger
        log_file: Optional[str] = None,
        debug: bool = False,
    ):

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

        # TODO: Implement admin user checks for opening and closing datasets
        self.admin_users = None

    @property
    def datasets(self) -> Dict[str, Dict]:
        """Return the datasets without the internal attributes."""
        if not self._datasets:
            return {}
        return {
            dataset_id: {
                key: value
                for key, value in dataset_info.items()
                if not key.startswith("_")
            }
            for dataset_id, dataset_info in self._datasets.items()
        }

    def _load_dataset_info(self, data_dir) -> Dict[str, Dict]:
        """Read and parse a manifest.yaml file."""
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
                        f"Manifest file not found in {dataset_path}. Skipping dataset."
                    )
                    continue

                with open(manifest_file, "r") as f:
                    manifest = yaml.safe_load(f)

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
        """Define the FastAPI app for serving files."""
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
            self.logger.debug(
                f'{context["user"]["id"]} - {scope["client"]} - {scope["method"]} - {scope["path"]}'
            )
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
        """Initialize the dataset manager with a Hypha server connection

        Args:
            server: Hypha server connection
        """
        # Store server connection and list of admin users
        self.server = server
        self.admin_users = admin_users

    async def get_status(self) -> Dict[str, dict]:
        """Get the status of the dataset manager."""
        return {
            "available_datasets": self.datasets,
            "loaded_datasets": self.loaded_datasets,
        }

    async def monitor_datasets(self) -> None:
        """Monitor the datasets and log their status."""
        # TODO: Implement monitoring
        pass

    @schema_method
    async def load_dataset(self, dataset_id, context=None) -> str:
        """Load a dataset by ID."""
        try:
            if dataset_id not in self._datasets.keys():
                raise ValueError(f"Dataset '{dataset_id}' not available.")
            if dataset_id in self.loaded_datasets:
                self.logger.info(f"Dataset {dataset_id} is already open.")
                return self.loaded_datasets[dataset_id]["url"]

            await self._register_service(dataset_id)

            return self.loaded_datasets[dataset_id]["url"]
        except Exception as e:
            self.logger.error(f"Error loading dataset {dataset_id}: {e}")
            raise e

    @schema_method
    async def close_dataset(self, dataset_id, context=None) -> str:
        """Close a dataset by ID."""
        try:
            if dataset_id not in self._datasets.keys():
                raise ValueError(f"Dataset '{dataset_id}' not available.")
            if dataset_id not in self.loaded_datasets:
                self.logger.info(f"Dataset {dataset_id} is not loaded.")
                raise ValueError(f"Dataset '{dataset_id}' is not loaded.")

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
        """Close all loaded datasets."""
        try:
            if not self.loaded_datasets:
                self.logger.info("No datasets are currently loaded.")
                return "No datasets to close."

            for dataset_id in list(self.loaded_datasets.keys()):
                await self.close_dataset(dataset_id)

            return "All datasets closed successfully."
        except Exception as e:
            self.logger.error(f"Error closing all datasets: {e}")
            raise e
