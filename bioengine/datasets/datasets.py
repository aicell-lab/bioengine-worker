import asyncio
import uuid
from typing import Dict, List, Optional, Union

import httpx
import zarr
from ray.serve import get_replica_context

from bioengine.datasets.http_zarr_store import HttpZarrStore


class BioEngineDatasets:
    """
    Client interface for accessing remote scientific datasets in BioEngine.

    This class provides a comprehensive client interface for accessing and streaming
    scientific datasets managed by the BioEngine Datasets service. It handles the
    complete lifecycle from discovery through connection to data access, with robust
    error handling and efficient streaming for large scientific data.

    The client implementation follows an asynchronous pattern for non-blocking I/O
    operations, making it suitable for use in interactive computing environments and
    high-performance applications where responsiveness is critical. It handles secure
    authentication, efficient streaming, and integration with scientific data formats
    through the Zarr protocol.

    The BioEngineDatasets client integrates with:
    - HTTP Zarr store for efficient partial data access
    - Ray Serve deployments for model-data integration
    - Remote dataset services with authentication
    - Asynchronous operation for high-performance data access

    Key Features:
    - Efficient partial data access through HttpZarrStore
    - Service discovery and connection management
    - Secure dataset access with authentication tokens
    - Rich metadata access through manifest files
    - Asynchronous API for non-blocking data operations
    - Integration with AnnData and other scientific data formats
    - Robust error handling with meaningful error messages

    Implementation Details:
    The client uses HTTP as the transport protocol and implements the Zarr store API
    for partial data access, allowing efficient operations on large datasets without
    loading entire files into memory.

    Dataset Access Workflow:
    The class implements a list → access → close pattern for dataset interaction,
    where datasets are first discovered through listing, then accessed through
    the get_dataset method which returns a Zarr group, and finally connections
    are managed automatically.

    Attributes:
        service_url (str): URL for the BioEngine Datasets service
        deployment_name (str): Name of the deployment using this client
        http_client (httpx.AsyncClient): HTTP client for service communication
        replica_id (str): Unique identifier for this client instance
    """

    def __init__(
        self,
        data_server_url: Union[str, None],  # set to None for no data server
        deployment_name: str,
        data_server_workspace: str = "public",
    ):
        """
        Initialize the BioEngineDatasets client for dataset access.

        Sets up a client connection to the BioEngine Datasets service for
        accessing scientific datasets. Configures the connection with proper
        authentication and logging for tracking access patterns.

        Args:
            data_server_url: URL of the datasets server, or None to disable remote access
            deployment_name: Identifier for the deployment using this client,
                           used for access logging and monitoring
            data_server_workspace: Hypha workspace name containing the datasets service,
                                 defaults to "public" for shared datasets

        Note:
            When data_server_url is None, only local dataset access will be available.
            The client uses a unique replica_id for tracking access patterns in logs.
        """
        # Get replica identifier for logging
        try:
            self.replica_id = get_replica_context().replica_tag
        except Exception:
            self.replica_id = f"uuid-{str(uuid.uuid4())[:8]}"

        print(
            f"🚀 [{self.replica_id}] Initializing {self.__class__.__name__} "
            f"for '{deployment_name}'"
        )
        print(f"🔗 [{self.replica_id}] Data server URL: {data_server_url}")
        print(
            f"🏢 [{self.replica_id}] Data service workspace: '{data_server_workspace}'"
        )

        if data_server_url is not None:
            self.service_url = (
                f"{data_server_url}/{data_server_workspace}/services/bioengine-datasets"
            )
            self.http_client = httpx.AsyncClient(timeout=20)  # seconds
        else:
            self.service_url = None

        self.deployment_name = deployment_name

    async def ping_data_server(self):
        """
        Verify connectivity to the dataset service.

        Tests the connection to the dataset service by sending a simple ping request.
        This method should be called before attempting to access datasets to ensure
        that the service is available and responsive.

        Raises:
            RuntimeError: If the connection to the data server fails for any reason
        """
        if self.service_url is not None:
            try:
                # Try to ping dataset service
                await self.http_client.get(f"{self.service_url}/ping")
            except Exception as e:
                print(
                    f"⚠️ [{self.replica_id}] Connection to data server "
                    f"failed for '{self.deployment_name}': {e}"
                )
                raise RuntimeError("Connection to data server failed")

    async def list_datasets(self) -> Dict[str, dict]:
        """
        Retrieve a dictionary of available datasets from the service.

        Queries the dataset service for all datasets that are available to the current
        user. This is typically the first step in the dataset access workflow and
        provides the names needed for subsequent operations.

        Returns:
            Dictionary of dataset names and their manifest available to the current user.
            Empty dictionary if service_url is None.

        Raises:
            httpx.HTTPStatusError: If the request fails due to HTTP error
        """
        if self.service_url is None:
            return {}

        start_time = asyncio.get_event_loop().time()
        response = await self.http_client.get(f"{self.service_url}/list_datasets")
        response.raise_for_status()
        datasets = response.json()
        end_time = asyncio.get_event_loop().time()
        print(
            f"🕒 [{self.replica_id}] Listed {len(datasets)} dataset(s) in {end_time - start_time:.2f} seconds"
        )

        return datasets

    async def list_files(
        self, dataset_name: str, token: Optional[str] = None
    ) -> List[str]:
        """
        Retrieve a list of available files within a specific dataset.

        Queries the dataset service for all files available within the specified dataset.
        This is typically the second step in the dataset access workflow, after listing
        available datasets. The method handles authentication and access control through
        the optional token parameter.

        Args:
            dataset_name: Name of the dataset to list files from
            token: Optional authentication token for accessing protected datasets

        Returns:
            List of file paths available within the specified dataset

        Raises:
            ValueError: If the service_url is None or dataset does not exist
            httpx.HTTPStatusError: If the request fails due to HTTP error
        """
        if self.service_url is None:
            raise ValueError(f"Dataset '{dataset_name}' does not exist")

        start_time = asyncio.get_event_loop().time()
        query_url = (
            f"{self.service_url}/list_files?dataset_name={dataset_name}&token={token}"
        )
        response = await self.http_client.get(query_url)
        response.raise_for_status()
        files = response.json()
        end_time = asyncio.get_event_loop().time()
        print(
            f"🕒 [{self.replica_id}] Listed {len(files)} file(s) in dataset '{dataset_name}' in {end_time - start_time:.2f} seconds"
        )

        return files

    async def get_dataset(
        self, dataset_name: str, file: Optional[str] = None, token: Optional[str] = None
    ) -> zarr.Group:
        """
        Access a dataset as a Zarr group for efficient data operations.

        This is the primary method for accessing dataset content, providing access to
        the data through Zarr's efficient partial data access mechanisms. The method
        validates dataset and file existence, handles authentication, and returns a
        connected Zarr group for immediate data access.

        Dataset Access Process:
        1. Validates dataset existence through list_datasets
        2. Checks file availability through list_files
        3. Auto-selects the file if only one is available
        4. Creates an HttpZarrStore for efficient streaming access
        5. Opens and returns a Zarr group for data operations

        Args:
            dataset_name: Name of the dataset to access
            file: Optional specific file within the dataset to access.
                 If None and only one file exists, that file is automatically selected.
            token: Optional authentication token for accessing protected datasets

        Returns:
            Connected Zarr group for data access operations

        Raises:
            ValueError: If dataset/file doesn't exist or ambiguous file selection
            RuntimeError: If connection to data server fails
        """
        start_time = asyncio.get_event_loop().time()

        available_datasets = await self.list_datasets()
        if dataset_name not in available_datasets.keys():
            raise ValueError(f"Dataset '{dataset_name}' does not exist")

        available_files = await self.list_files(dataset_name=dataset_name, token=token)
        if len(available_files) == 0:
            raise ValueError(f"No files found in dataset '{dataset_name}'")

        if file is None:
            if len(available_files) > 1:
                raise ValueError(
                    f"File not specified and multiple files found in dataset '{dataset_name}'"
                )
            file = available_files[0]
        else:
            if file not in available_files:
                raise ValueError(f"File '{file}' not found in dataset '{dataset_name}'")

        store = HttpZarrStore(
            service_url=self.service_url, dataset_name=dataset_name, token=token
        )

        dataset = await asyncio.to_thread(zarr.open_group, store, mode="r", path=file)

        end_time = asyncio.get_event_loop().time()
        print(
            f"🕒 [{self.replica_id}] Time taken to get dataset: {end_time - start_time:.2f} seconds"
        )

        return dataset


if __name__ == "__main__":
    from pathlib import Path

    from anndata.experimental import read_lazy

    async def test_bioengine_datasets():
        cache_dir = Path.home() / ".bioengine" / "datasets"

        current_server_file = cache_dir / "bioengine_current_server"

        data_server_url = current_server_file.read_text()

        bioengine_datasets = BioEngineDatasets(
            data_server_url=data_server_url,
            deployment_name="test-deployment",
            data_server_workspace="public",
        )
        await bioengine_datasets.ping_data_server()

        available_datasets = await bioengine_datasets.list_datasets()

        dataset_name = list(available_datasets.keys())[0]

        dataset = await bioengine_datasets.get_dataset(dataset_name)
        print(dataset)

        # `read_lazy` from anndata==0.12.0rc1
        adata = await asyncio.to_thread(read_lazy, dataset, load_annotation_index=True)
        print(adata)
        print(adata.obs)
        print(adata.X)

    asyncio.run(test_bioengine_datasets())
