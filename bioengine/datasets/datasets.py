import asyncio
import uuid
from typing import List, Optional, Union

import httpx
import zarr
from ray.serve import get_replica_context

from bioengine.datasets.http_zarr_store import HttpZarrStore


class BioEngineDatasets:
    def __init__(
        self,
        data_server_url: Union[str, None],  # set to None for no data server
        deployment_name: str,
        data_server_workspace: str = "public",
    ):
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

    async def list_datasets(self) -> List[str]:
        if self.service_url is None:
            return []

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
        start_time = asyncio.get_event_loop().time()

        available_datasets = await self.list_datasets()
        if dataset_name not in available_datasets:
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

        dataset_name = available_datasets[0]

        dataset = await bioengine_datasets.get_dataset(dataset_name)
        print(dataset)

        # `read_lazy` from anndata==0.12.0rc1
        adata = await asyncio.to_thread(read_lazy, dataset, load_annotation_index=True)
        print(adata)
        print(adata.obs)
        print(adata.X)

    asyncio.run(test_bioengine_datasets())
