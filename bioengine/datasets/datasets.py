import asyncio
from typing import List, Optional

import httpx
import zarr

from bioengine.datasets.http_zarr_store import HttpZarrStore


class BioEngineDatasets:
    def __init__(
        self,
        server_url: str,
        workspace: str = "public",
        token: Optional[str] = None,
        max_n_datasets: int = 100,
    ):
        self.server_url = server_url
        self.workspace = workspace
        self.max_n_datasets = max_n_datasets

        download_timeout = httpx.Timeout(30.0)
        self.httpx_client = httpx.AsyncClient(
            timeout=download_timeout, follow_redirects=True
        )
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}

    async def list_datasets(self) -> List[str]:
        url = f"{self.server_url}/{self.workspace}/artifacts/bioengine-datasets/children?limit={self.max_n_datasets}"
        response = await self.httpx_client.get(url, headers=self.headers)
        response.raise_for_status()

        datasets = [artifact["alias"] for artifact in response.json()]

        return datasets

    async def list_files(self, dataset: str) -> List[str]:
        url = f"{self.server_url}/{self.workspace}/artifacts/{dataset}/files/?limit={self.max_n_datasets}"
        response = await self.httpx_client.get(url, headers=self.headers)
        response.raise_for_status()

        files = [
            file["name"]
            for file in response.json()
            if file["name"].endswith(".zarr") and file["type"] == "directory"
        ]

        return files

    async def get(self, dataset_name: str, file: Optional[str] = None) -> zarr.Group:
        # TODO: How to validate user access permissions for datasets and files
        available_datasets = await self.list_datasets()
        if dataset_name not in available_datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")

        available_files = await self.list_files(dataset_name)
        if len(available_files) == 0:
            raise ValueError(f"No files found in dataset '{dataset_name}'")

        if file is None:
            if len(available_files) < 1:
                raise ValueError(
                    f"File not specified and multiple files found in dataset '{dataset_name}'"
                )
            file = available_files[0]
        else:
            if file not in available_files:
                raise ValueError(f"File '{file}' not found in dataset '{dataset_name}'")

        base_url = f"{self.server_url}/{self.workspace}/artifacts/{dataset_name}/files/"
        store = HttpZarrStore(base_url=base_url, headers=self.headers)

        dataset = await asyncio.to_thread(zarr.open_group, store, mode="r", path=file)

        return dataset


if __name__ == "__main__":
    from pathlib import Path

    from anndata.experimental import read_lazy

    async def test_bioengine_datasets():
        cache_dir = Path.home() / ".bioengine" / "datasets"

        current_server_file = cache_dir / "bioengine_current_server"
        access_token_file = cache_dir / ".access_token"

        server_url = current_server_file.read_text()
        workspace = "public"
        token = access_token_file.read_text()

        bioengine_datasets = BioEngineDatasets(
            server_url=server_url, workspace=workspace, token=token
        )

        available_datasets = await bioengine_datasets.list_datasets()

        dataset_name = available_datasets[0]

        dataset = await bioengine_datasets.get(dataset_name)
        print(dataset)

        # `read_lazy` or `zarr.open_group` depending on your use
        adata = read_lazy(dataset, load_annotation_index=True)
        print(adata)

    asyncio.run(test_bioengine_datasets())
