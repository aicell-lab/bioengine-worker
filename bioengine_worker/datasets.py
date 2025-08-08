import asyncio

import httpx
from zarr import open_group

# Use relative import to not require the 'bioengine_worker' package in the deployment
from .http_zarr_store import HttpZarrStore


class BioEngineDatasets:
    def __init__(
        self,
        server_url: str,
        workspace: str,
        token: str,
        max_n_datasets: int = 100,
    ):
        self.server_url = server_url
        self.workspace = workspace
        self.max_n_datasets = max_n_datasets

        download_timeout = httpx.Timeout(30.0)
        self.httpx_client = httpx.AsyncClient(
            timeout=download_timeout, follow_redirects=True
        )
        self.headers = {"Authorization": f"Bearer {token}"}

    async def list(self):
        url = f"{self.server_url}/{self.workspace}/artifacts/bioengine-datasets/children?limit={self.max_n_datasets}"
        response = await self.httpx_client.get(url, headers=self.headers)
        response.raise_for_status()

        datasets = [artifact["alias"] for artifact in response.json()]

        return datasets

    async def get(self, dataset_name):
        available_datasets = await self.list()
        if dataset_name not in available_datasets:
            return None

        base_url = f"{self.server_url}/{self.workspace}/artifacts/{dataset_name}/files/"
        store = HttpZarrStore(base_url=base_url, headers=self.headers)

        dataset = await asyncio.to_thread(open_group, store, mode="r")

        return dataset


if __name__ == "__main__":
    import os

    from anndata.experimental import read_lazy
    from hypha_rpc import connect_to_server, login

    async def test_bioengine_datasets(server_url="https://hypha.aicell.io"):
        token = os.environ["HYPHA_TOKEN"] or await login({"server_url": server_url})

        hypha_client = await connect_to_server(
            {"server_url": server_url, "token": token}
        )
        workspace = hypha_client.config.workspace

        bioengine_datasets = BioEngineDatasets(
            server_url=server_url, workspace=workspace, token=token
        )

        available_datasets = await bioengine_datasets.list()

        dataset_name = available_datasets[0]

        dataset = await bioengine_datasets.get(dataset_name)

        # `read_lazy` or `zarr.open_group` depending on your use
        adata = read_lazy(dataset, load_annotation_index=True)
        print(adata)

    asyncio.run(test_bioengine_datasets())
