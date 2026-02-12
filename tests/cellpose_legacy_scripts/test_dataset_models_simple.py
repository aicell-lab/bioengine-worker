"""Quick test of list_models_by_dataset functionality."""

import asyncio
import os
from hypha_rpc import connect_to_server

async def main():
    server_url = os.environ.get("HYPHA_SERVER_URL", "https://hypha.aicell.io")
    token = os.environ.get("HYPHA_TOKEN")

    if not token:
        print("ERROR: HYPHA_TOKEN not set")
        return

    print(f"Connecting to {server_url}...")
    async with connect_to_server({"server_url": server_url, "token": token}) as server:
        print("Connected! Getting service...")
        cellpose_service = await server.get_service("bioimage-io/cellpose-finetuning")
        print("Got service! Querying models...")

        models = await cellpose_service.list_models_by_dataset(
            dataset_id="ri-scale/zarr-demo",
            collection="bioimage-io/colab-annotations",
        )

        print(f"\nFound {len(models)} model(s):")
        for model in models:
            print(f"  - {model['name']} ({model['id']})")
            print(f"    URL: {model['url']}")

if __name__ == "__main__":
    asyncio.run(main())
