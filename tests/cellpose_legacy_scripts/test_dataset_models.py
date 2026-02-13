"""Test the list_models_by_dataset functionality."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from hypha_rpc import connect_to_server, login

if TYPE_CHECKING:
    from hypha_rpc.rpc import RemoteService

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_list_models_by_dataset() -> None:
    """Test listing models trained on a specific dataset."""
    server_url = os.environ.get("HYPHA_SERVER_URL", "https://hypha.aicell.io")
    token = os.environ.get("HYPHA_TOKEN") or await login({"server_url": server_url})

    async with connect_to_server({"server_url": server_url, "token": token}) as server:  # type: ignore[generalTypeIssues]
        cellpose_service = await server.get_service("bioimage-io/cellpose-finetuning")
        logger.info("Connected to Cellpose Fine-Tuning service")

        # Test dataset ID - using the same dataset from test_service.py
        dataset_id = "ri-scale/zarr-demo"
        collection = "bioimage-io/colab-annotations"

        logger.info(f"Querying models trained on dataset: {dataset_id}")
        logger.info(f"Searching in collection: {collection}")

        # List models trained on this dataset
        models = await cellpose_service.list_models_by_dataset(
            dataset_id=dataset_id,
            collection=collection,
        )

        logger.info(f"\nFound {len(models)} model(s) trained on this dataset:")

        if models:
            for i, model in enumerate(models, 1):
                logger.info(f"\n{i}. {model['name']}")
                logger.info(f"   ID: {model['id']}")
                logger.info(f"   URL: {model['url']}")
                logger.info(f"   Created: {model['created_at']}")
        else:
            logger.info("No models found. This may mean:")
            logger.info("  1. No models have been trained and exported from this dataset yet")
            logger.info("  2. The training_dataset_id was not stored correctly")
            logger.info("\nTo test this feature:")
            logger.info(f"  1. Run: python scripts/test_service.py")
            logger.info(f"  2. This will train and export a model using dataset: {dataset_id}")
            logger.info(f"  3. Then run this script again to see the exported model")


if __name__ == "__main__":
    asyncio.run(test_list_models_by_dataset())
