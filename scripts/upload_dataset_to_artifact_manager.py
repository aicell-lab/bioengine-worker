import asyncio
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import aiofiles
import httpx
from hypha_rpc import connect_to_server, login
from tqdm.asyncio import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def ensure_collection(
    artifact_manager,
    workspace: str,
) -> str:
    """Ensure the collection exists, creating it if necessary."""
    collection_alias = "bioengine-datasets"
    collection_id = f"{workspace}/{collection_alias}"
    collection_manifest = {
        "name": "BioEngine Datasets",
        "description": f"A collection of Zarr-file datasets for workspace {workspace}",
    }
    collection_config = {
        "permissions": {
            "@": "*"
        },  # Allow all users of the same workspace to access the collection
    }

    try:
        collection = await artifact_manager.edit(
            artifact_id=collection_id,
            manifest=collection_manifest,
            type="collection",
            config=collection_config,
        )
        logger.info(f"Collection '{collection_id}' already exists")
    except Exception as collection_error:
        expected_error = (
            f"KeyError: \"Artifact with ID '{collection_id}' does not exist.\""
        )
        if str(collection_error).strip().endswith(expected_error):
            logger.info(f"Collection '{collection_id}' does not exist. Creating it.")

            try:
                collection = await artifact_manager.create(
                    type="collection",
                    alias=collection_alias,
                    manifest=collection_manifest,
                    collection_config=collection_config,
                )
                logger.info(
                    f"{collection_manifest['name']} collection created with ID: {collection.id}"
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to create {collection_manifest['name']} collection: {e}"
                )
        else:
            raise RuntimeError(
                f"Failed to check {collection_manifest['name']} collection '{collection_id}': {collection_error}"
            )


async def stage_artifact(
    artifact_manager,
    workspace: str,
    dataset_name: str,
):
    """Create a staged artifact in the artifact manager."""
    collection_id = f"{workspace}/bioengine-datasets"
    dataset_manifest = {
        "name": dataset_name,
        "description": f"A streamable BioEngine dataset",
        "type": "zarr",
    }
    dataset_config = {
        "permissions": {
            "@": "*"
        },  # Allow all users of the same workspace to access the dataset
    }
    try:
        # Edit the dataset artifact if it exists
        artifact = await artifact_manager.edit(
            artifact_id=f"{workspace}/{dataset_name}",
            manifest=dataset_manifest,
            type="dataset",  # Fixed from "application"
            config=dataset_config,
            stage=True,
        )
        logger.info(f"Dataset '{dataset_name}' already exists, editing it.")
    except Exception as e:
        # Create the dataset artifact
        artifact = await artifact_manager.create(
            type="dataset",
            parent_id=collection_id,
            alias=dataset_name,
            manifest=dataset_manifest,
            config=dataset_config,
            stage=True,
        )
        logger.info(
            f"Created new dataset artifact '{dataset_name}' with ID: {artifact.id}"
        )

    return artifact


async def upload_file_to_artifact_manager(
    artifact_manager,
    artifact_id: str,
    file_path: Path,
    relative_path: str,
    httpx_client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    progress_bar: Optional[tqdm] = None,
) -> bool:
    """Upload a file to the artifact manager with retry logic and validation."""
    async with semaphore:  # Control concurrent uploads
        file_size_mb = file_path.stat().st_size / (1024 * 1024)

        logger.info(f"Uploading {relative_path} ({file_size_mb:.2f}MB)")

        max_attempts = 4
        backoff = 1.0  # backoff: 1.0s, 2.0s, 4.0s
        backoff_multiplier = 2.0

        # Retry upload logic
        for attempt in range(1, max_attempts + 1):
            try:
                upload_url = await artifact_manager.put_file(
                    artifact_id, file_path=relative_path
                )

                # Stream upload for memory efficiency
                async with aiofiles.open(file_path, "rb") as f:
                    content = await f.read()

                response = await httpx_client.put(
                    upload_url,
                    content=content,
                )
                response.raise_for_status()

                logger.info(f"Successfully uploaded {relative_path}")

                # Update progress bar
                if progress_bar:
                    progress_bar.update(1)

                return True

            except Exception as e:
                # If this was the last attempt, don't wait
                if attempt < max_attempts:
                    logger.warning(
                        f"⚠️ Upload attempt {attempt}/{max_attempts} failed for {relative_path}: {e}. "
                        f"Retrying in {backoff:.2f}s..."
                    )

                    # Sleep with exponential backoff before retrying
                    await asyncio.sleep(backoff)
                    backoff *= backoff_multiplier
                else:
                    logger.error(
                        f"⚠️ Failed to upload {relative_path} after {max_attempts} attempts: {e}"
                    )
                    # Update progress bar even for failed uploads
                    if progress_bar:
                        progress_bar.update(1)
                        progress_bar.set_postfix_str(f"Failed: {relative_path}")


async def upload_zarr_dataset_batch(
    artifact_manager,
    artifact_id: str,
    zarr_path: Path,
    relative_paths: List[str],
    httpx_client: httpx.AsyncClient,
    batch_size: int = 10,
) -> Dict[str, bool]:
    """Upload a batch of zarr files with controlled concurrency."""
    semaphore = asyncio.Semaphore(batch_size)

    # Create progress bar
    progress_bar = tqdm(
        total=len(relative_paths), desc="Uploading files", unit="file", colour="green"
    )

    upload_tasks = [
        upload_file_to_artifact_manager(
            artifact_manager,
            artifact_id,
            zarr_path / relative_path,
            relative_path,
            httpx_client,
            semaphore,
            progress_bar,
        )
        for relative_path in relative_paths
    ]

    results = await asyncio.gather(*upload_tasks, return_exceptions=True)

    # Close progress bar
    progress_bar.close()

    # Process results
    upload_results = {}
    for i, result in enumerate(results):
        relative_path = relative_paths[i]
        if isinstance(result, Exception):
            logger.error(f"Upload failed for {relative_path}: {result}")
            upload_results[relative_path] = False
        else:
            upload_results[relative_path] = result

    return upload_results


async def main(
    zarr_files: dict,
    server_url: str = "https://hypha.aicell.io",
    upload_batch_size: int = 30,
):
    """Main function to upload zarr datasets to artifact manager."""
    # Connect to artifact manager
    token = os.getenv("HYPHA_TOKEN") or await login({"server_url": server_url})
    hypha_client = await connect_to_server({"server_url": server_url, "token": token})
    workspace = hypha_client.config.workspace
    artifact_manager = await hypha_client.get_service("public/artifact-manager")

    # Create BioEngine Datasets collection
    await ensure_collection(artifact_manager, workspace)

    # Upload Zarr files to the collection with improved batching
    upload_timeout = httpx.Timeout(120.0)  # Increased timeout
    async with httpx.AsyncClient(timeout=upload_timeout) as httpx_client:
        for dataset_name, zarr_file_path in zarr_files.items():
            # Create a staged artifact
            artifact = await stage_artifact(
                artifact_manager=artifact_manager,
                workspace=workspace,
                dataset_name=dataset_name,
            )

            # Create a list of relative paths in the Zarr file
            zarr_file_path = Path(zarr_file_path)
            relative_paths = [
                str(item.relative_to(zarr_file_path))
                for item in zarr_file_path.glob("**/*")
                if item.is_file()
            ]

            # Upload the Zarr dataset in batches
            results = await upload_zarr_dataset_batch(
                artifact_manager=artifact_manager,
                artifact_id=artifact.id,
                zarr_path=zarr_file_path,
                relative_paths=relative_paths,
                httpx_client=httpx_client,
                batch_size=upload_batch_size,
            )

            successful_uploads = sum(1 for success in results.values() if success)
            logger.info(
                f"Successfully uploaded {successful_uploads}/{len(relative_paths)} files"
            )

            await artifact_manager.commit(artifact.id)


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent.parent / "data"

    zarr_files = {
        "thymus": data_dir / "thymus" / "filter_22.zarr",
    }

    asyncio.run(main(zarr_files))
