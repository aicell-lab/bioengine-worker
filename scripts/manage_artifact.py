import argparse
import asyncio
import os
from pathlib import Path

import httpx
import yaml
from hypha_rpc import connect_to_server, login

from bioengine_worker.utils import create_logger


async def manage_artifact(
    directory: str,
    server_url: str,
    workspace: str = None,
    token: str = None,
    delete: bool = False,
) -> str:
    """Create an deployment artifact in Hypha

    Args:
        artifact_manager: Hypha artifact manager instance
        path_to_artifact: Path to the artifact directory
        logger: Optional logger instance

    Returns:
        str: ID of the created artifact
    """
    logger = create_logger("ArtifactManager")

    # Connect to Hypha and get artifact manager
    token = token or os.environ.get("HYPHA_TOKEN")
    if not token:
        token = await login({"server_url": server_url})
    server = await connect_to_server(
        {
            "server_url": server_url,
            "workspace": workspace,
            "token": token,
        }
    )
    logger.info(
        f"Connected to workspace {server.config.workspace} as client {server.config.client_id}"
    )

    artifact_manager = await server.get_service("public/artifact-manager")
    logger.info("Connected to artifact manager")

    # Get the deployment manifest and content
    with open(directory / "manifest.yaml", "r") as f:
        deployment_manifest = yaml.safe_load(f)
    logger.info("Deployment manifest loaded")

    artifact_id = deployment_manifest["id"]
    invalid = any(
        [
            not artifact_id.islower(),
            "_" in artifact_id,
            not artifact_id.replace("-", "_").isidentifier(),
        ]
    )
    if invalid:
        raise ValueError(
            f"Invalid deployment id: '{artifact_id}'. Please use lowercase letters, numbers, and hyphens only."
        )

    workspace = server.config.workspace
    artifact_id = artifact_id if "/" in artifact_id else f"{workspace}/{artifact_id}"

    if delete is True:
        try:
            await artifact_manager.delete(artifact_id=artifact_id)
            logger.info(f"Artifact {artifact_id} deleted")
        except Exception as e:
            logger.error(f"Failed to delete artifact {artifact_id}: {e}")
        return

    try:
        # Edit the existing deployment and stage it for review
        artifact = await artifact_manager.edit(
            artifact_id=artifact_id,
            manifest=deployment_manifest,
            type=deployment_manifest.get("type", "application"),
            stage=True,
        )
    except:
        artifact_workspace = artifact_id.split("/")[0]
        collection_id = f"{artifact_workspace}/applications"
        try:
            await artifact_manager.read(collection_id)
        except Exception as e:
            expected_error = (
                f"KeyError: \"Artifact with ID '{collection_id}' does not exist.\""
            )
            if str(e).strip().endswith(expected_error):
                logger.info(
                    f"Collection '{collection_id}' does not exist. Creating it."
                )

            collection_manifest = {
                "name": "BioEngine Apps",
                "description": "A collection of Ray deployments for the BioEngine.",
            }
            collection = await artifact_manager.create(
                alias=collection_id,
                type="collection",
                manifest=collection_manifest,
                config={"permissions": {"*": "r", "@": "r+"}},
            )
            logger.info(f"Bioengine Apps collection created with ID: {collection.id}")

        # Add the deployment to the gallery and stage it for review
        artifact = await artifact_manager.create(
            alias=artifact_id,
            parent_id=collection_id,
            manifest=deployment_manifest,
            type="application",
            stage=True,
        )
        logger.info(f"Artifact created with ID: {artifact.id}")

    # Upload all files in the deployment directory
    for file_path in directory.rglob("*"):
        if file_path.is_file():
            relative_path = file_path.relative_to(directory)

            # Get upload URL
            upload_url = await artifact_manager.put_file(
                artifact.id, file_path=str(relative_path)
            )

            # Read file content and upload
            try:
                # Try to read as text first
                with open(file_path, "r", encoding="utf-8") as f:
                    file_content = f.read()

                # Upload as text
                async with httpx.AsyncClient(timeout=30) as client:
                    response = await client.put(upload_url, data=file_content)
                    response.raise_for_status()
                    logger.info(f"Uploaded {relative_path} to artifact (text)")

            except UnicodeDecodeError:
                # If text reading fails, read as binary
                with open(file_path, "rb") as f:
                    file_content = f.read()

                # Upload as binary
                async with httpx.AsyncClient(timeout=30) as client:
                    response = await client.put(upload_url, content=file_content)
                    response.raise_for_status()
                    logger.info(f"Uploaded {relative_path} to artifact (binary)")

    # Commit the artifact
    await artifact_manager.commit(
        artifact_id=artifact.id,
    )
    logger.info(f"Committed artifact with ID: {artifact.id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create or update a deployment artifact",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-d",
        "--directory",
        type=Path,
        required=True,
        help="Path to the deployment directory",
    )
    parser.add_argument(
        "--server_url",
        default="https://hypha.aicell.io",
        type=str,
        help="URL of the Hypha server",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        help="Hypha workspace to connect to",
    )
    parser.add_argument(
        "--token",
        type=str,
        help="Authentication token for Hypha server",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete the artifact instead of creating or updating it",
    )

    args = parser.parse_args()

    asyncio.run(
        manage_artifact(
            directory=args.directory,
            server_url=args.server_url,
            workspace=args.workspace,
            token=args.token,
            delete=args.delete,
        )
    )
