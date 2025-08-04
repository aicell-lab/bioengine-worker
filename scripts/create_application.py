import argparse
import asyncio
import os
from pathlib import Path

from hypha_rpc import connect_to_server, login

from bioengine_worker.utils import (
    create_artifact_from_files,
    create_file_list_from_directory,
    create_logger,
    ensure_applications_collection,
)


async def manage_artifact(
    directory: str,
    server_url: str,
    workspace: str = None,
    token: str = None,
) -> str:
    """Create or delete a deployment artifact in Hypha

    Args:
        directory: Path to the artifact directory
        server_url: URL of the Hypha server
        workspace: Hypha workspace to connect to
        token: Authentication token for Hypha server
        delete: Delete the artifact instead of creating/updating it

    Returns:
        str: ID of the created artifact (None for delete operations)
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
    workspace = server.config.workspace

    logger.info(
        f"Connected to workspace {workspace} as client {server.config.client_id}"
    )

    artifact_manager = await server.get_service("public/artifact-manager")
    logger.info("Connected to artifact manager")

    # Create file list from directory
    try:
        files, artifact_alias = create_file_list_from_directory(
            directory_path=directory,
        )
        extracted_artifact_id = f"{workspace}/{artifact_alias}"
        logger.info(
            f"Created file list with {len(files)} files for artifact {extracted_artifact_id}"
        )
    except Exception as e:
        logger.error(f"Failed to create file list from directory: {e}")
        raise

    # Ensure applications collection exists
    await ensure_applications_collection(
        artifact_manager=artifact_manager,
        workspace=workspace,
        logger=logger,
    )

    # Create or update the artifact using the utility function
    try:
        artifact_id = await create_artifact_from_files(
            artifact_manager=artifact_manager,
            files=files,
            workspace=workspace,
            collection_id=f"{workspace}/applications",
            logger=logger,
        )
        return artifact_id
    except Exception as e:
        logger.error(f"Failed to create/update artifact: {e}")
        raise


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

    args = parser.parse_args()

    asyncio.run(
        manage_artifact(
            directory=args.directory,
            server_url=args.server_url,
            workspace=args.workspace,
            token=args.token,
        )
    )
