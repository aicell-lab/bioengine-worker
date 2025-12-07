import argparse
import asyncio
import os
from pathlib import Path

from hypha_rpc import connect_to_server, login

from bioengine.utils import (
    create_application_from_files,
    create_file_list_from_directory,
    create_logger,
)


async def save_application(
    directory: str,
    server_url: str,
    workspace: str = None,
    token: str = None,
) -> str:
    """
    Creates or updates a BioEngine application artifact in the Hypha artifact manager.

    Args:
        directory: Path to the artifact directory
        server_url: URL of the Hypha server
        workspace: Hypha workspace to connect to
        token: Authentication token for Hypha server

    Returns:
        str: ID of the created artifact

    Raises:
        Exception: If any step in the process fails
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
    user_id = server.config.user["id"]

    logger.info(
        f"Connected to workspace {workspace} as user {user_id} (client {server.config.client_id})"
    )

    artifact_manager = await server.get_service("public/artifact-manager")
    logger.info("Connected to artifact manager")

    # Create file list from directory
    try:
        files = create_file_list_from_directory(
            directory_path=directory,
        )
        logger.info(f"Created file list with {len(files)} files")
        # Remove any files originating from __pycache__ directories
        pycache_files = [f for f in files if "__pycache__" in f.get("name", "")]
        if pycache_files:
            original_count = len(files)
            files = [f for f in files if f not in pycache_files]
            logger.info(
                f"Removed {len(pycache_files)} __pycache__ files; {original_count - len(pycache_files)} files remain"
            )
    except Exception as e:
        logger.error(f"Failed to create file list from directory: {e}")
        raise

    # Create or update the artifact using the utility function
    try:
        artifact_id = await create_application_from_files(
            artifact_manager=artifact_manager,
            files=files,
            workspace=workspace,
            user_id=user_id,
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
        save_application(
            directory=args.directory,
            server_url=args.server_url,
            workspace=args.workspace,
            token=args.token,
        )
    )
