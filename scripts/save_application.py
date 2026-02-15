import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

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
    worker_service_id: str = None,
    manifest_file: str = None,
    permissions: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Creates or updates a BioEngine application artifact in the Hypha artifact manager.

    Args:
        directory: Path to the artifact directory
        server_url: URL of the Hypha server
        workspace: Hypha workspace to connect to
        token: Authentication token for Hypha server
        worker_service_id: Optional worker service ID to use creating/updating the artifact
        manifest_file: Optional path to a specific manifest file to use (will be renamed to manifest.yaml)
        permissions: Optional permissions to set on the artifact

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
    
    # Connect without forcing workspace (allow cross-workspace access)
    server = await connect_to_server(
        {
            "server_url": server_url,
            "token": token,
        }
    )
    
    # Use provided workspace or fallback to connected workspace
    if not workspace:
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

        if manifest_file:
            manifest_path = Path(manifest_file)
            if not manifest_path.is_absolute():
                manifest_path = Path(directory) / manifest_path

            if not manifest_path.exists():
                raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

            # Remove existing manifest.yaml if present in the list
            files = [f for f in files if f["name"] != "manifest.yaml"]

            relative_manifest_path = manifest_path.relative_to(directory).as_posix()

            manifest_entry = next(
                (f for f in files if f["name"] == relative_manifest_path), None
            )

            if manifest_entry:
                logger.info(f"Using {manifest_file} as manifest.yaml")
                manifest_entry["name"] = "manifest.yaml"
            else:
                logger.info(f"Adding {manifest_file} as manifest.yaml")
                with open(manifest_path, "rb") as f:
                    manifest_content = f.read()
                files.append(
                    {
                        "name": "manifest.yaml",
                        "content": manifest_content,
                        "type": "application/x-yaml",
                    }
                )

    except Exception as e:
        logger.error(f"Failed to create file list from directory: {e}")
        raise

    if worker_service_id is not None:
        logger.info(f"Using worker service ID: {worker_service_id}")
        try:
            bioengine_service = await server.get_service(worker_service_id)
        except KeyError:
            logger.error(f"Worker service {worker_service_id} not found")
            raise
        except Exception as e:
            logger.error(f"Failed to get worker service {worker_service_id}: {e}")
            raise

        # Create or update the artifact using the bioengine service
        try:
            artifact_id = await bioengine_service.save_application(files=files)
            logger.info(
                f"Successfully created/updated application artifact '{artifact_id}'."
            )
            return artifact_id
        except Exception as e:
            logger.error(f"Failed to create/update artifact: {e}")
            raise

    else:
        # Create or update the artifact using the utility function
        try:
            create_kwargs = {
                "artifact_manager": artifact_manager,
                "files": files,
                "workspace": workspace,
                "user_id": user_id,
                "logger": logger,
            }
            if permissions is not None:
                create_kwargs["permissions"] = permissions

            try:
                artifact_id = await create_application_from_files(**create_kwargs)
            except TypeError as e:
                if "permissions" in str(e):
                    create_kwargs.pop("permissions", None)
                    artifact_id = await create_application_from_files(**create_kwargs)
                else:
                    raise
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
        "--server-url",
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
        "--worker-service-id",
        type=str,
        help="Optional worker service ID to use creating/updating the artifact",
    )
    parser.add_argument(
        "--manifest-file",
        type=str,
        help="Optional path to a specific manifest file (relative to directory or absolute)",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Make the artifact publicly matching",
    )
    parser.add_argument(
        "--permissions",
        type=str,
        help="JSON string of permissions to set on the artifact",
    )

    args = parser.parse_args()

    perms = None
    if args.permissions:
        perms = json.loads(args.permissions)
    elif args.public:
        perms = {"*": "r"}

    asyncio.run(
        save_application(
            directory=args.directory,
            server_url=args.server_url,
            workspace=args.workspace,
            token=args.token,
            worker_service_id=args.worker_service_id,
            manifest_file=args.manifest_file,
            permissions=perms,
        )
    )
