import argparse
import asyncio
import os
from pathlib import Path

import httpx
import yaml
from hypha_rpc import connect_to_server, login

from bioengine_worker.utils import create_logger


async def manage_artifact(
    deployment_dir: str,
    parent_id: str,
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
    with open(deployment_dir / "manifest.yaml", "r") as f:
        deployment_manifest = yaml.safe_load(f)
    logger.info("Deployment manifest loaded")

    deployment_alias = deployment_manifest["id"]
    invalid = any(
        [
            not deployment_alias.islower(),
            "_" in deployment_alias,
            not deployment_alias.replace("-", "_").isidentifier(),
        ]
    )
    if invalid:
        raise ValueError(
            f"Invalid deployment id: '{deployment_alias}'. Please use lowercase letters, numbers, and hyphens only."
        )
    if delete is True:
        await artifact_manager.delete(artifact_id=deployment_alias)
        if "/" not in deployment_alias:
            deployment_alias = f"{server.config.workspace}/{deployment_alias}"
        logger.info(f"Artifact {deployment_alias} deleted")
        return

    python_file = deployment_manifest.get("python_file", "main.py")
    with open(deployment_dir / python_file, "r") as f:
        deployment_content = f.read()
    logger.info(f"Deployment content loaded from {python_file}")

    try:
        # Edit the existing deployment and stage it for review
        artifact = await artifact_manager.edit(
            artifact_id=deployment_alias,
            manifest=deployment_manifest,
            type=deployment_manifest.get("type", "generic"),
            version="stage",
        )
    except:
        # Add the deployment to the gallery and stage it for review
        artifact = await artifact_manager.create(
            alias=deployment_alias,
            parent_id=parent_id,
            manifest=deployment_manifest,
            type=deployment_manifest.get("type", "generic"),
            version="stage",
        )
        logger.info(f"Artifact created with ID: {artifact.id}")

    # Upload manifest.yaml
    upload_url = await artifact_manager.put_file(artifact.id, file_path="manifest.yaml")
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.put(upload_url, data=deployment_manifest)
        response.raise_for_status()
        logger.info(f"Uploaded manifest.yaml to artifact")

    # Upload the entry point
    upload_url = await artifact_manager.put_file(artifact.id, file_path=python_file)
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.put(upload_url, data=deployment_content)
        response.raise_for_status()
        logger.info(f"Uploaded {python_file} to artifact")

    # Commit the artifact
    await artifact_manager.commit(
        artifact_id=artifact.id,
        version="new",
    )
    logger.info(f"Committed artifact with ID: {artifact.id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create or update a deployment artifact",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--deployment_dir",
        type=Path,
        required=True,
        help="Path to the deployment directory",
    )
    parser.add_argument(
        "--parent_id",
        default="ray-deployments",
        type=str,
        help="Parent ID for the artifact",
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
            deployment_dir=args.deployment_dir,
            parent_id=args.parent_id,
            server_url=args.server_url,
            workspace=args.workspace,
            token=args.token,
            delete=args.delete,
        )
    )
