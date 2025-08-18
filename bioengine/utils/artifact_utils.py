import base64
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import httpx
import yaml
from hypha_rpc.rpc import ObjectProxy

from .logger import create_logger


def create_file_list_from_directory(
    directory_path: Union[str, Path],
    _artifact_id_suffix: Optional[str] = None,
) -> tuple[List[Dict[str, Any]], str]:
    """
    Convert a local directory to a list of file dictionaries for artifact creation.

    This utility function reads all files from a directory and converts them to the
    format expected by create_application_from_files. It automatically handles text
    and binary files, and can modify the manifest ID with a suffix for testing.

    Args:
        directory_path: Path to the directory containing application files
        _artifact_id_suffix: Optional suffix to append to the manifest ID
                           (useful for creating unique test artifacts)

    Returns:
        Tuple of (list of file dictionaries, artifact alias)
        File dictionaries contain 'name', 'content', and 'type' keys
        Artifact alias is the ID from manifest.yaml without workspace prefix

    Raises:
        ValueError: If no manifest.yaml found or directory doesn't exist
        RuntimeError: If file reading fails

    Example:
        files, artifact_alias = create_file_list_from_directory(
            directory_path="/path/to/my-app",
            _artifact_id_suffix="test-123"
        )
        # Returns files list and "my-app-test-123"
        # Then construct full ID: f"{workspace}/{artifact_alias}"
    """
    directory_path = Path(directory_path)
    if not directory_path.exists():
        raise ValueError(f"Directory does not exist: {directory_path}")
    if not directory_path.is_dir():
        raise ValueError(f"Path is not a directory: {directory_path}")

    files = []
    artifact_alias = None

    for file_path in directory_path.rglob("*"):
        if file_path.is_file():
            # Get relative path from the directory
            relative_path = file_path.relative_to(directory_path)

            # Read file content
            try:
                # Try to read as text first
                content = file_path.read_text(encoding="utf-8")
                file_type = "text"
            except UnicodeDecodeError:
                # If it fails, read as binary and encode as base64
                content = base64.b64encode(file_path.read_bytes()).decode("ascii")
                file_type = "base64"
            except Exception as e:
                raise RuntimeError(f"Failed to read file {relative_path}: {e}")

            # Update manifest files with suffix if specified
            if str(relative_path) == "manifest.yaml" and _artifact_id_suffix:
                try:
                    manifest = yaml.safe_load(content)
                    if not isinstance(manifest, dict):
                        raise ValueError(
                            f"Invalid manifest structure: expected dict, got {type(manifest)}"
                        )
                    if "id" not in manifest:
                        raise ValueError("Manifest missing required 'id' field")

                    # Append suffix to ID
                    hyphen_suffix = _artifact_id_suffix.replace("_", "-")
                    original_id = manifest["id"]
                    manifest["id"] = f"{original_id}-{hyphen_suffix}"
                    content = yaml.dump(manifest)
                except yaml.YAMLError as e:
                    raise ValueError(f"Failed to parse manifest YAML: {e}")
                except Exception as e:
                    raise ValueError(f"Failed to update manifest: {e}")

            # Extract artifact alias from manifest
            if str(relative_path) == "manifest.yaml":
                try:
                    manifest = yaml.safe_load(content)
                    if "id" in manifest:
                        artifact_alias = manifest["id"]
                except yaml.YAMLError:
                    pass  # Will be caught later if artifact_alias is None

            files.append(
                {"name": str(relative_path), "content": content, "type": file_type}
            )

    if artifact_alias is None:
        raise ValueError("No manifest.yaml file found or manifest missing 'id' field")

    return files, artifact_alias


async def ensure_applications_collection(
    artifact_manager: ObjectProxy,
    workspace: str,
    logger: Optional[logging.Logger] = None,
) -> str:
    """
    Ensure the 'applications' collection exists in the Hypha artifact manager.

    Creates the applications collection if it doesn't exist, providing organized
    storage for BioEngine application artifacts within the current workspace.
    The collection acts as a container for grouping related applications.

    Args:
        artifact_manager: Hypha artifact manager service instance
        workspace: Hypha workspace identifier
        logger: Optional logger instance for debugging

    Returns:
        The collection ID in format 'workspace/applications'

    Raises:
        RuntimeError: If the collection cannot be created or accessed

    Example:
        collection_id = await ensure_applications_collection(
            artifact_manager=am,
            workspace="my-workspace"
        )
        # Returns "my-workspace/applications"
    """
    if logger is None:
        logger = create_logger("ArtifactUtils")

    collection_id = f"{workspace}/applications"

    try:
        await artifact_manager.read(collection_id)
        logger.debug(f"Collection '{collection_id}' already exists")
    except Exception as collection_error:
        expected_error = (
            f"KeyError: \"Artifact with ID '{collection_id}' does not exist.\""
        )
        if str(collection_error).strip().endswith(expected_error):
            logger.info(f"Collection '{collection_id}' does not exist. Creating it.")

            collection_manifest = {
                "name": "Applications",
                "description": f"A collection of applications for workspace {workspace}",
            }

            try:
                collection = await artifact_manager.create(
                    type="collection",
                    alias="applications",
                    manifest=collection_manifest,
                )
                logger.info(f"Applications collection created with ID: {collection.id}")
            except Exception as e:
                raise RuntimeError(f"Failed to create applications collection: {e}")
        else:
            raise RuntimeError(
                f"Failed to check applications collection '{collection_id}': {collection_error}"
            )

    return collection_id


async def extract_and_validate_manifest(
    files: List[Dict[str, Any]],
    manifest_updates: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Extract and validate the manifest from a list of files.

    Args:
        files: List of file dictionaries
        manifest_updates: Optional dictionary of fields to add/update in the manifest

    Returns:
        Validated deployment manifest

    Raises:
        ValueError: If manifest is missing or invalid
    """
    # Find and validate manifest file
    manifest_file = None
    for file in files:
        if file["name"].lower() == "manifest.yaml":
            manifest_file = file
            break

    if not manifest_file:
        raise ValueError(
            "No manifest file found in files list. Expected 'manifest.yaml'"
        )

    # Load the manifest content
    manifest_content = manifest_file["content"]
    if manifest_file["type"] == "base64":
        # Remove `data:...` prefix if present and decode base64
        if manifest_content.startswith("data:"):
            manifest_content = manifest_content.split(",")[1]
        manifest_content = base64.b64decode(manifest_content).decode("utf-8")

    try:
        deployment_manifest = yaml.safe_load(manifest_content)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in manifest file: {e}")

    # Apply any manifest updates
    if manifest_updates:
        deployment_manifest.update(manifest_updates)

    return deployment_manifest


def validate_artifact_id(
    deployment_manifest: Dict[str, Any],
    workspace: str,
    artifact_id: Optional[str] = None,
) -> str:
    """
    Validate and normalize the artifact ID.

    Args:
        deployment_manifest: The deployment manifest
        workspace: Hypha workspace identifier
        artifact_id: Optional artifact ID for updates

    Returns:
        Full normalized artifact ID

    Raises:
        ValueError: If artifact ID format is incorrect
    """
    if artifact_id is None:
        if "id" not in deployment_manifest:
            raise ValueError(
                "No artifact_id provided and no 'id' field found in manifest"
            )
        alias = deployment_manifest["id"]

        # Validate alias format
        invalid = any(
            [
                not alias.islower(),
                "_" in alias,
                "/" in alias,
                not alias.replace("-", "_").isidentifier(),
            ]
        )
        if invalid:
            raise ValueError(
                f"Invalid artifact alias: '{alias}'. Please use lowercase letters, numbers, and hyphens only."
            )

        full_artifact_id = f"{workspace}/{alias}"
    else:
        # Use provided artifact_id, ensure it's fully qualified
        if "/" not in artifact_id:
            full_artifact_id = f"{workspace}/{artifact_id}"
        else:
            full_artifact_id = artifact_id

        # Ensure artifact belongs to workspace
        if not full_artifact_id.startswith(f"{workspace}/"):
            raise ValueError(
                f"Artifact ID '{full_artifact_id}' does not belong to workspace '{workspace}'"
            )

    return full_artifact_id


async def get_or_create_artifact(
    artifact_manager: ObjectProxy,
    full_artifact_id: str,
    workspace: str,
    deployment_manifest: Dict[str, Any],
    artifact_type: str = "application",
    logger: Optional[logging.Logger] = None,
) -> ObjectProxy:
    """
    Get an existing artifact or create a new one.

    Args:
        artifact_manager: Hypha artifact manager service instance
        full_artifact_id: Full normalized artifact ID (format: workspace/name)
        workspace: Hypha workspace identifier
        deployment_manifest: The deployment manifest
        artifact_type: The type of artifact to create
        logger: Optional logger instance for debugging

    Returns:
        The artifact object

    Raises:
        RuntimeError: If artifact creation fails
        ValueError: If the full_artifact_id doesn't match the workspace
    """
    if logger is None:
        logger = create_logger("ArtifactUtils")

    # Extract workspace from full_artifact_id
    if "/" not in full_artifact_id:
        raise ValueError(f"Invalid artifact ID format: {full_artifact_id}, expected 'workspace/name'")

    artifact_workspace = full_artifact_id.split("/")[0]
    if workspace != artifact_workspace:
        logger.warning(
            f"Workspace mismatch: provided workspace '{workspace}' doesn't match "
            f"artifact ID workspace '{artifact_workspace}'"
        )

    # Use workspace from full_artifact_id for consistency
    collection_id = f"{artifact_workspace}/applications"

    # Check if artifact exists and handle collection placement
    artifact = None
    try:
        # Check if artifact already exists
        logger.debug(f"Checking if artifact '{full_artifact_id}' exists...")
        existing_artifact = await artifact_manager.read(full_artifact_id)

        # Check if artifact is in the correct collection
        current_parent_id = getattr(existing_artifact, "parent_id", None)
        if current_parent_id != collection_id:
            logger.info(
                f"Artifact '{full_artifact_id}' exists but is in wrong collection "
                f"(current: {current_parent_id}, expected: {collection_id}). "
                "Deleting and recreating..."
            )
            # Delete the existing artifact
            await artifact_manager.delete(artifact_id=full_artifact_id)
            # Will create new one below
            existing_artifact = None

        if existing_artifact:
            # Edit existing artifact
            logger.debug(f"Editing existing artifact '{full_artifact_id}'...")
            artifact = await artifact_manager.edit(
                artifact_id=full_artifact_id,
                manifest=deployment_manifest,
                type=artifact_type,
                stage=True,
            )
            logger.debug(f"Successfully edited existing artifact '{full_artifact_id}'")
    except Exception as e:
        # Artifact doesn't exist or read failed
        logger.debug(
            f"Artifact '{full_artifact_id}' does not exist or read failed: {e}"
        )

    # Create new artifact if it doesn't exist or was deleted
    if artifact is None:
        try:
            logger.debug(f"Creating new artifact '{full_artifact_id}'...")
            artifact = await artifact_manager.create(
                parent_id=collection_id,
                type=artifact_type,
                alias=full_artifact_id,
                manifest=deployment_manifest,
                stage=True,
            )
            logger.debug(f"Successfully created new artifact '{full_artifact_id}'")
        except Exception as e:
            raise RuntimeError(f"Failed to create artifact '{full_artifact_id}': {e}")

    return artifact


async def upload_file_to_artifact(
    artifact_manager: ObjectProxy,
    artifact_id: str,
    file_name: str,
    file_content: Union[str, bytes],
    file_type: str = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Upload a file to an artifact.

    Args:
        artifact_manager: Hypha artifact manager service instance
        artifact_id: The artifact ID
        file_name: The file name/path
        file_content: The file content (str for text, str for base64, or raw bytes)
        file_type: The file type ('text', 'base64', or None for raw binary)
        logger: Optional logger instance for debugging

    Raises:
        ValueError: If file type is unsupported
        RuntimeError: If file upload fails
    """
    if logger is None:
        logger = create_logger("ArtifactUtils")

    logger.debug(f"Uploading file '{file_name}' to artifact...")

    # Get upload URL
    try:
        upload_url = await artifact_manager.put_file(
            artifact_id, file_path=file_name
        )
    except Exception as e:
        raise RuntimeError(f"Failed to get upload URL for '{file_name}': {e}")

    # Prepare content for upload
    if isinstance(file_content, bytes):
        # Direct binary content
        upload_data = file_content
        content_type = "binary"
    elif file_type == "text" or (file_type is None and isinstance(file_content, str)):
        # Text content
        upload_data = file_content
        content_type = "text"
    elif file_type == "base64":
        # Decode base64 content for binary files
        if isinstance(file_content, str):
            if file_content.startswith("data:"):
                file_content = file_content.split(",")[1]
            upload_data = base64.b64decode(file_content)
            content_type = "binary"
        else:
            raise ValueError("Base64 content must be a string")
    else:
        raise ValueError(
            f"Unsupported file type '{file_type}'. Expected 'text', 'base64', or None for raw binary"
        )

    # Upload the file with timeout
    upload_timeout = httpx.Timeout(30.0)
    try:
        async with httpx.AsyncClient(timeout=upload_timeout) as client:
            if content_type == "text":
                response = await client.put(upload_url, data=upload_data)
            else:
                response = await client.put(upload_url, content=upload_data)
            response.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Failed to upload file '{file_name}': {e}")


async def commit_artifact(
    artifact_manager: ObjectProxy,
    artifact_id: str,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Commit an artifact.

    Args:
        artifact_manager: Hypha artifact manager service instance
        artifact_id: The artifact ID
        logger: Optional logger instance for debugging

    Raises:
        RuntimeError: If artifact commit fails
    """
    if logger is None:
        logger = create_logger("ArtifactUtils")

    try:
        await artifact_manager.commit(artifact_id=artifact_id)
    except Exception as e:
        raise RuntimeError(f"Failed to commit artifact '{artifact_id}': {e}")

    logger.info(f"Successfully committed artifact '{artifact_id}'")


async def create_application_from_files(
    artifact_manager: ObjectProxy,
    files: List[Dict[str, Any]],
    workspace: str,
    artifact_id: Optional[str] = None,
    manifest_updates: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
) -> str:
    """
    Create or update a Hypha artifact from a list of files.

    This utility function handles the common pattern of creating artifacts in Hypha
    from a list of files, including manifest validation, collection management,
    file uploads, and artifact commitment.

    Args:
        artifact_manager: Hypha artifact manager service instance
        files: List of file dictionaries with keys:
               - 'name': relative file path (str)
               - 'content': file content (str or bytes)
               - 'type': 'text' for text files, 'base64' for binary files
        workspace: Hypha workspace identifier
        artifact_id: Optional artifact ID for updates. If None, extracts from manifest
        manifest_updates: Optional dictionary of fields to add/update in the manifest
        logger: Optional logger instance for debugging

    Returns:
        The artifact ID of the created or updated artifact

    Raises:
        ValueError: If manifest is missing, invalid, or artifact ID format is incorrect
        RuntimeError: If artifact creation, file upload, or commit fails

    Example:
        files = [
            {'name': 'manifest.yaml', 'content': manifest_yaml, 'type': 'text'},
            {'name': 'app.py', 'content': app_code, 'type': 'text'},
            {'name': 'data.bin', 'content': base64_data, 'type': 'base64'}
        ]
        artifact_id = await create_application_from_files(
            artifact_manager=am,
            files=files,
            workspace="my-workspace",
            manifest_updates={"created_by": "user123"}
        )
    """
    if logger is None:
        logger = create_logger("ArtifactUtils")

    # Extract and validate manifest
    deployment_manifest = await extract_and_validate_manifest(
        files=files, manifest_updates=manifest_updates
    )

    # Validate and normalize artifact ID
    full_artifact_id = validate_artifact_id(
        deployment_manifest=deployment_manifest,
        workspace=workspace,
        artifact_id=artifact_id,
    )

    # Ensure applications collection exists
    await ensure_applications_collection(
        artifact_manager=artifact_manager,
        workspace=workspace,
        logger=logger,
    )

    # Get or create artifact
    artifact = await get_or_create_artifact(
        artifact_manager=artifact_manager,
        full_artifact_id=full_artifact_id,
        workspace=workspace,
        deployment_manifest=deployment_manifest,
        logger=logger,
    )

    # Upload all files
    for file in files:
        file_name = file["name"]
        file_content = file["content"]
        file_type = file["type"]

        await upload_file_to_artifact(
            artifact_manager=artifact_manager,
            artifact_id=artifact.id,
            file_name=file_name,
            file_content=file_content,
            file_type=file_type,
            logger=logger,
        )

    # Commit the artifact
    await commit_artifact(
        artifact_manager=artifact_manager,
        artifact_id=artifact.id,
        logger=logger,
    )

    return artifact.id
