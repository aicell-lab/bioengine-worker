"""
End-to-end tests for BioEngine Worker AppsManager component.

This module tests the AppsManager functionality through the Hypha service API,
including application deployment, undeployment, startup applications, WebSocket services,
peer connections, artifact management, and cleanup operations.
"""

import base64
import warnings
from pathlib import Path
from typing import List, Tuple

import pytest
import yaml
from hypha_rpc.rpc import ObjectProxy


def _create_file_list_from_directory(
    directory_path: Path, test_id: str, hypha_workspace: str
) -> Tuple[List[dict], str]:
    """
    Convert a local directory to a list of file dictionaries for create_artifact,
    automatically updating manifest files with unique test ID.

    Args:
        directory_path: Path to the directory containing app files
        test_id: Unique test ID to append to manifest id
        hypha_workspace: Hypha workspace identifier to prepend to artifact ID

    Returns:
        Tuple of (list of file dictionaries with 'name', 'content', and 'type' keys, artifact_id)
    """
    files = []
    artifact_id = None

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
                try:
                    content = base64.b64encode(file_path.read_bytes()).decode("ascii")
                    file_type = "base64"
                except Exception as e:
                    pytest.fail(f"Failed to read file {relative_path}: {e}")
            except Exception as e:
                pytest.fail(f"Failed to read file {relative_path}: {e}")

            # Update manifest files with test ID
            if str(relative_path) == "manifest.yaml":
                try:
                    manifest = yaml.safe_load(content)
                    if not isinstance(manifest, dict):
                        pytest.fail(
                            f"Invalid manifest structure: expected dict, got {type(manifest)}"
                        )
                    if "id" not in manifest:
                        pytest.fail("Manifest missing required 'id' field")

                    hyphen_test_id = test_id.replace("_", "-")
                    artifact_id = f"{manifest['id']}-{hyphen_test_id}"
                    manifest["id"] = artifact_id
                    content = yaml.dump(manifest)
                except yaml.YAMLError as e:
                    pytest.fail(f"Failed to parse manifest YAML: {e}")
                except Exception as e:
                    pytest.fail(f"Failed to update manifest: {e}")

            files.append(
                {"name": str(relative_path), "content": content, "type": file_type}
            )

    if artifact_id is None:
        pytest.fail("No manifest.yaml file found in directory")

    artifact_id = f"{hypha_workspace}/{artifact_id}"

    return files, artifact_id


@pytest.mark.asyncio
async def test_create_and_delete_artifacts(
    bioengine_worker_service: ObjectProxy,
    tests_dir: Path,
    test_id: str,
    hypha_workspace: str,
):
    """
    Test creating and deleting artifacts from both demo-app and composition-app applications.

    Steps:
    - Create artifacts from both applications
    - Wait for artifact creation completion
    - Verify artifacts appear in available artifacts list
    - Check all files in artifact directories
    - Delete both artifacts using delete_artifact API
    - Verify artifacts are removed from available list
    - Confirm storage cleanup and accessibility removal
    """
    # Define paths to test applications
    demo_app_path = tests_dir / "demo_app"
    composition_app_path = tests_dir / "composition_app"

    # Verify the test directories exist
    assert demo_app_path.exists(), f"Demo app directory not found: {demo_app_path}"
    assert (
        composition_app_path.exists()
    ), f"Composition app directory not found: {composition_app_path}"

    # Create file lists from both directories
    demo_app_files, demo_artifact_id = _create_file_list_from_directory(
        demo_app_path, test_id, hypha_workspace
    )
    composition_app_files, composition_artifact_id = _create_file_list_from_directory(
        composition_app_path, test_id, hypha_workspace
    )

    # Verify we have files to upload
    assert len(demo_app_files) > 0, "Demo app directory should contain files"
    assert (
        len(composition_app_files) > 0
    ), "Composition app directory should contain files"

    # Verify artifact has manifest files
    demo_manifest = next(
        (f for f in demo_app_files if f["name"] == "manifest.yaml"), None
    )
    composition_manifest = next(
        (f for f in composition_app_files if f["name"] == "manifest.yaml"), None
    )

    # Ensure manifest files are present
    assert demo_manifest, "Demo app manifest not found"
    assert composition_manifest, "Composition app manifest not found"

    # Ensure manifest files are valid YAML
    try:
        yaml.safe_load(demo_manifest["content"])
    except yaml.YAMLError as e:
        pytest.fail(f"Invalid YAML in demo app manifest: {e}")

    try:
        yaml.safe_load(composition_manifest["content"])
    except yaml.YAMLError as e:
        pytest.fail(f"Invalid YAML in composition app manifest: {e}")

    # Verify the artifact IDs do not already exist
    existing_artifacts = await bioengine_worker_service.list_applications()
    assert (
        demo_artifact_id not in existing_artifacts
    ), f"Demo artifact ID {demo_artifact_id} already exists"
    assert (
        composition_artifact_id not in existing_artifacts
    ), f"Composition artifact ID {composition_artifact_id} already exists"

    test_completed = False
    try:
        # Create demo-app artifact
        created_demo_artifact_id = await bioengine_worker_service.create_application(
            files=demo_app_files
        )

        # Verify artifact creation returned the correct ID
        assert created_demo_artifact_id, "Demo artifact creation should return an ID"
        assert isinstance(
            created_demo_artifact_id, str
        ), "Artifact ID should be a string"
        assert (
            created_demo_artifact_id == demo_artifact_id
        ), "Created demo artifact ID should match manifest ID and workspace"

        # Create composition-app artifact
        created_composition_artifact_id = (
            await bioengine_worker_service.create_application(
                files=composition_app_files
            )
        )

        # Verify artifact creation returned the correct ID
        assert (
            created_composition_artifact_id
        ), "Composition artifact creation should return an ID"
        assert isinstance(
            created_composition_artifact_id, str
        ), "Artifact ID should be a string"
        assert (
            created_composition_artifact_id == composition_artifact_id
        ), "Created composition artifact ID should match manifest ID and workspace"

        # Verify artifacts exist
        available_artifacts = await bioengine_worker_service.list_applications()
        assert (
            demo_artifact_id in available_artifacts
        ), "Demo artifact should be listed in available artifacts"
        assert (
            composition_artifact_id in available_artifacts
        ), "Composition artifact should be listed in available artifacts"

        # Update demo-app artifact
        updated_demo_artifact_id = await bioengine_worker_service.create_application(
            artifact_id=demo_artifact_id, files=demo_app_files
        )
        assert (
            updated_demo_artifact_id == demo_artifact_id
        ), "Updated demo artifact ID should match original"

        # Update composition-app artifact
        updated_composition_artifact_id = (
            await bioengine_worker_service.create_application(
                artifact_id=composition_artifact_id, files=composition_app_files
            )
        )
        assert (
            updated_composition_artifact_id == composition_artifact_id
        ), "Updated composition artifact ID should match original"

        # Verify updated artifacts still exist
        available_artifacts = await bioengine_worker_service.list_applications()
        assert (
            demo_artifact_id in available_artifacts
        ), "Demo artifact should be listed in available artifacts"
        assert (
            composition_artifact_id in available_artifacts
        ), "Composition artifact should be listed in available artifacts"

        # Verify all files in demo-app artifact

        assert all(
            f["name"] in available_artifacts[demo_artifact_id] for f in demo_app_files
        ), "All demo app files should be listed in artifact files"

        # Verify all files in composition-app artifact
        assert all(
            f["name"] in available_artifacts[composition_artifact_id]
            for f in composition_app_files
        ), "All composition app files should be listed in artifact files"

        # Delete both artifacts
        await bioengine_worker_service.delete_application(artifact_id=demo_artifact_id)

        await bioengine_worker_service.delete_application(
            artifact_id=composition_artifact_id
        )

        # Verify artifacts no longer exist
        available_artifacts = await bioengine_worker_service.list_applications()
        assert (
            demo_artifact_id not in available_artifacts
        ), "Demo artifact should be removed from available artifacts"
        assert (
            composition_artifact_id not in available_artifacts
        ), "Composition artifact should be removed from available artifacts"

        test_completed = True

    finally:
        # Cleanup: Ensure artifacts are deleted even if test fails
        cleanup_errors = []
        if demo_artifact_id and not test_completed:
            try:
                await bioengine_worker_service.delete_application(
                    artifact_id=demo_artifact_id
                )
            except Exception as e:
                cleanup_errors.append(
                    f"Failed to cleanup demo artifact {demo_artifact_id}: {e}"
                )

        if composition_artifact_id and not test_completed:
            try:
                await bioengine_worker_service.delete_application(
                    artifact_id=composition_artifact_id
                )
            except Exception as e:
                cleanup_errors.append(
                    f"Failed to cleanup composition artifact {composition_artifact_id}: {e}"
                )

        # Log cleanup errors but don't fail the test if cleanup fails
        if cleanup_errors:
            for error in cleanup_errors:
                warnings.warn(error)


@pytest.mark.asyncio
async def test_startup_application_and_cleanup(bioengine_worker_service):
    """
    Test that the startup application 'demo-app' is properly deployed and cleanup operations.

    This test validates:
    1. Startup application configuration is processed
    2. The 'demo-app' is automatically deployed during worker startup
    3. Application status shows as running and healthy
    4. Required resources are allocated correctly
    5. Application endpoints are accessible
    6. Cleanup operation through cleanup_deployments API
    7. Complete resource deallocation and cleanup

    Steps:
    - Connect to Hypha server and get worker service
    - Check worker status to see deployed applications
    - Verify 'demo-app' is listed in active deployments
    - Check application health and resource allocation
    - Validate deployment configuration matches startup spec
    - Call cleanup_deployments to remove all applications
    - Verify all deployments are properly cleaned up
    """
    # TODO: Implement test logic
    raise NotImplementedError


@pytest.mark.asyncio
async def test_deploy_application_locally(bioengine_worker_service):
    """
    Test deploying the 'composition-app' application from local artifact path.

    This test validates:
    1. Local artifact deployment (BIOENGINE_WORKER_LOCAL_ARTIFACT_PATH is set)
    2. Application deployment through deploy_application API
    3. Successful application startup and health checks
    4. Resource allocation for the deployed application
    5. Service registration and accessibility

    Steps:
    - Connect to worker service
    - Call deploy_application with artifact_id="composition-app"
    - Wait for deployment completion
    - Verify application appears in worker status
    - Check application health and endpoints
    - Validate resource usage and allocation
    """
    # TODO: Implement test logic
    raise NotImplementedError


@pytest.mark.asyncio
async def test_call_composition_app_functions(bioengine_worker_service):
    """
    Test calling specific functions (calculate_result and ping) of the demo-app.

    This test validates:
    1. Service function discovery and access
    2. Remote function invocation through Hypha RPC
    3. Parameter passing and result retrieval
    4. Function execution in Ray Serve environment
    5. Response handling and error management

    Steps:
    - Connect to worker service
    - Get demo-app service reference
    - Call calculate_result function with test parameters
    - Verify calculation results and response format
    - Call ping function for connectivity testing
    - Check function execution timing and performance
    - Validate error handling for invalid parameters
    """
    # TODO: Implement test logic
    raise NotImplementedError


@pytest.mark.asyncio
async def test_undeploy_application(bioengine_worker_service):
    """
    Test undeploying the 'composition-app' application.

    This test validates:
    1. Application undeployment through undeploy_application API
    2. Graceful shutdown of application services
    3. Resource cleanup and deallocation
    4. Removal from active deployments list
    5. Service deregistration from Hypha server

    Steps:
    - Ensure composition-app is deployed first
    - Call undeploy_application with application_id="composition-app"
    - Wait for undeployment completion
    - Verify application no longer appears in worker status
    - Check that resources are properly freed
    - Confirm service endpoints are no longer accessible
    """
    # TODO: Implement test logic
    raise NotImplementedError


@pytest.mark.asyncio
async def test_get_websocket_service(bioengine_worker_service):
    """
    Test accessing the WebSocket service of the startup application.

    This test validates:
    1. WebSocket service availability for demo-app
    2. WebSocket connection establishment
    3. Service endpoint discovery through Hypha
    4. Real-time communication capabilities
    5. WebSocket message handling and responses

    Steps:
    - Connect to worker service
    - Get demo-app service information
    - Locate WebSocket service endpoint
    - Establish WebSocket connection
    - Test basic message exchange
    - Verify connection stability and cleanup
    """
    # TODO: Implement test logic
    raise NotImplementedError


@pytest.mark.asyncio
async def test_get_peer_connection_websocket_service(bioengine_worker_service):
    """
    Test accessing peer connection and WebSocket service of the startup application.

    This test validates:
    1. Peer connection establishment for demo-app
    2. WebRTC peer connection setup and signaling
    3. Combined peer connection and WebSocket functionality
    4. Real-time data channels and communication
    5. Connection management and cleanup

    Steps:
    - Connect to worker service
    - Get demo-app service with peer connection support
    - Establish WebRTC peer connection
    - Set up WebSocket communication channel
    - Test bidirectional data exchange
    - Verify connection quality and performance
    - Clean up connections properly
    """
    # TODO: Implement test logic
    raise NotImplementedError
