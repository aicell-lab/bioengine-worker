"""
End-to-end tests for BioEngine Worker AppsManager component.

This module tests the AppsManager functionality through the Hypha service API,
including application deployment, undeployment, startup applications, WebSocket services,
peer connections, artifact management, and cleanup operations.
"""

import asyncio
import base64
import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import pytest
import yaml
from hypha_rpc.rpc import ObjectProxy, RemoteService


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
async def test_startup_application(
    worker_mode: str,
    bioengine_worker_service: ObjectProxy,
    startup_applications: List[Dict],
):
    """
    Test that all startup applications are properly deployed.

    This test validates the AppsManager status reporting and startup application deployment:
    1. Retrieves comprehensive worker status including AppsManager state
    2. Validates "bioengine_apps" field structure and content
    3. Checks that startup applications are properly deployed and healthy
    4. Verifies application metadata, resource allocation, and service registration
    5. Ensures deployment configuration matches expected startup specifications

    Expected AppsManager Status Structure:
    - bioengine_apps: Dict[application_id, application_info] where each application_info contains:
      - display_name, description, artifact_id, version
      - start_time, status (RUNNING/HEALTHY/etc), message
      - deployments: Dict of deployment status and replica states
      - resource allocation: application_resources, deployment_options
      - service_ids: WebSocket and WebRTC service endpoints
      - access control: authorized_users, last_updated_by
      - available_methods: List of exposed application methods
    """
    if worker_mode == "external-cluster":
        pytest.skip(
            "Startup applications are disabled in external cluster mode. Skipping test."
        )

    # Ensure at least one startup application is configured
    assert (
        startup_applications and len(startup_applications) > 0
    ), "No startup applications configured for this test. Please define at least one in the fixture."

    # Get comprehensive worker status
    status = await bioengine_worker_service.get_status()

    # Validate bioengine_apps field exists and is properly structured
    assert (
        "bioengine_apps" in status
    ), "Worker status should contain 'bioengine_apps' field"
    apps_status = status["bioengine_apps"]
    assert isinstance(apps_status, dict), "bioengine_apps should be a dictionary"

    # Assert that applications are deployed based on startup configuration
    expected_app_count = len(startup_applications)
    assert (
        len(apps_status) > 0
    ), f"Expected {expected_app_count} startup applications to be deployed, but found {len(apps_status)} applications"

    # Validate each deployed application's status structure
    for application_id, app_info in apps_status.items():
        assert isinstance(
            application_id, str
        ), f"Application ID '{application_id}' should be a string"
        assert isinstance(
            app_info, dict
        ), f"Application info for '{application_id}' should be a dictionary"

        # Required application metadata fields
        required_fields = [
            "display_name",
            "description",
            "artifact_id",
            "version",
            "status",
            "message",
            "deployments",
            "deployment_options",
            "application_resources",
            "authorized_users",
            "available_methods",
            "service_ids",
            "last_updated_by",
        ]

        for field in required_fields:
            assert (
                field in app_info
            ), f"Application '{application_id}' should contain '{field}' field"

        # Validate field types and values
        assert isinstance(
            app_info["display_name"], str
        ), f"display_name should be a string for '{application_id}'"
        assert isinstance(
            app_info["description"], str
        ), f"description should be a string for '{application_id}'"
        assert isinstance(
            app_info["artifact_id"], str
        ), f"artifact_id should be a string for '{application_id}'"
        assert isinstance(
            app_info["version"], str
        ), f"version should be a string for '{application_id}'"
        assert isinstance(
            app_info["status"], str
        ), f"status should be a string for '{application_id}'"
        assert isinstance(
            app_info["message"], str
        ), f"message should be a string for '{application_id}'"
        assert isinstance(
            app_info["deployments"], dict
        ), f"deployments should be a dictionary for '{application_id}'"
        assert isinstance(
            app_info["deployment_options"], dict
        ), f"deployment_options should be a dictionary for '{application_id}'"
        assert isinstance(
            app_info["application_resources"], dict
        ), f"application_resources should be a dictionary for '{application_id}'"
        assert isinstance(
            app_info["authorized_users"], list
        ), f"authorized_users should be a list for '{application_id}'"
        assert isinstance(
            app_info["available_methods"], list
        ), f"available_methods should be a list for '{application_id}'"
        assert isinstance(
            app_info["service_ids"], list
        ), f"service_ids should be a list for '{application_id}'"

        # Validate application status is in expected states
        valid_statuses = [
            "NOT_STARTED",
            "DEPLOYING",
            "DEPLOY_FAILED",
            "RUNNING",
            "UNHEALTHY",
            "DELETING",
        ]
        assert (
            app_info["status"] in valid_statuses
        ), f"Application status '{app_info['status']}' should be one of {valid_statuses}"

        # Validate start_time if present (can be None for failed deployments)
        if "start_time" in app_info and app_info["start_time"] is not None:
            assert isinstance(
                app_info["start_time"], (int, float)
            ), f"start_time should be a number for '{application_id}'"
            assert (
                app_info["start_time"] > 0
            ), f"start_time should be positive for '{application_id}'"

        # For healthy applications, check deployment details
        if app_info["status"] == "RUNNING":
            # Should have deployments
            assert (
                len(app_info["deployments"]) > 0
            ), f"Running application '{application_id}' should have active deployments"

            # Validate each deployment
            for deployment_name, deployment_info in app_info["deployments"].items():
                assert isinstance(
                    deployment_name, str
                ), f"Deployment name should be a string in '{application_id}'"
                assert isinstance(
                    deployment_info, dict
                ), f"Deployment info should be a dictionary in '{application_id}'"

                # Required deployment fields
                deployment_fields = ["status", "message", "replica_states"]
                for field in deployment_fields:
                    assert (
                        field in deployment_info
                    ), f"Deployment '{deployment_name}' should contain '{field}' field"

                # Validate deployment status
                valid_deployment_statuses = [
                    "UPDATING",
                    "HEALTHY",
                    "UNHEALTHY",
                    "UPSCALING",
                    "DOWNSCALING",
                ]
                assert (
                    deployment_info["status"] in valid_deployment_statuses
                ), f"Deployment status '{deployment_info['status']}' should be one of {valid_deployment_statuses}"

                # Validate replica states
                assert isinstance(
                    deployment_info["replica_states"], dict
                ), f"replica_states should be a dictionary for deployment '{deployment_name}'"

            # Should have service IDs for running applications
            assert (
                len(app_info["service_ids"]) > 0
            ), f"Running application '{application_id}' should have service IDs"

            # Validate service ID structure
            for service_info in app_info["service_ids"]:
                assert isinstance(
                    service_info, dict
                ), f"Service info should be a dictionary for '{application_id}'"
                assert (
                    "websocket_service_id" in service_info
                ), f"Service info should contain websocket_service_id for '{application_id}'"
                assert (
                    "webrtc_service_id" in service_info
                ), f"Service info should contain webrtc_service_id for '{application_id}'"

        # Validate resource allocation structure
        if app_info["application_resources"]:
            assert isinstance(
                app_info["application_resources"], dict
            ), f"application_resources should be a dictionary for '{application_id}'"
            # Common resource fields (may vary based on deployment)
            for resource_key, resource_value in app_info[
                "application_resources"
            ].items():
                assert isinstance(
                    resource_key, str
                ), f"Resource key should be a string in '{application_id}'"
                assert isinstance(
                    resource_value, (int, float, str)
                ), f"Resource value should be numeric or string in '{application_id}'"

        # Validate deployment options structure
        if app_info["deployment_options"]:
            assert isinstance(
                app_info["deployment_options"], dict
            ), f"deployment_options should be a dictionary for '{application_id}'"

        # Validate authorized users
        assert (
            len(app_info["authorized_users"]) > 0
        ), f"Application '{application_id}' should have authorized users"
        for user in app_info["authorized_users"]:
            assert isinstance(
                user, str
            ), f"Authorized user should be a string in '{application_id}'"

        # Validate available methods
        for method in app_info["available_methods"]:
            assert isinstance(
                method, str
            ), f"Available method should be a string in '{application_id}'"

    # Log summary of application status for debugging
    app_count = len(apps_status)
    running_count = sum(1 for app in apps_status.values() if app["status"] == "RUNNING")
    healthy_count = sum(
        1
        for app in apps_status.values()
        if app["status"] == "RUNNING"
        and any(dep["status"] == "HEALTHY" for dep in app["deployments"].values())
    )

    print(
        f"AppsManager Status Summary: {app_count} total applications, "
        f"{running_count} running, {healthy_count} healthy"
    )
    print(f"Expected startup applications: {expected_app_count}")

    # Validate startup applications deployment - ALL must be running and healthy

    # Assert that exactly the expected number of applications are deployed
    assert (
        app_count == expected_app_count
    ), f"Expected exactly {expected_app_count} startup applications, but found {app_count} total applications"

    # ALL startup applications must be running
    assert (
        running_count == expected_app_count
    ), f"Expected all {expected_app_count} startup applications to be running, but found only {running_count} running applications"

    # ALL startup applications must be healthy
    assert (
        healthy_count == expected_app_count
    ), f"Expected all {expected_app_count} startup applications to be healthy, but found only {healthy_count} healthy applications"

    # Validate that each configured startup application exists and is in perfect state
    startup_apps_validated = 0
    for startup_app in startup_applications:
        artifact_id = startup_app.get("artifact_id")
        if artifact_id:
            # Check if any deployed app has this artifact_id
            matching_apps = [
                app
                for app in apps_status.values()
                if app.get("artifact_id").endswith(artifact_id)
            ]
            assert (
                len(matching_apps) == 1
            ), f"Expected exactly one deployment for startup application with artifact_id '{artifact_id}', but found {len(matching_apps)}"

            app_info = matching_apps[0]

            # Check that the app is running
            assert (
                app_info.get("status") == "RUNNING"
            ), f"Startup application with artifact_id '{artifact_id}' is not running (status: {app_info.get('status')})"

            # Check that all deployments are healthy
            assert (
                len(app_info["deployments"]) > 0
            ), f"Startup application with artifact_id '{artifact_id}' has no deployments"

            for deployment_name, deployment_info in app_info["deployments"].items():
                assert (
                    deployment_info["status"] == "HEALTHY"
                ), f"Deployment '{deployment_name}' of startup application '{artifact_id}' is not healthy (status: {deployment_info['status']})"

            # Check that service IDs are properly configured
            assert (
                len(app_info["service_ids"]) > 0
            ), f"Startup application with artifact_id '{artifact_id}' has no service IDs configured"

            for service_info in app_info["service_ids"]:
                # WebSocket service must be present
                has_valid_service = service_info.get("websocket_service_id") is not None
                assert (
                    has_valid_service
                ), f"Startup application with artifact_id '{artifact_id}' has no valid service endpoints"

            # Check that the application has available methods
            assert (
                len(app_info["available_methods"]) > 0
            ), f"Startup application with artifact_id '{artifact_id}' has no available methods"

            # Check that start_time is set (indicating successful deployment)
            assert (
                app_info.get("start_time") is not None
            ), f"Startup application with artifact_id '{artifact_id}' has no start_time set"
            assert isinstance(
                app_info["start_time"], (int, float)
            ), f"Startup application with artifact_id '{artifact_id}' has invalid start_time type"
            assert (
                app_info["start_time"] > 0
            ), f"Startup application with artifact_id '{artifact_id}' has invalid start_time value"

            startup_apps_validated += 1

    # Ensure we validated all expected startup applications
    assert (
        startup_apps_validated == expected_app_count
    ), f"Expected to validate {expected_app_count} startup applications, but only validated {startup_apps_validated}"


@pytest.mark.asyncio
async def test_deploy_application_locally(
    monkeypatch: pytest.MonkeyPatch,
    tests_dir: Path,
    worker_mode: str,
    hypha_workspace: str,
    bioengine_worker_service: ObjectProxy,
):
    """
    Test deploying the 'demo-app' and 'composition-app' applications from local artifact path.
    """
    if worker_mode == "external-cluster":
        pytest.skip(
            "Startup applications are disabled in external cluster mode. Skipping test."
        )

    # Set environment variables for startup application deployment from local path
    monkeypatch.setenv("BIOENGINE_WORKER_LOCAL_ARTIFACT_PATH", str(tests_dir))
    assert os.getenv("BIOENGINE_WORKER_LOCAL_ARTIFACT_PATH") == str(tests_dir)

    # iterate over demo_app and composition_app directories
    # Note: the demo app is already deployed by startup_applications, but the deployment below will use a different application ID

    demo_artifact_id = f"{hypha_workspace}/demo-app"
    demo_app_config = {
        "artifact_id": demo_artifact_id,
        "num_cpus": 2,
        "num_gpus": 0,
        "memory": 1.2 * 1024**3,
    }  # Test random application ID generation, provide resource limits

    composition_artifact_id = f"{hypha_workspace}/composition-app"
    composition_app_config = {
        "artifact_id": composition_artifact_id,
        "application_id": f"composition-app",
        "deployment_kwargs": {
            "CompositionDeployment": {"demo_input": "Hello World!"},
            "Deployment2": {"start_number": 10},
        },
    }  # provide custom application id and deployment kwargs

    app_configs = [demo_app_config, composition_app_config]
    deployed_app_ids = []

    for app_config in app_configs:
        # Deploy the application
        application_id = await bioengine_worker_service.deploy_application(**app_config)
        deployed_app_ids.append(application_id)
        print(f"Deployed application: {application_id}")

    # Wait for both applications to finish deploying
    timeout = 30  # 30 seconds timeout
    poll_interval = 2  # Check every 2 seconds
    start_time = time.time()

    while time.time() - start_time < timeout:
        status = await bioengine_worker_service.get_status()
        bioengine_apps = status.get("bioengine_apps", {})

        # Check if all apps are no longer in DEPLOYING state
        all_deployed = True
        for app_id in deployed_app_ids:
            if app_id in bioengine_apps:
                app_status = bioengine_apps[app_id].get("status", "")
                if app_status in ["NOT_STARTED", "DEPLOYING"]:
                    all_deployed = False
                    break
            else:
                all_deployed = False
                break

        if all_deployed:
            break

        await asyncio.sleep(poll_interval)
    else:
        raise TimeoutError(
            f"Applications did not finish deploying within {timeout} seconds"
        )

    # Check that both apps are healthy and have running replicas
    status = await bioengine_worker_service.get_status()
    bioengine_apps = status.get("bioengine_apps", {})

    for app_id in deployed_app_ids:
        assert app_id in bioengine_apps, f"Application {app_id} not found in status"

        app_info = bioengine_apps[app_id]
        assert (
            app_info["status"] == "RUNNING"
        ), f"Application {app_id} is not running: {app_info['status']}"

        # Check deployments are healthy
        assert (
            len(app_info["deployments"]) > 0
        ), f"Application {app_id} should have active deployments"

        for deployment_name, deployment_info in app_info["deployments"].items():
            assert (
                deployment_info["status"] == "HEALTHY"
            ), f"Deployment {deployment_name} of app {app_id} is not healthy: {deployment_info['status']}"

            # Check replica states
            replica_states = deployment_info["replica_states"]
            assert (
                len(replica_states) > 0
            ), f"Deployment {deployment_name} of app {app_id} should have replicas"

            # Ensure at least one replica is running
            running_replicas = replica_states.get("RUNNING", 0)
            assert (
                running_replicas > 0
            ), f"Deployment {deployment_name} of app {app_id} should have at least one running replica"

        print(f"Application {app_id} is healthy with running replicas")

    # Does not need any cleanup because of function-scoped BioEngine worker


@pytest.mark.asyncio
async def test_deploy_application_from_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tests_dir: Path,
    worker_mode: str,
    test_id: str,
    hypha_workspace: str,
    bioengine_worker_service: ObjectProxy,
):
    """
    Test deploying the 'demo-app' and 'composition-app' applications from remote artifact.

    Note: The demo app is already deployed by startup_applications, deploying again will update the app
    """
    if worker_mode == "external-cluster":
        pytest.skip(
            "Startup applications are disabled in external cluster mode. Skipping test."
        )

    # Ensure BIOENGINE_WORKER_LOCAL_ARTIFACT_PATH is not set to avoid local deployment
    monkeypatch.delenv("BIOENGINE_WORKER_LOCAL_ARTIFACT_PATH", raising=False)
    assert os.getenv("BIOENGINE_WORKER_LOCAL_ARTIFACT_PATH") is None

    hyphen_test_id = test_id.replace("_", "-")

    demo_app_path = tests_dir / "demo_app"
    demo_artifact_id = f"{hypha_workspace}/demo-app-{hyphen_test_id}"
    demo_app_config = {
        "artifact_id": demo_artifact_id,
        "num_cpus": 2,
        "num_gpus": 0,
        "memory": 1.2 * 1024**3,
    }  # Test with resource limits

    composition_app_path = tests_dir / "composition_app"
    composition_artifact_id = f"{hypha_workspace}/composition-app-{hyphen_test_id}"
    composition_app_config = {
        "artifact_id": composition_artifact_id,
        "application_id": f"composition-app-{hyphen_test_id}",
        "deployment_kwargs": {
            "CompositionDeployment": {"demo_input": "Hello World!"},
            "Deployment2": {"start_number": 10},
        },
    }

    app_paths = [demo_app_path, composition_app_path]
    app_configs = [demo_app_config, composition_app_config]
    artifact_ids = [demo_artifact_id, composition_artifact_id]
    deployed_app_ids = []

    # Verify the test directories exist
    assert demo_app_path.exists(), f"Demo app directory not found: {demo_app_path}"
    assert (
        composition_app_path.exists()
    ), f"Composition app directory not found: {composition_app_path}"

    test_completed = False
    try:
        # Create artifacts first
        for app_path, artifact_id in zip(app_paths, artifact_ids):
            # Create file list from directory
            files, created_artifact_id = _create_file_list_from_directory(
                app_path, test_id, hypha_workspace
            )
            assert (
                created_artifact_id == artifact_id
            ), f"Artifact ID mismatch: expected {artifact_id}, got {created_artifact_id}"

            # Create the artifact
            result_artifact_id = await bioengine_worker_service.create_application(
                files=files
            )
            assert (
                result_artifact_id == artifact_id
            ), f"Created artifact ID should match expected: {artifact_id}"
            print(f"Created artifact: {artifact_id}")

        # Verify artifacts exist
        available_artifacts = await bioengine_worker_service.list_applications()
        for artifact_id in artifact_ids:
            assert (
                artifact_id in available_artifacts
            ), f"Artifact {artifact_id} should be listed in available artifacts"

        # Deploy applications from artifacts
        for app_config in app_configs:
            application_id = await bioengine_worker_service.deploy_application(
                **app_config
            )
            deployed_app_ids.append(application_id)
            print(f"Deployed application: {application_id}")

        # Wait for both applications to finish deploying
        timeout = 30  # 30 seconds timeout
        poll_interval = 2  # Check every 2 seconds
        start_time = time.time()

        while time.time() - start_time < timeout:
            status = await bioengine_worker_service.get_status()
            bioengine_apps = status.get("bioengine_apps", {})

            # Check if all apps are no longer in DEPLOYING state
            all_deployed = True
            for app_id in deployed_app_ids:
                if app_id in bioengine_apps:
                    app_status = bioengine_apps[app_id].get("status", "")
                    if app_status in ["NOT_STARTED", "DEPLOYING"]:
                        all_deployed = False
                        break
                else:
                    all_deployed = False
                    break

            if all_deployed:
                break

            await asyncio.sleep(poll_interval)
        else:
            raise TimeoutError(
                f"Applications did not finish deploying within {timeout} seconds"
            )

        # Check that both apps are healthy and have running replicas
        status = await bioengine_worker_service.get_status()
        bioengine_apps = status.get("bioengine_apps", {})

        for app_id in deployed_app_ids:
            assert app_id in bioengine_apps, f"Application {app_id} not found in status"

            app_info = bioengine_apps[app_id]
            assert (
                app_info["status"] == "RUNNING"
            ), f"Application {app_id} is not running: {app_info['status']}"

            # Check deployments are healthy
            assert (
                len(app_info["deployments"]) > 0
            ), f"Application {app_id} should have active deployments"

            for deployment_name, deployment_info in app_info["deployments"].items():
                assert (
                    deployment_info["status"] == "HEALTHY"
                ), f"Deployment {deployment_name} of app {app_id} is not healthy: {deployment_info['status']}"

                # Check replica states
                replica_states = deployment_info["replica_states"]
                assert (
                    len(replica_states) > 0
                ), f"Deployment {deployment_name} of app {app_id} should have replicas"

                # Ensure at least one replica is running
                running_replicas = replica_states.get("RUNNING", 0)
                assert (
                    running_replicas > 0
                ), f"Deployment {deployment_name} of app {app_id} should have at least one running replica"

            print(f"Application {app_id} is healthy with running replicas")

        # Undeploy applications
        for app_id in deployed_app_ids:
            await bioengine_worker_service.undeploy_application(app_id)
            print(f"Undeployed application: {app_id}")

        test_completed = True

    finally:
        # Cleanup: Delete all created artifacts
        cleanup_errors = []
        for artifact_id in artifact_ids:
            try:
                await bioengine_worker_service.delete_application(
                    artifact_id=artifact_id
                )
                print(f"Deleted artifact: {artifact_id}")
            except Exception as e:
                cleanup_errors.append(f"Failed to cleanup artifact {artifact_id}: {e}")

        # Also cleanup any remaining deployed applications if test failed
        if not test_completed:
            for app_id in deployed_app_ids:
                try:
                    await bioengine_worker_service.undeploy_application(app_id)
                except Exception as e:
                    cleanup_errors.append(
                        f"Failed to cleanup application {app_id}: {e}"
                    )

        # Log cleanup errors but don't fail the test if cleanup fails
        if cleanup_errors:
            for error in cleanup_errors:
                print(f"Cleanup warning: {error}")


@pytest.mark.asyncio
async def test_call_demo_app_functions(
    monkeypatch: pytest.MonkeyPatch,
    tests_dir: Path,
    bioengine_worker_service: ObjectProxy,
    hypha_client: RemoteService,
):
    """
    Test calling functions of the deployed demo application.

    Exposed methods:
    - `ping`
    - `ascii_art`
    """
    # Set environment variables for startup application deployment from local path
    monkeypatch.setenv("BIOENGINE_WORKER_LOCAL_ARTIFACT_PATH", str(tests_dir))
    assert os.getenv("BIOENGINE_WORKER_LOCAL_ARTIFACT_PATH") == str(tests_dir)

    # Deploy the demo-app with apps_manager.deploy_application from local path

    # Get the service ID from the worker status

    # Get the websocket service using hypha_client

    # Call the application functions using the WebSocket service

    # Get the peer connection

    # Get the service using the peer connection instead of hypha_client

    # Call the application functions using the peer connection service

    # TODO: Implement test logic
    raise NotImplementedError


@pytest.mark.asyncio
async def test_call_composition_app_functions(
    monkeypatch: pytest.MonkeyPatch,
    tests_dir: Path,
    bioengine_worker_service: ObjectProxy,
    hypha_client: RemoteService,
):
    """
    Test calling functions of the deployed composition application.

    Exposed methods:
    - `ping`
    - `calculate_result`
    """
    # Set environment variables for startup application deployment from local path
    monkeypatch.setenv("BIOENGINE_WORKER_LOCAL_ARTIFACT_PATH", str(tests_dir))
    assert os.getenv("BIOENGINE_WORKER_LOCAL_ARTIFACT_PATH") == str(tests_dir)

    # Deploy the composition-app with apps_manager.deploy_application from local path

    # Get the service ID from the worker status

    # Get the websocket service using hypha_client

    # Call the application functions using the WebSocket service

    # Get the peer connection

    # Get the service using the peer connection instead of hypha_client

    # Call the application functions using the peer connection service

    # TODO: Implement test logic
    raise NotImplementedError
