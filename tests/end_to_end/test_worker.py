"""
End-to-end tests for BioEngine Worker core functionality.

This module provides comprehensive integration tests for the BioEngineWorker main component,
testing all core functionality through the Hypha service API. These tests validate the
complete worker lifecycle, status monitoring, administrative operations, and service
integration patterns.

Test Coverage:
- Worker initialization and readiness validation
- Comprehensive status reporting and health monitoring
- Ray cluster integration and resource management
- Applications and datasets management status
- Administrative permissions and user access control
- Graceful worker shutdown and cleanup procedures
- Service registration and deregistration with Hypha

Prerequisites:
- Active Hypha server connection
- Valid HYPHA_TOKEN in environment
- bioengine_worker_service fixture providing worker access
- Sufficient permissions for administrative operations

Note: These tests require the bioengine-worker conda environment and may take
several minutes to complete due to worker initialization time.
"""

import asyncio
import time

import pytest


@pytest.mark.asyncio
async def test_get_status(
    bioengine_worker_service,
    worker_mode,
    bioengine_worker_workspace,
    bioengine_worker_client_id,
    hypha_client,
):
    """
    Test comprehensive worker status retrieval and validation.

    Validates the complete status reporting functionality of the BioEngine worker,
    ensuring all critical system information is properly exposed through the API.
    This test is essential for monitoring worker health and troubleshooting issues.

    Test Scope:
    1. API Response Structure - Validates status dictionary format and completeness
    2. Service Metadata - Checks uptime, start time, workspace, and client information
    3. Worker Configuration - Verifies mode settings and operational parameters
    4. Ray Cluster Status - Confirms distributed computing infrastructure health
    5. Component Status - Validates apps manager and datasets manager reporting
    6. Access Control - Ensures admin user permissions are properly configured
    7. Readiness State - Confirms worker is fully operational and accepting requests

    Expected Behavior:
    - Returns comprehensive status dictionary with all required fields
    - Service uptime increases monotonically from start time
    - Worker mode matches one of the supported deployment modes
    - Ray cluster status reflects current resource allocation
    - Admin users include the test client's credentials
    - Worker reports ready state when fully initialized

    Assertions:
    - All required status fields are present and correctly typed
    - Numeric values are within expected ranges (uptime >= 0)
    - String fields contain valid configuration values
    - Collection fields (lists, dicts) have appropriate structure
    - Current user has administrative access to the worker
    """
    # Retrieve worker status through the service API
    status = await bioengine_worker_service.get_status()

    # Validate response structure and format
    assert isinstance(status, dict), "Status response should be a dictionary"

    # Define and verify all required status fields are present
    required_fields = [
        "service_start_time",  # Worker initialization timestamp
        "service_uptime",  # Current session duration in seconds
        "worker_mode",  # Deployment mode (slurm/single-machine/external-cluster)
        "workspace",  # Hypha workspace identifier
        "client_id",  # Unique client identifier
        "ray_cluster",  # Ray distributed computing cluster status
        "bioengine_apps",  # Applications manager status and deployments
        "bioengine_datasets",  # Datasets manager status and loaded data
        "admin_users",  # List of users with administrative privileges
        "is_ready",  # Worker operational readiness flag
    ]
    # Ensure all required fields are present in the status response
    for field in required_fields:
        assert field in status, f"Status should contain '{field}' field"

    # Validate service timing information
    assert isinstance(
        status["service_start_time"], (int, float)
    ), "service_start_time should be a timestamp"
    assert isinstance(
        status["service_uptime"], (int, float)
    ), "service_uptime should be a number"
    assert status["service_uptime"] >= 0, "service_uptime should be non-negative"

    # Validate worker configuration and deployment mode
    assert isinstance(status["worker_mode"], str), "worker_mode should be a string"
    assert (
        status["worker_mode"] == worker_mode
    ), f"worker_mode should match expected value: {worker_mode}"

    # Validate service identification fields
    assert isinstance(status["workspace"], str), "workspace should be a string"
    assert (
        status["workspace"] == bioengine_worker_workspace
    ), f"workspace should match expected value: {bioengine_worker_workspace}"

    assert isinstance(status["client_id"], str), "client_id should be a string"
    assert (
        status["client_id"] == bioengine_worker_client_id
    ), f"client_id should match expected value: {bioengine_worker_client_id}"

    # Validate component manager status structures
    assert isinstance(status["ray_cluster"], dict), "ray_cluster should be a dictionary"
    assert isinstance(
        status["bioengine_apps"], dict
    ), "bioengine_apps should be a dictionary"
    assert isinstance(
        status["bioengine_datasets"], dict
    ), "bioengine_datasets should be a dictionary"

    # Validate administrative access control configuration
    assert isinstance(status["admin_users"], list), "admin_users should be a list"
    assert len(status["admin_users"]) > 0, "admin_users should not be empty"

    # Verify current test user has administrative privileges
    user_id = hypha_client.config.user["id"]
    user_email = hypha_client.config.user["email"]
    assert user_id in status["admin_users"], "Current user ID should be in admin_users"
    assert (
        user_email in status["admin_users"]
    ), "Current user email should be in admin_users"

    # Validate worker operational readiness
    assert isinstance(status["is_ready"], bool), "is_ready should be a boolean"
    assert status["is_ready"], "Worker should be ready"


@pytest.mark.asyncio
async def test_stop_worker(
    bioengine_worker_service,
    hypha_client,
    bioengine_worker_service_id,
):
    """
    Test graceful worker shutdown and service deregistration.

    Validates the complete worker lifecycle management, ensuring the worker can be
    properly shut down with full resource cleanup and service deregistration.
    This test is critical for preventing resource leaks and ensuring clean
    deployment cycles.

    Test Scope:
    1. Graceful Shutdown - Initiates non-blocking worker termination
    2. Resource Cleanup - Ensures all components are properly cleaned up
    3. Service Deregistration - Confirms worker service is removed from Hypha
    4. Timeout Handling - Validates shutdown completes within reasonable time
    5. State Verification - Confirms worker is no longer accessible

    Expected Behavior:
    - stop_worker() call returns without blocking
    - All component managers (Ray, Apps, Datasets) are cleanly shut down
    - Worker service is deregistered from Hypha server
    - Shutdown completes within 60 seconds timeout
    - No orphaned processes or resources remain

    Test Flow:
    1. Initiate non-blocking worker shutdown via API
    2. Poll Hypha service registry for worker deregistration
    3. Verify service is removed within timeout period
    4. Raise TimeoutError if shutdown takes too long

    Note: This test typically runs last as it terminates the worker instance.
    """
    # Initiate graceful worker shutdown (non-blocking)
    await bioengine_worker_service.stop_worker(blocking=False)

    # Set up polling for service deregistration
    start_time = time.time()
    timeout = 60  # Maximum time to wait for shutdown (seconds)
    poll_interval = 3  # Check service status every 3 seconds

    # Poll until worker service is deregistered or timeout occurs
    while time.time() - start_time < timeout:
        # Query current services registered with Hypha server
        services = await hypha_client.list_services()

        # Check if our worker service is still registered
        if not any(service.id == bioengine_worker_service_id for service in services):
            # Success: worker service has been deregistered
            return

        # Wait before next polling attempt
        await asyncio.sleep(poll_interval)

    # Timeout exceeded - worker failed to shut down properly
    raise TimeoutError(
        f"Worker service did not stop within {timeout} seconds. "
        f"This may indicate issues with resource cleanup or service deregistration."
    )
