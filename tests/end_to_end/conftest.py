"""
End-to-end test configuration for BioEngine Worker.

Provides specialized fixtures for end-to-end testing including BioEngine Worker
initialization and test application configurations.
"""

import asyncio
import os
import time
from pathlib import Path
from typing import Dict, List

import pytest

# Test configuration constants
WORKER_READY_TIMEOUT = 120  # 2 minutes for worker to become ready
CLEANUP_TIMEOUT = 60  # 1 minute for cleanup operations


@pytest.fixture(scope="session")
def startup_applications() -> List[Dict]:
    """
    Define test applications for startup deployment validation.

    Provides a consistent set of test applications used across all
    end-to-end tests to validate worker functionality.

    Returns:
        List of application configurations for testing including:
        - Simple demo application
        - Complex composition application with deployment parameters
    """
    return [
        {
            "artifact_id": "demo-app",
            "application_id": "demo-app",
        },
        {
            "artifact_id": "composition-app",
            "application_id": "composition-app",
            "deployment_kwargs": {
                "CompositionDeployment": {"demo_input": "Hello World!"},
                "Deployment2": {"start_number": 10},
            },
        },
    ]


@pytest.fixture(scope="function")  # Changed from session to function scope
async def bioengine_worker(
    workspace_folder: Path,
    cache_dir: Path,
    data_dir: Path,
    startup_applications: List[Dict],
    server_url: str,
    hypha_token: str,
):
    """
    Create and manage a BioEngine worker instance for testing.

    This function-scoped fixture provides a BioEngine worker instance
    for each test function. While this is less efficient than session scope,
    it avoids async event loop conflicts and ensures test isolation.

    Args:
        hypha_token: Authentication token for Hypha server
        cache_dir: Test-specific cache directory
        startup_applications: Applications to deploy at startup

    Returns:
        BioEngineWorker instance configuration (not started)

    Note:
        The worker is configured but not started. Individual tests should
        start the worker as needed and handle cleanup.
    """
    os.environ["BIOENGINE_WORKER_LOCAL_ARTIFACT_PATH"] = str(workspace_folder / "tests")
    
    # Import BioEngineWorker here to avoid Ray initialization during test collection
    from bioengine_worker.worker import BioEngineWorker

    bioengine_worker = BioEngineWorker(
        mode="single-machine",
        admin_users=None,
        cache_dir=cache_dir,
        data_dir=data_dir,
        startup_applications=startup_applications,
        monitoring_interval_seconds=10,
        server_url=server_url,
        workspace=None,
        token=hypha_token,
        client_id=f"test_worker_{int(time.time())}",
        ray_cluster_config={
            "head_num_cpus": 6,
            "head_num_gpus": 0,
            "head_memory_in_gb": 8,
        },
        dashboard_url="https://bioimage.io/#/bioengine",
        log_file=None,
        debug=False,
    )

    # Start the worker in non-blocking mode
    await bioengine_worker.start(block=False)

    try:
        yield bioengine_worker
    finally:
        try:
            # Send stop signal to the worker
            await bioengine_worker.stop()

            if not bioengine_worker._shutdown_event.is_set():
                # Wait for shutdown event to complete if it exists
                # This ensures all cleanup operations are finished
                await asyncio.wait_for(bioengine_worker._shutdown_event.wait(), timeout=CLEANUP_TIMEOUT)
                await asyncio.sleep(0.1)  # Allow time for cleanup tasks to finish
        except Exception as e:
            # Log cleanup errors but don't fail tests
            print(f"Warning: BioEngine worker cleanup failed: {e}")

        # Ensure cache directory is removed
        os.rmdir(cache_dir, ignore_errors=True)

# Utility fixtures for test configuration
@pytest.fixture
def worker_ready_timeout():
    """Timeout for worker readiness checks."""
    return 60  # 1 minute for individual test operations


@pytest.fixture
def application_check_timeout():
    """Timeout for application connectivity checks."""
    return 30  # 30 seconds for application ping operations


@pytest.fixture
def test_timeout():
    """Overall test timeout for complex operations."""
    return 180  # 3 minutes for complete test workflows
