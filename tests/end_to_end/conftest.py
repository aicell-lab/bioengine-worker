"""
End-to-end test configuration for BioEngine Worker.

Provides specialized fixtures for end-to-end testing including BioEngine Worker
initialization and test application configurations.
"""

import asyncio
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List

import pytest


# Utility fixtures for test configuration
@pytest.fixture
def worker_ready_timeout():
    """Timeout for worker readiness checks."""
    return 60  # 1 minute for individual test operations


@pytest.fixture
def worker_cleanup_timeout():
    """Timeout for worker cleanup operations."""
    return 60  # 1 minute for individual test operations


@pytest.fixture
def application_check_timeout():
    """Timeout for application connectivity checks."""
    return 30  # 30 seconds for application ping operations


# Test application configurations for startup deployments
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


# Test fixture for BioEngine Worker instance
@pytest.fixture(scope="function")
async def bioengine_worker(
    workspace_folder: Path,
    cache_dir: Path,
    data_dir: Path,
    startup_applications: List[Dict],
    server_url: str,
    hypha_token: str,
    worker_cleanup_timeout: int,
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
            await bioengine_worker.stop(context=bioengine_worker._admin_context)

            if not bioengine_worker._shutdown_event.is_set():
                # Wait for shutdown event to complete if it exists
                # This ensures all cleanup operations are finished
                await asyncio.wait_for(
                    bioengine_worker._shutdown_event.wait(),
                    timeout=worker_cleanup_timeout,
                )
                await asyncio.sleep(0.1)  # Allow time for cleanup tasks to finish
        except Exception as e:
            # Log cleanup errors but don't fail tests
            print(f"\n⚠️  Warning: BioEngine worker cleanup failed: {e}")

        # Ensure cache directory is removed
        try:
            shutil.rmtree(str(cache_dir))
        except Exception as e:
            print(f"\n⚠️  Warning: Could not remove cache directory: {e}")
