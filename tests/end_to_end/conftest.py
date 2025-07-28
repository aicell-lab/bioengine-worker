"""
End-to-end test configuration for BioEngine Worker.

Provides specialized fixtures for comprehensive testing including worker
initialization, application deployment, and service interaction patterns.

Supports both local worker instances and connecting to running workers
via USE_RUNNING_WORKER environment variable.
"""

import os
from typing import Dict, List

import pytest
import pytest_asyncio

from bioengine_worker.worker import BioEngineWorker

# Configuration for test execution mode
# When True, tests connect to an already running worker instead of starting their own
USE_RUNNING_WORKER = os.getenv("USE_RUNNING_WORKER", "False").lower() == "true"


@pytest.fixture(scope="session")
def worker_mode() -> str:
    """Return 'single-machine' mode for local Ray cluster testing."""
    return "single-machine"


# Test application configurations for automatic startup deployment
@pytest.fixture(scope="session")
def startup_applications() -> List[Dict]:
    """
    Define test applications for automatic deployment during worker startup.
    
    Returns list with demo-app for basic functionality testing.
    """
    return [
        {
            "artifact_id": "demo-app",
            "application_id": "demo-app",
        },
    ]


@pytest.fixture(scope="session")
def monitoring_interval_seconds() -> int:
    """Return 10-second monitoring interval for responsive test feedback."""
    return 10


@pytest.fixture(scope="session")
def dashboard_url() -> str:
    """Return BioEngine dashboard URL for worker integration."""
    return "https://bioimage.io/#/bioengine"


@pytest.fixture(scope="session")
def graceful_shutdown_timeout() -> int:
    """Return 60-second timeout for graceful worker shutdown."""
    return 60


@pytest.fixture(scope="session")
def application_check_timeout() -> int:
    """Return 30-second timeout for application readiness checks."""
    return 30


@pytest.fixture(scope="session")
def num_cpus() -> int:
    """Return 4 CPU cores for Ray cluster head node."""
    return 4


@pytest.fixture(scope="session")
def num_gpus() -> int:
    """Return 0 GPUs for testing (no GPU required)."""
    return 0


@pytest.fixture(scope="session")
def memory_in_gb() -> int:
    """Return 4GB memory allocation for worker."""
    return 4


@pytest_asyncio.fixture(scope="function")
async def bioengine_worker_service_id(
    worker_mode,
    cache_dir,
    data_dir,
    startup_applications,
    monitoring_interval_seconds,
    server_url,
    hypha_token,
    session_id,
    num_cpus,
    num_gpus,
    memory_in_gb,
    dashboard_url,
    graceful_shutdown_timeout,
    hypha_client,
):
    """
    Create BioEngine worker instance and return service ID.
    
    Initializes worker with startup applications and manages lifecycle.
    Automatically starts worker and cleans up after test completion.
    """
    # Initialize the BioEngine worker with startup applications
    bioengine_worker = BioEngineWorker(
        mode=worker_mode,
        admin_users=None,
        cache_dir=cache_dir,
        data_dir=data_dir,
        startup_applications=startup_applications,
        monitoring_interval_seconds=monitoring_interval_seconds,
        server_url=server_url,
        workspace=None,
        token=hypha_token,
        client_id=f"bioengine_test_worker_{session_id}",
        ray_cluster_config={
            "head_num_cpus": num_cpus,
            "head_num_gpus": num_gpus,
            "head_memory_in_gb": memory_in_gb,
        },
        dashboard_url=dashboard_url,
        log_file="off",
        debug=True,
        graceful_shutdown_timeout=graceful_shutdown_timeout,
    )

    try:
        # Start the worker
        await bioengine_worker.start(blocking=False)

        # Return the worker service for use in tests
        yield bioengine_worker.full_service_id

    finally:
        # Cleanup after all tests are done
        if bioengine_worker:
            await bioengine_worker._stop(blocking=True)


if USE_RUNNING_WORKER:

    @pytest_asyncio.fixture(scope="function")
    async def bioengine_worker_service(hypha_client, hypha_workspace):
        """Connect to existing BioEngine worker service."""
        # Get the BioEngine worker service
        bioengine_worker_service_id = f"{hypha_workspace}/bioengine-worker"
        bioengine_worker_service = await hypha_client.get_service(
            bioengine_worker_service_id
        )

        # Return the worker service for use in tests
        return bioengine_worker_service

else:

    @pytest_asyncio.fixture(scope="function")
    async def bioengine_worker_service(hypha_client, bioengine_worker_service_id):
        """Get BioEngine worker service from created worker instance."""
        # Get the BioEngine worker service
        bioengine_worker_service = await hypha_client.get_service(
            bioengine_worker_service_id
        )

        # Return the worker service for use in tests
        return bioengine_worker_service


@pytest.fixture(scope="function")
def bioengine_worker_workspace(bioengine_worker_service_id) -> str:
    """Extract workspace from worker service ID."""
    return bioengine_worker_service_id.split("/")[0]


@pytest.fixture(scope="function")
def bioengine_worker_client_id(bioengine_worker_service_id) -> str:
    """Extract client ID from worker service ID."""
    return bioengine_worker_service_id.split("/")[1].split(":")[0]
