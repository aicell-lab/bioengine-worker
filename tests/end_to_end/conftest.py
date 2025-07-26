"""
End-to-end test configuration for BioEngine Worker.

Provides specialized fixtures for end-to-end testing including BioEngine Worker
initialization and test application configurations.
"""

import asyncio
import time
from typing import Dict, List

import pytest

from bioengine_worker.worker import BioEngineWorker


@pytest.fixture(scope="session")
def worker_mode() -> str:
    """
    Define the worker mode for end-to-end tests.

    Returns:
        The mode in which the BioEngine Worker will operate.
    """
    return "single-machine"


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
            "artifact_id": "composition-app",
            "application_id": "composition-app",
            "deployment_kwargs": {
                "CompositionDeployment": {"demo_input": "Hello World!"},
                "Deployment2": {"start_number": 10},
            },
        },
    ]


@pytest.fixture(scope="session")
def monitoring_interval_seconds() -> int:
    """Define the monitoring interval for the worker."""
    return 10


@pytest.fixture(scope="session")
def dashboard_url() -> str:
    """Define the URL for the dashboard."""
    return "https://bioimage.io/#/bioengine"


@pytest.fixture(scope="session")
def graceful_shutdown_timeout() -> int:
    """Define the timeout for graceful shutdown of the worker."""
    return 60


@pytest.fixture(scope="session")
def application_check_timeout() -> int:
    """Define the timeout for application connectivity checks."""
    return 30


@pytest.fixture(scope="session")
def num_cpus() -> int:
    """Define the number of CPUs available for the worker."""
    return 4


@pytest.fixture(scope="session")
def num_gpus() -> int:
    """Define the number of GPUs available for the worker."""
    return 0


@pytest.fixture(scope="session")
def memory_in_gb() -> int:
    """Define the memory available for the worker."""
    return 4


@pytest.fixture(scope="session", autouse=True)
def bioengine_worker(
    worker_mode,
    cache_dir,
    data_dir,
    startup_applications,
    monitoring_interval_seconds,
    server_url,
    hypha_token,
    num_cpus,
    num_gpus,
    memory_in_gb,
    dashboard_url,
    graceful_shutdown_timeout,
):
    """
    Create a shared BioEngine worker for remote interaction tests.

    This fixture provides a single worker instance that is shared across all
    remote interaction tests to avoid the overhead of starting multiple workers.
    The worker includes startup applications for comprehensive testing.
    """
    # Generate unique client ID for test isolation
    test_client_id = f"remote_test_worker_{int(time.time())}"

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
        client_id=test_client_id,
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

    async def cleanup_worker():
        nonlocal bioengine_worker

        if bioengine_worker:
            await bioengine_worker._stop(blocking=True)

    # Setup worker in a background task
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.create_task(bioengine_worker.start(blocking=True))

    # Wait for the worker to be ready before yielding
    loop.run_until_complete(bioengine_worker.is_ready.wait())

    yield bioengine_worker

    # Cleanup after all tests are done
    loop.run_until_complete(cleanup_worker())
    loop.close()


@pytest.fixture(scope="session")
def bioengine_worker_service_id(bioengine_worker) -> str:
    """
    Provide the shared BioEngine worker service ID for each test function.
    """
    return bioengine_worker.full_service_id
