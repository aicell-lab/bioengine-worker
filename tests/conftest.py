"""
Global pytest configuration for BioEngine Worker tests.

Provides common fixtures for environment setup, Hypha authentication,
and test isolation across the entire test suite.

Requires HYPHA_TOKEN in environment and bioengine-worker conda environment.
"""

import os
import tempfile
from pathlib import Path
from datetime import datetime

import pytest
import pytest_asyncio
from dotenv import load_dotenv
from hypha_rpc import connect_to_server
from hypha_rpc.rpc import RemoteService

# Load environment variables from .env file
load_dotenv()


@pytest.fixture(scope="session")
def session_id() -> str:
    """Generate unique timestamp-based session ID for test isolation."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


@pytest.fixture(scope="session")
def workspace_folder() -> Path:
    """
    Return project root directory and set BIOENGINE_WORKER_LOCAL_ARTIFACT_PATH.
    
    Configures environment for local test artifact discovery.
    """
    folder = Path(__file__).resolve().parent.parent

    # Set the local artifact path for BioEngine Worker tests
    os.environ["BIOENGINE_WORKER_LOCAL_ARTIFACT_PATH"] = str(folder / "tests")

    return folder


@pytest.fixture(scope="session")
def server_url() -> str:
    """Return Hypha server URL for test connections."""
    return "https://hypha.aicell.io"


@pytest.fixture(scope="session")
def hypha_token() -> str:
    """
    Get Hypha authentication token from HYPHA_TOKEN environment variable.
    
    Skips test if token not found. Required for Hypha server access.
    """
    token = os.environ.get("HYPHA_TOKEN")
    if not token:
        pytest.skip(
            "HYPHA_TOKEN environment variable not set. "
            "Please ensure .env file contains HYPHA_TOKEN and activate bioengine-worker environment."
        )
    return token


@pytest.fixture(scope="session")
def cache_dir():
    """Create temporary cache directory with automatic cleanup."""
    with tempfile.TemporaryDirectory(
        prefix=f"bioengine_worker_cache_"
    ) as temp_cache_dir:
        cache_dir = Path(temp_cache_dir)
        yield cache_dir

    # Cleanup is handled automatically by TemporaryDirectory context manager


@pytest.fixture(scope="session")
def data_dir(workspace_folder: Path) -> Path:
    """Create and return BioEngine Worker data directory."""
    data_dir = workspace_folder / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


@pytest_asyncio.fixture(scope="function")
async def hypha_client(hypha_token: str, session_id: str):
    """
    Create Hypha RPC client with unique ID for each test function.
    
    Automatically connects and disconnects for proper test isolation.
    """
    client: RemoteService
    client = await connect_to_server(
        {
            "server_url": "https://hypha.aicell.io",
            "token": hypha_token,
            "client_id": f"bioengine_test_client_{session_id}",
        }
    )
    yield client

    await client.disconnect()


@pytest.fixture(scope="function")
def hypha_workspace(hypha_client: RemoteService) -> str:
    """Extract workspace ID from connected Hypha client."""
    return hypha_client.config.workspace


@pytest.fixture(scope="function")
def hypha_client_id(hypha_client: RemoteService) -> str:
    """Extract unique client ID from connected Hypha client."""
    return hypha_client.config.client_id


# Configure asyncio for pytest
pytest_plugins = ("pytest_asyncio",)
