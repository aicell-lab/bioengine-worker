"""
Global pytest configuration for BioEngine Worker tests.

This configuration applies to all test modules in the tests/ directory.
Provides common fixtures for environment setup, authentication, and Hypha client management.
"""

import os
import tempfile
from pathlib import Path
import asyncio

import pytest
import pytest_asyncio
from dotenv import load_dotenv
from hypha_rpc import connect_to_server
from hypha_rpc.rpc import RemoteService

# Load environment variables from .env file
load_dotenv()


@pytest.fixture(scope="session")
def workspace_folder() -> Path:
    """
    Return the workspace folder path for testing.

    Returns:
        Path to the workspace folder for test operations
    """
    folder = Path(__file__).resolve().parent.parent

    # Set the local artifact path for BioEngine Worker tests
    os.environ["BIOENGINE_WORKER_LOCAL_ARTIFACT_PATH"] = str(folder / "tests")

    return folder


@pytest.fixture(scope="session")
def server_url() -> str:
    """
    Return the Hypha server URL for testing.

    Returns:
        URL of the Hypha server for test connections
    """
    return "https://hypha.aicell.io"


@pytest.fixture(scope="session")
def hypha_token() -> str:
    """
    Retrieve Hypha authentication token from environment.

    The token should be available in the HYPHA_TOKEN environment variable,
    typically loaded from a .env file.

    Returns:
        Authentication token for Hypha server access

    Raises:
        pytest.skip: If HYPHA_TOKEN environment variable is not set

    Note:
        Activate the bioengine-worker conda environment before running tests:
        ```bash
        conda activate bioengine-worker
        pytest tests/
        ```
    """
    token = os.environ.get("HYPHA_TOKEN")
    if not token:
        pytest.skip(
            "HYPHA_TOKEN environment variable not set. "
            "Please ensure .env file contains HYPHA_TOKEN and activate bioengine-worker environment."
        )
    return token


@pytest.fixture(scope="session")
def cache_dir() -> Path:
    """
    Create and return a test cache directory.

    Creates a unique cache directory for each test session to avoid conflicts
    and ensure test isolation. The directory is cleaned up after the session ends.

    Returns:
        Path to the test cache directory
    """
    with tempfile.TemporaryDirectory(
        prefix="bioengine_worker_cache_"
    ) as temp_cache_dir:
        cache_dir = Path(temp_cache_dir)
        yield cache_dir

    # Cleanup is handled automatically by TemporaryDirectory context manager


@pytest.fixture(scope="session")
def data_dir(workspace_folder: Path) -> Path:
    """
    Return the BioEngine Worker data directory.

    Args:
        workspace_folder: Path to the workspace folder

    Returns:
        Path to the BioEngine Worker data directory
    """
    data_dir = workspace_folder / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


@pytest_asyncio.fixture(scope="function")
async def hypha_client(hypha_token: str) -> RemoteService:
    """
    Create a Hypha client for service interaction testing.

    Provides a fresh Hypha client connection for each test function,
    ensuring test isolation and proper connection management.

    Args:
        hypha_token: Authentication token for Hypha server

    Yields:
        Connected Hypha client instance

    Raises:
        Exception: If client connection fails

    Note:
        Each test gets its own client instance to avoid connection
        conflicts and ensure proper cleanup between tests.
    """
    client = await connect_to_server(
        {
            "server_url": "https://hypha.aicell.io",
            "token": hypha_token,
        }
    )
    yield client

    try:
        await client.disconnect()
    except Exception as e:
        # Log disconnect errors but don't fail tests
        print(f"⚠️  Warning: Hypha client disconnect failed: {e}")


@pytest.fixture(scope="function")
def hypha_workspace(hypha_client: RemoteService) -> str:
    """
    Hypha Workspace
    """
    return hypha_client.config.workspace


# Configure asyncio for pytest
pytest_plugins = ("pytest_asyncio",)
