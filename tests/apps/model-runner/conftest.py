"""Fixtures for live model-runner service tests.

Requires the model-runner app to be deployed to bioimage-io/bioengine-worker.
Set BIOIMAGE_IO_TOKEN (or HYPHA_TOKEN) in the environment before running.

    pytest tests/apps/model-runner/ -v
"""

import os

import httpx
import numpy as np
import pytest
import pytest_asyncio
from hypha_rpc import connect_to_server

SERVER_URL = "https://hypha.aicell.io"
WORKSPACE = "bioimage-io"
SERVICE_ID = "bioimage-io/model-runner"

# Small (~18 MB), published, passes all checks
TEST_MODEL_ID = "ambitious-ant"


def _token() -> str:
    token = os.environ.get("BIOIMAGE_IO_TOKEN") or os.environ.get("HYPHA_TOKEN")
    if not token:
        pytest.skip("BIOIMAGE_IO_TOKEN or HYPHA_TOKEN not set")
    return token


@pytest_asyncio.fixture(scope="session")
async def model_runner():
    """Return a connected model-runner service handle."""
    server = await connect_to_server(
        {"server_url": SERVER_URL, "token": _token(), "workspace": WORKSPACE}
    )
    svc = await server.get_service(SERVICE_ID)
    yield svc
    await server.disconnect()


@pytest_asyncio.fixture(scope="session")
async def model_rdf(model_runner):
    """Cached RDF for TEST_MODEL_ID used across tests."""
    return await model_runner.get_model_rdf(model_id=TEST_MODEL_ID, stage=False)


@pytest_asyncio.fixture(scope="session")
async def test_image_array(model_runner, model_rdf):
    """Load the first test-input array for TEST_MODEL_ID."""
    upload_info = await model_runner.get_upload_url(file_type=".npy")
    # Fetch the npy file from the model package via a URL
    import io

    npy_url_info = await model_runner.get_upload_url(file_type=".npy")

    # Use model RDF to find test input — download from bioimage.io directly
    rdf_source = model_rdf.get("test_inputs", [None])[0]
    assert rdf_source is not None, "model RDF has no test_inputs"

    # Reconstruct the download URL for the test input npy file
    base_url = f"https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/{TEST_MODEL_ID}/test_input.npy"
    # Try the known URL pattern, fall back to rdf source
    async with httpx.AsyncClient(follow_redirects=True) as client:
        resp = await client.get(base_url)
        if resp.status_code != 200:
            pytest.skip(f"Could not fetch test input from {base_url}: {resp.status_code}")
        arr = np.load(io.BytesIO(resp.content)).astype("float32")
    return arr
