"""
Targeted test for artifact version commit behaviour.

Verifies that save_application commits the artifact under the version
declared in manifest.yaml instead of always defaulting to "latest".

Run with:
    conda activate bioengine-worker
    source .env
    pytest tests/test_artifact_version.py -v
"""

import asyncio
import os

import pytest
import pytest_asyncio
from dotenv import load_dotenv
from hypha_rpc import connect_to_server
from pathlib import Path

load_dotenv(Path(__file__).parent.parent / ".env")


# ── helpers ────────────────────────────────────────────────────────────────────

MANIFEST_TMPL = """\
name: Version Test App
id: {artifact_id}
id_emoji: "🧪"
description: "Temporary app used to test artifact version commits"
type: ray-serve
format_version: 0.5.0
version: {version}
authors:
  - {{name: "Test"}}
license: MIT
deployments:
  - test_dep:TestDep
authorized_users:
  - "*"
"""

DEPLOYMENT_SRC = """\
from ray import serve

@serve.deployment(ray_actor_options={"num_cpus": 0, "num_gpus": 0, "memory": 128*1024**2, "runtime_env": {"pip": []}})
class TestDep:
    async def async_init(self): pass
    async def test_deployment(self): pass
    async def check_health(self): pass
"""


def _make_files(artifact_id: str, version: str):
    return [
        {
            "name": "manifest.yaml",
            "content": MANIFEST_TMPL.format(artifact_id=artifact_id, version=version),
            "type": "text",
        },
        {"name": "test_dep.py", "content": DEPLOYMENT_SRC, "type": "text"},
    ]


# ── fixtures ────────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture(scope="module")
async def hypha_client():
    token = os.environ.get("BIOIMAGE_IO_TOKEN") or os.environ.get("HYPHA_TOKEN")
    assert token, "No Hypha token found in environment"
    client = await connect_to_server(
        {"server_url": "https://hypha.aicell.io", "token": token}
    )
    yield client
    await client.disconnect()


@pytest_asyncio.fixture(scope="module")
async def worker(hypha_client):
    return await hypha_client.get_service("bioimage-io/bioengine-worker")


@pytest_asyncio.fixture(scope="module")
async def artifact_manager(hypha_client):
    return await hypha_client.get_service("public/artifact-manager")


# ── tests ───────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_save_application_commits_manifest_version(worker, artifact_manager):
    """save_application must commit the artifact under the version in manifest.yaml."""
    artifact_alias = "version-test-app-pytest"
    artifact_id = f"bioimage-io/{artifact_alias}"
    test_version = "2.3.4"

    # Clean up any leftover artifact from a previous failed run
    try:
        await artifact_manager.delete(artifact_id)
    except Exception:
        pass

    try:
        files = _make_files(artifact_alias, test_version)
        saved_id = await worker.save_application(files=files)
        assert saved_id == artifact_id, f"Unexpected artifact ID: {saved_id}"

        # Read the committed artifact and verify the version tag
        artifact = await artifact_manager.read(artifact_id=artifact_id, version=test_version)
        assert artifact is not None, "Artifact not found at expected version"
        manifest = artifact.manifest if hasattr(artifact, "manifest") else artifact.get("manifest", {})
        assert manifest.get("version") == test_version, (
            f"Manifest version mismatch: expected {test_version!r}, "
            f"got {manifest.get('version')!r}"
        )

    finally:
        try:
            await artifact_manager.delete(artifact_id)
        except Exception:
            pass


@pytest.mark.asyncio
async def test_save_application_version_update(worker, artifact_manager):
    """Re-saving with a bumped version should update the artifact manifest version.

    Note: Hypha's artifact model maintains a single version tag per artifact.
    Re-committing with a new version updates the tag content in place rather
    than creating an additional tag. The manifest version field is the source
    of truth for what version is deployed.
    """
    artifact_alias = "version-update-test-pytest"
    artifact_id = f"bioimage-io/{artifact_alias}"
    v1, v2 = "1.0.0", "1.1.0"

    try:
        await artifact_manager.delete(artifact_id)
    except Exception:
        pass

    try:
        # Save version 1.0.0
        await worker.save_application(files=_make_files(artifact_alias, v1))
        a1 = await artifact_manager.read(artifact_id=artifact_id, version=v1)
        assert a1 is not None, f"v1 artifact not found at version {v1}"
        assert a1.manifest.get("version") == v1

        # Save version 1.1.0 — Hypha updates the tag in place
        await worker.save_application(files=_make_files(artifact_alias, v2))
        # The artifact should reflect the new manifest version
        a2 = await artifact_manager.read(artifact_id=artifact_id, version=v1)
        assert a2 is not None
        assert a2.manifest.get("version") == v2, (
            f"Expected manifest version {v2!r} after update, got {a2.manifest.get('version')!r}"
        )

    finally:
        try:
            await artifact_manager.delete(artifact_id)
        except Exception:
            pass


pytest_plugins = ("pytest_asyncio",)
