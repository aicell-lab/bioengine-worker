"""
Targeted test for artifact version commit behaviour.

Verifies that upload_app commits the artifact under the version
declared in manifest.yaml instead of always defaulting to "latest".

Three cases are tested:
1. New version tag → creates an isolated snapshot (previous versions intact)
2. Re-saving the latest version → updates in place (no duplicate tag)
3. Re-saving an older non-latest version → raises ValueError

Run with:
    conda activate bioengine-worker
    source .env
    pytest tests/test_artifact_version.py -v
"""

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
async def test_upload_app_commits_manifest_version(worker, artifact_manager):
    """upload_app must commit the artifact under the version in manifest.yaml."""
    artifact_alias = "version-test-app-pytest"
    artifact_id = f"bioimage-io/{artifact_alias}"
    test_version = "2.3.4"

    try:
        await artifact_manager.delete(artifact_id)
    except Exception:
        pass

    try:
        saved_id = await worker.upload_app(files=_make_files(artifact_alias, test_version))
        assert saved_id == artifact_id, f"Unexpected artifact ID: {saved_id}"

        # The artifact must be retrievable by its version tag
        artifact = await artifact_manager.read(artifact_id=artifact_id, version=test_version)
        assert artifact is not None, "Artifact not found at expected version tag"
        assert artifact.manifest.get("version") == test_version, (
            f"Manifest version mismatch: expected {test_version!r}, "
            f"got {artifact.manifest.get('version')!r}"
        )
    finally:
        try:
            await artifact_manager.delete(artifact_id)
        except Exception:
            pass


@pytest.mark.asyncio
async def test_upload_app_new_version_creates_isolated_snapshot(worker, artifact_manager):
    """Bumping the version in manifest.yaml creates a new isolated snapshot.

    After saving v1 and then v2, both version tags must be independently
    readable and reflect the manifest version they were saved with.
    """
    artifact_alias = "version-bump-test-pytest"
    artifact_id = f"bioimage-io/{artifact_alias}"
    v1, v2 = "1.0.0", "1.1.0"

    try:
        await artifact_manager.delete(artifact_id)
    except Exception:
        pass

    try:
        await worker.upload_app(files=_make_files(artifact_alias, v1))
        a1 = await artifact_manager.read(artifact_id=artifact_id, version=v1)
        assert a1 is not None, f"Artifact not found at version {v1}"
        assert a1.manifest.get("version") == v1

        await worker.upload_app(files=_make_files(artifact_alias, v2))
        a2 = await artifact_manager.read(artifact_id=artifact_id, version=v2)
        assert a2 is not None, f"Artifact not found at version {v2}"
        assert a2.manifest.get("version") == v2

        # v1 must still be independently readable
        a1_after = await artifact_manager.read(artifact_id=artifact_id, version=v1)
        assert a1_after is not None, f"v1 disappeared after saving v2"
        assert a1_after.manifest.get("version") == v1, (
            f"v1 manifest was mutated after saving v2: got {a1_after.manifest.get('version')!r}"
        )
    finally:
        try:
            await artifact_manager.delete(artifact_id)
        except Exception:
            pass


@pytest.mark.asyncio
async def test_upload_app_resave_latest_version_updates_inplace(worker, artifact_manager):
    """Re-saving the same (latest) version updates the artifact in place.

    No new version tag is created; the existing version tag reflects the
    updated content.
    """
    artifact_alias = "version-resave-pytest"
    artifact_id = f"bioimage-io/{artifact_alias}"
    version = "1.0.0"

    try:
        await artifact_manager.delete(artifact_id)
    except Exception:
        pass

    try:
        await worker.upload_app(files=_make_files(artifact_alias, version))
        # Re-saving the same version must not raise
        await worker.upload_app(files=_make_files(artifact_alias, version))

        artifact = await artifact_manager.read(artifact_id=artifact_id, version=version)
        assert artifact is not None
        assert artifact.manifest.get("version") == version
    finally:
        try:
            await artifact_manager.delete(artifact_id)
        except Exception:
            pass


@pytest.mark.asyncio
async def test_upload_app_resave_older_version_raises(worker, artifact_manager):
    """Re-saving an older (non-latest) version must raise a ValueError.

    Once a newer version exists, trying to overwrite an older version is
    not supported by Hypha. The worker must surface a clear error.
    """
    artifact_alias = "version-older-resave-pytest"
    artifact_id = f"bioimage-io/{artifact_alias}"
    v1, v2 = "1.0.0", "1.1.0"

    try:
        await artifact_manager.delete(artifact_id)
    except Exception:
        pass

    try:
        await worker.upload_app(files=_make_files(artifact_alias, v1))
        await worker.upload_app(files=_make_files(artifact_alias, v2))

        # Attempting to re-save v1 when v2 is the latest must raise
        with pytest.raises(Exception, match=r"[Cc]annot re-save|newer version|already exists"):
            await worker.upload_app(files=_make_files(artifact_alias, v1))
    finally:
        try:
            await artifact_manager.delete(artifact_id)
        except Exception:
            pass


pytest_plugins = ("pytest_asyncio",)
