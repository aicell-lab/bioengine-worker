from __future__ import annotations

import asyncio
import os
import sys
import uuid
from pathlib import Path
from typing import Any

import httpx
import pytest

sys.path.append(str(Path(__file__).resolve().parent))
from live_test_utils import resolve_cellpose_service, with_live_server

RUN_LIVE_REGRESSION_TESTS = os.environ.get("RUN_LIVE_REGRESSION_TESTS") == "1"
ROOT_DIR = Path(__file__).resolve().parents[2]
DEMO_DATASET_DIR = ROOT_DIR / "demo-dataset"


def _require_live_env() -> tuple[str, str]:
    if not RUN_LIVE_REGRESSION_TESTS:
        pytest.skip("Set RUN_LIVE_REGRESSION_TESTS=1 to run live e2e tests")

    token = os.environ.get("HYPHA_TOKEN")
    if not token:
        pytest.skip("HYPHA_TOKEN is required for live e2e tests")

    workspace = os.environ.get("HYPHA_TEST_WORKSPACE", "ri-scale")
    return token, workspace


async def _upload_demo_dataset(server: Any, workspace: str, local_dir: Path) -> str:
    artifact_manager = await server.get_service("public/artifact-manager")
    alias = f"demo-dataset-e2e-{uuid.uuid4().hex[:8]}"

    artifact = await artifact_manager.create(
        type="dataset",
        alias=alias,
        manifest={
            "name": alias,
            "description": "Temporary demo dataset for cellpose e2e tests",
            "type": "dataset",
        },
        config={"permissions": {"@": "*"}},
        stage=True,
    )

    all_files = [p for p in local_dir.rglob("*") if p.is_file()]
    timeout = httpx.Timeout(120.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        for file_path in all_files:
            rel_path = file_path.relative_to(local_dir).as_posix()
            put_url = await artifact_manager.put_file(artifact.id, file_path=rel_path)
            with open(file_path, "rb") as f:
                response = await client.put(put_url, content=f.read())
            response.raise_for_status()

    await artifact_manager.commit(artifact.id)
    return artifact.id


async def _run_training_to_finish(
    service: Any,
    *,
    artifact: str,
    train_images: str | None,
    train_annotations: str | None,
    n_samples: float = 5,
    retry_sample_fractions: tuple[float, ...] = (8, 10),
    train_split_ratio: float = 0.8,
    n_epochs: int = 2,
    timeout_seconds: int = 360,
) -> dict[str, Any]:
    print(
        f"[TRAIN] artifact={artifact} n_samples={n_samples} n_epochs={n_epochs}",
        flush=True,
    )

    async def _get_status_with_retries(
        session_id: str,
        attempts: int = 10,
        delay_seconds: float = 5.0,
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        for _ in range(attempts):
            try:
                return await service.get_training_status(session_id=session_id)
            except Exception as e:
                last_error = e
                text = str(e).lower()
                is_timeout = "timed out" in text or isinstance(e, TimeoutError)
                if not is_timeout:
                    raise
                await asyncio.sleep(delay_seconds)
        if last_error is not None:
            return {
                "status_type": "preparing",
                "message": f"Transient status polling timeout: {last_error}",
            }
        return {}

    async def _stop_active_sessions() -> None:
        try:
            sessions = await service.list_training_sessions(
                status_types=["waiting", "preparing", "running"]
            )
        except Exception:
            return

        if not isinstance(sessions, dict):
            return

        for existing_session_id, session_status in sessions.items():
            status_type = str((session_status or {}).get("status_type", "")).lower()
            if status_type in {"waiting", "preparing", "running"}:
                try:
                    await service.stop_training(session_id=existing_session_id)
                except Exception:
                    continue

        await asyncio.sleep(2)

    candidate_fractions = (n_samples, *retry_sample_fractions)
    last_status: dict[str, Any] = {}

    for sample_fraction in candidate_fractions:
        for _network_retry in range(2):
            print(
                f"[TRAIN] attempt sample_fraction={sample_fraction} network_retry={_network_retry}",
                flush=True,
            )
            await _stop_active_sessions()

            kwargs: dict[str, Any] = {
                "artifact": artifact,
                "split_mode": "auto",
                "train_split_ratio": train_split_ratio,
                "n_samples": sample_fraction,
                "n_epochs": n_epochs,
                "validation_interval": 1,
                "min_train_masks": 0,
            }
            if train_images is not None:
                kwargs["train_images"] = train_images
            if train_annotations is not None:
                kwargs["train_annotations"] = train_annotations

            try:
                started = await service.start_training(**kwargs)
                session_id = started["session_id"]
                print(
                    f"[TRAIN] started session_id={session_id} status={started.get('status_type')} msg={started.get('message')}",
                    flush=True,
                )

                started_type = str(started.get("status_type", "")).lower()
                started_message = str(started.get("message", "")).lower()
                deferred_start = (
                    started_type == "stopped"
                    and "deferred start" in started_message
                    and "gpu contention" in started_message
                )
                if deferred_start:
                    restarted = await service.restart_training(
                        session_id=session_id, n_epochs=n_epochs
                    )
                    session_id = restarted["session_id"]
                    print(
                        f"[TRAIN] restarted deferred session -> {session_id}",
                        flush=True,
                    )

                assert session_id

                deadline = asyncio.get_running_loop().time() + timeout_seconds
                latest_status: dict[str, Any] = {}

                try:
                    while asyncio.get_running_loop().time() < deadline:
                        latest_status = await _get_status_with_retries(
                            session_id=session_id
                        )
                        status_type = str(latest_status.get("status_type", "")).lower()
                        print(
                            f"[TRAIN] session_id={session_id} status={status_type} msg={latest_status.get('message')}",
                            flush=True,
                        )
                        if status_type in {"completed", "failed", "stopped"}:
                            break
                        await asyncio.sleep(3)
                finally:
                    status_type = str(latest_status.get("status_type", "")).lower()
                    if status_type in {"waiting", "preparing", "running"}:
                        try:
                            await service.stop_training(session_id=session_id)
                        except Exception:
                            pass

                latest_status = await _get_status_with_retries(session_id=session_id)
                last_status = latest_status
                final_type = str(latest_status.get("status_type", "")).lower()
                print(
                    f"[TRAIN] final session_id={session_id} status={final_type} msg={latest_status.get('message')}",
                    flush=True,
                )
                if final_type == "completed":
                    latest_status["n_samples_used"] = sample_fraction
                    latest_status["session_id"] = session_id
                    return latest_status

                message = str(latest_status.get("message") or "").lower()
                retriable = (
                    "no training samples available" in message
                    or "no training pairs found" in message
                    or "float division by zero" in message
                )
                if not retriable:
                    break
            except Exception as e:
                err = str(e).lower()
                transient = (
                    "websocket" in err
                    or "http 404" in err
                    or "http 502" in err
                    or "method call timed out" in err
                )
                if transient and _network_retry == 0:
                    await asyncio.sleep(5)
                    continue
                raise

    final_type = str(last_status.get("status_type", "")).lower()
    assert final_type == "completed", (
        f"Training did not complete for artifact={artifact}. "
        f"final status={final_type}, message={last_status.get('message')}"
    )
    return last_status


async def _infer_with_session_model(
    service: Any,
    *,
    session_id: str,
    artifact: str,
    image_path: str,
) -> dict[str, Any]:
    print(
        f"[INFER] model_session={session_id} artifact={artifact} image={image_path}",
        flush=True,
    )
    result = await service.infer(
        model=session_id,
        artifact=artifact,
        image_paths=[image_path],
        diameter=40,
        json_safe=True,
    )
    assert isinstance(result, list)
    assert len(result) == 1
    output = result[0].get("output")
    assert output is not None
    assert output.get("encoding") in {"ndarray_base64", "mask_png_base64"}
    print(
        f"[INFER] ok model_session={session_id} encoding={output.get('encoding')}",
        flush=True,
    )
    return result[0]


@pytest.mark.asyncio
async def test_training_bia_url_small_percentage_auto_split() -> None:
    token, workspace = _require_live_env()

    async def _run(server: Any) -> None:
        service = await resolve_cellpose_service(server, workspace)
        await _run_training_to_finish(
            service,
            artifact="https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD1392",
            train_images="images/*/",
            train_annotations="annotations/*/",
            n_samples=0.005,
            train_split_ratio=0.8,
            timeout_seconds=420,
        )

    await with_live_server(token, workspace, _run)


@pytest.mark.asyncio
async def test_training_ri_scale_zarr_demo_small_percentage_auto_split() -> None:
    token, workspace = _require_live_env()

    async def _run(server: Any) -> None:
        service = await resolve_cellpose_service(server, workspace)
        await _run_training_to_finish(
            service,
            artifact="ri-scale/zarr-demo",
            train_images="images/108bb69d-2e52-4382-8100-e96173db24ee/*.ome.tif",
            train_annotations="annotations/108bb69d-2e52-4382-8100-e96173db24ee/*_mask.ome.tif",
            n_samples=0.005,
            train_split_ratio=0.8,
            timeout_seconds=1200,
        )

    await with_live_server(token, workspace, _run)


@pytest.mark.asyncio
async def test_training_local_demo_dataset_small_percentage_auto_split() -> None:
    token, workspace = _require_live_env()

    if not DEMO_DATASET_DIR.exists():
        pytest.skip(f"demo dataset not found at {DEMO_DATASET_DIR}")

    async def _run(server: Any) -> None:
        service = await resolve_cellpose_service(server, workspace)
        artifact_id = await _upload_demo_dataset(server, workspace, DEMO_DATASET_DIR)

        await _run_training_to_finish(
            service,
            artifact=artifact_id,
            train_images="images/*/*.ome.tif",
            train_annotations="annotations/*/*_mask.ome.tif",
            n_samples=0.005,
            train_split_ratio=0.8,
            timeout_seconds=1200,
        )

    await with_live_server(token, workspace, _run)


@pytest.mark.asyncio
async def test_three_sources_sequential_with_inference() -> None:
    token, workspace = _require_live_env()

    async def _run(server: Any) -> None:
        service = await resolve_cellpose_service(server, workspace)

        if not DEMO_DATASET_DIR.exists():
            pytest.skip(f"demo dataset not found at {DEMO_DATASET_DIR}")

        shared_infer_artifact = "ri-scale/zarr-demo"
        shared_infer_image = "images/108bb69d-2e52-4382-8100-e96173db24ee/t0000.ome.tif"

        # 1) Artifact source (pre-existing artifact id) using uploaded demo dataset
        print("[E2E] Source 1/3: artifact-id source", flush=True)
        artifact_source_id = await _upload_demo_dataset(server, workspace, DEMO_DATASET_DIR)
        artifact_status = await _run_training_to_finish(
            service,
            artifact=artifact_source_id,
            train_images="images/*/*.ome.tif",
            train_annotations="annotations/*/*_mask.ome.tif",
            n_samples=5,
            n_epochs=2,
            timeout_seconds=1200,
        )
        artifact_session_id = str(artifact_status.get("session_id") or "")
        assert artifact_session_id
        await _infer_with_session_model(
            service,
            session_id=artifact_session_id,
            artifact=shared_infer_artifact,
            image_path=shared_infer_image,
        )

        # 2) BioImage Archive URL source
        print("[E2E] Source 2/3: BioImage Archive URL source", flush=True)
        bia_status = await _run_training_to_finish(
            service,
            artifact="https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD1392",
            train_images="images/*/",
            train_annotations="annotations/*/",
            n_samples=5,
            n_epochs=2,
            timeout_seconds=1200,
        )
        bia_session_id = str(bia_status.get("session_id") or "")
        assert bia_session_id
        await _infer_with_session_model(
            service,
            session_id=bia_session_id,
            artifact=shared_infer_artifact,
            image_path=shared_infer_image,
        )

        # 3) Uploaded local demo dataset source
        print("[E2E] Source 3/3: uploaded local dataset source", flush=True)
        uploaded_artifact_id = await _upload_demo_dataset(server, workspace, DEMO_DATASET_DIR)
        uploaded_status = await _run_training_to_finish(
            service,
            artifact=uploaded_artifact_id,
            train_images="images/*/*.ome.tif",
            train_annotations="annotations/*/*_mask.ome.tif",
            n_samples=5,
            n_epochs=2,
            timeout_seconds=1200,
        )
        uploaded_session_id = str(uploaded_status.get("session_id") or "")
        assert uploaded_session_id
        await _infer_with_session_model(
            service,
            session_id=uploaded_session_id,
            artifact=shared_infer_artifact,
            image_path=shared_infer_image,
        )
        print("[E2E] Completed all three sources with inference", flush=True)

    await with_live_server(token, workspace, _run)
