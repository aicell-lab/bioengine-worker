from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

import pytest
from hypha_rpc import connect_to_server

sys.path.append(str(Path(__file__).resolve().parent))
from live_test_utils import resolve_cellpose_service

SERVER_URL = os.environ.get("HYPHA_SERVER_URL", "https://hypha.aicell.io")
RUN_LIVE_REGRESSION_TESTS = os.environ.get("RUN_LIVE_REGRESSION_TESTS") == "1"


def _require_live_env() -> tuple[str, str, str]:
    if not RUN_LIVE_REGRESSION_TESTS:
        pytest.skip("Set RUN_LIVE_REGRESSION_TESTS=1 to run live e2e tests")

    token = os.environ.get("HYPHA_TOKEN")
    if not token:
        pytest.skip("HYPHA_TOKEN is required for live e2e tests")

    workspace = os.environ.get("HYPHA_TEST_WORKSPACE", "ri-scale")
    service_id = os.environ.get("HYPHA_TEST_SERVICE_ID", "bioimage-io/cellpose-finetuning")
    return token, workspace, service_id


@pytest.mark.asyncio
async def test_small_percentage_auto_split_progresses() -> None:
    token, workspace, service_id = _require_live_env()

    async with connect_to_server(
        {"server_url": SERVER_URL, "token": token, "workspace": workspace}
    ) as server:
        service = await resolve_cellpose_service(
            server,
            workspace,
            requested_service_id=service_id,
        )

        started = await service.start_training(
            artifact="ri-scale/zarr-demo",
            train_images="images/*/",
            train_annotations="annotations/*/",
            split_mode="auto",
            train_split_ratio=0.8,
            n_samples=0.02,
            n_epochs=2,
            min_train_masks=1,
        )

        session_id = started["session_id"]
        assert session_id

        reached_non_preparing = False
        deadline = time.time() + 240

        try:
            while time.time() < deadline:
                status = await service.get_training_status(session_id)
                status_type = str(status.get("status_type", "")).lower()

                if status_type != "preparing":
                    reached_non_preparing = True
                    break

                await asyncio.sleep(3)
        finally:
            latest = await service.get_training_status(session_id)
            latest_type = str(latest.get("status_type", "")).lower()
            if latest_type in {"preparing", "running", "waiting"}:
                await service.stop_training(session_id=session_id)

        assert reached_non_preparing, "Training remained stuck in 'preparing' state"


@pytest.mark.asyncio
async def test_infer_pretrained_model_e2e() -> None:
    token, workspace, service_id = _require_live_env()

    async with connect_to_server(
        {"server_url": SERVER_URL, "token": token, "workspace": workspace}
    ) as server:
        service = await resolve_cellpose_service(
            server,
            workspace,
            requested_service_id=service_id,
        )

        result = await service.infer(
            artifact="ri-scale/zarr-demo",
            image_paths=["images/108bb69d-2e52-4382-8100-e96173db24ee/t0000.ome.tif"],
            model="cpsam",
            diameter=40,
            json_safe=True,
        )

        assert isinstance(result, list)
        assert len(result) == 1
        output = result[0].get("output")
        assert output is not None
        assert output.get("encoding") in {"ndarray_base64", "mask_png_base64"}
