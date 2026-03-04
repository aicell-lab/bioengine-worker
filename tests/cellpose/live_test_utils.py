from __future__ import annotations

import asyncio
import os
from typing import Any

import pytest
from hypha_rpc import connect_to_server

SERVER_URL = os.environ.get("HYPHA_SERVER_URL", "https://hypha.aicell.io")


def is_transient_connection_error(err: Exception) -> bool:
    text = str(err).lower()
    return (
        "websocket" in text
        or "http 404" in text
        or "http 502" in text
        or "server rejected websocket connection" in text
        or "method call timed out" in text
    )


def is_unavailable_workspace_error(err: Exception) -> bool:
    text = str(err).lower()
    return (
        "workspace" in text
        and ("does not exist" in text or "not accessible" in text)
    )


async def with_live_server(
    token: str,
    workspace: str,
    callback: Any,
    *,
    retries: int = 6,
    method_timeout: int = 240000,
) -> Any:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            async with connect_to_server(
                {
                    "server_url": SERVER_URL,
                    "token": token,
                    "workspace": workspace,
                    "method_timeout": method_timeout,
                }
            ) as server:
                return await callback(server)
        except Exception as e:
            last_error = e
            if attempt < retries - 1 and is_transient_connection_error(e):
                await asyncio.sleep(10)
                continue
            if is_transient_connection_error(e):
                pytest.skip(
                    "Live Hypha websocket endpoint is temporarily unavailable "
                    "(HTTP 404/502/timeouts during connection)."
                )
            if is_unavailable_workspace_error(e):
                pytest.skip(
                    "Live workspace is currently not accessible for this token/session."
                )
            raise

    if last_error is not None:
        raise last_error
    return None


def _supports_split_mode(service: Any) -> bool:
    try:
        schema = getattr(service.start_training, "__schema__", None)
        if isinstance(schema, dict):
            params = schema.get("parameters")
            if isinstance(params, dict):
                return "split_mode" in params.get("properties", {})
            if isinstance(params, list):
                return any(
                    isinstance(p, dict) and p.get("name") == "split_mode"
                    for p in params
                )
    except Exception:
        return False
    return False


async def resolve_cellpose_service(
    server: Any,
    workspace: str,
    *,
    application_id: str = "cellpose-finetuning",
    requested_service_id: str | None = None,
) -> Any:
    if requested_service_id:
        service = await server.get_service(requested_service_id)
        if _supports_split_mode(service):
            return service

    worker_candidates = [
        "bioimage-io/bioengine-worker",
        f"{workspace}/bioengine-worker",
    ]
    for worker_id in worker_candidates:
        try:
            worker = await server.get_service(worker_id)
            status = await worker.get_application_status(
                application_ids=[application_id]
            )
        except Exception:
            continue

        app_data = status.get(application_id, {}) if isinstance(status, dict) else {}
        service_entries = app_data.get("service_ids", [])
        for entry in reversed(service_entries):
            if isinstance(entry, dict):
                websocket_id = (
                    entry.get("websocket_service_id")
                    or entry.get("service_id")
                    or entry.get("id")
                )
            else:
                websocket_id = entry

            if isinstance(websocket_id, str) and websocket_id:
                try:
                    service = await server.get_service(websocket_id)
                    if _supports_split_mode(service):
                        return service
                except Exception:
                    continue

    candidate_ids = [
        os.environ.get("HYPHA_TEST_SERVICE_ID"),
        f"{workspace}/{application_id}",
        "bioimage-io/cellpose-finetuning",
    ]
    for candidate in candidate_ids:
        if not candidate:
            continue
        try:
            service = await server.get_service(candidate)
            if _supports_split_mode(service):
                return service
        except Exception:
            continue

    raise RuntimeError(
        "Could not resolve cellpose service id from workspace or worker status. "
        "Verify app deployment and websocket service ids in worker status."
    )
