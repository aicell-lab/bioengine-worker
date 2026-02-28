import micropip

await micropip.install(["hypha-rpc", "httpx"])

import json
import os
from typing import Any

from hypha_rpc import connect_to_server, login

SERVER_URL = os.environ.get("HYPHA_SERVER_URL", "https://hypha.aicell.io")
DEFAULT_SERVICE = "bioimage-io/cellpose-finetuning"

_state: dict[str, Any] = {
    "server": None,
    "workspace": None,
    "service": None,
    "service_id": None,
    "schemas": {},
}


async def connect_cellpose(
    server_url: str | None = None,
    workspace: str | None = None,
    token: str | None = None,
):
    target_server = server_url or SERVER_URL
    target_token = token or os.environ.get("HYPHA_TOKEN")
    if not target_token:
        target_token = await login({"server_url": target_server})

    config: dict[str, Any] = {"server_url": target_server, "token": target_token}
    if workspace:
        config["workspace"] = workspace

    server = await connect_to_server(config)
    _state["server"] = server
    _state["workspace"] = server.config.workspace
    return {
        "server_url": target_server,
        "workspace": server.config.workspace,
        "user_id": server.config.user.get("id"),
    }


async def connect_cellpose_service():
    return await select_cellpose_service(DEFAULT_SERVICE)


async def list_cellpose_services():
    if not _state["server"]:
        await connect_cellpose()

    server = _state["server"]
    workspace = _state.get("workspace")
    candidates: list[str] = []

    env_service = os.environ.get("HYPHA_TEST_SERVICE_ID")
    if env_service:
        candidates.append(env_service)

    if workspace:
        candidates.append(f"{workspace}/cellpose-finetuning")

    candidates.extend(
        [
            DEFAULT_SERVICE,
            "bioimage-io/cellpose-finetuning",
        ]
    )

    # Worker-derived service ids (best source after redeploys)
    worker_candidates = ["bioimage-io/bioengine-worker"]
    if workspace:
        worker_candidates.append(f"{workspace}/bioengine-worker")

    for worker_id in worker_candidates:
        try:
            worker = await server.get_service(worker_id)
            app_status = await worker.get_application_status(
                application_ids=["cellpose-finetuning"]
            )
            app_data = (
                app_status.get("cellpose-finetuning", {})
                if isinstance(app_status, dict)
                else {}
            )
            for entry in app_data.get("service_ids", []):
                if isinstance(entry, dict):
                    service_id = entry.get("websocket_service_id")
                else:
                    service_id = entry
                if isinstance(service_id, str) and service_id:
                    candidates.append(service_id)
        except Exception:
            continue

    deduped: list[str] = []
    seen: set[str] = set()
    for service_id in candidates:
        if service_id and service_id not in seen:
            seen.add(service_id)
            deduped.append(service_id)

    services: list[dict[str, Any]] = []
    for service_id in deduped:
        available = False
        error = None
        try:
            await server.get_service(service_id)
            available = True
        except Exception as e:
            error = str(e)
        services.append(
            {"service_id": service_id, "available": available, "error": error}
        )

    return services


async def select_cellpose_service(service_id: str | None = None):
    if not _state["server"]:
        await connect_cellpose()

    server = _state["server"]
    target = service_id
    if not target:
        services = await list_cellpose_services()
        for service_info in services:
            if service_info.get("available"):
                target = service_info.get("service_id")
                break

    if not target:
        raise RuntimeError(
            "Could not resolve an available Cellpose service. "
            "Call list_cellpose_services() and pass an explicit service_id."
        )

    service = await server.get_service(target)
    _state["service"] = service
    _state["service_id"] = target
    await _refresh_service_schemas()
    return {"service_id": target, "workspace": _state.get("workspace")}


async def _refresh_service_schemas():
    if not _state["service"]:
        await connect_cellpose_service()

    service = _state["service"]
    method_names = [
        "start_training",
        "get_training_status",
        "list_training_sessions",
        "stop_training",
        "restart_training",
        "infer",
        "export_model",
    ]

    schemas: dict[str, Any] = {}
    for name in method_names:
        method = getattr(service, name, None)
        if method is None:
            continue
        schema = getattr(method, "__schema__", None)
        if schema is not None:
            schemas[name] = schema

    _state["schemas"] = schemas
    return schemas


async def start_cellpose_training(**kwargs):
    if not _state["service"]:
        await connect_cellpose_service()
    return await _state["service"].start_training(**kwargs)


async def get_training_status(session_id: str):
    if not _state["service"]:
        await connect_cellpose_service()
    return await _state["service"].get_training_status(session_id=session_id)


async def list_training_sessions(status_types: list[str] | None = None):
    if not _state["service"]:
        await connect_cellpose_service()
    if status_types:
        return await _state["service"].list_training_sessions(
            status_types=status_types,
        )
    return await _state["service"].list_training_sessions()


async def stop_training(session_id: str):
    if not _state["service"]:
        await connect_cellpose_service()
    return await _state["service"].stop_training(session_id=session_id)


async def restart_training(session_id: str, n_epochs: int | None = None):
    if not _state["service"]:
        await connect_cellpose_service()
    payload: dict[str, Any] = {"session_id": session_id}
    if n_epochs is not None:
        payload["n_epochs"] = int(n_epochs)
    return await _state["service"].restart_training(**payload)


async def infer_cellpose(**kwargs):
    if not _state["service"]:
        await connect_cellpose_service()
    return await _state["service"].infer(**kwargs)


async def export_model(
    session_id: str,
    model_name: str | None = None,
    collection: str = "bioimage-io/colab-annotations",
):
    if not _state["service"]:
        await connect_cellpose_service()
    payload: dict[str, Any] = {
        "session_id": session_id,
        "collection": collection,
    }
    if model_name:
        payload["model_name"] = model_name
    return await _state["service"].export_model(**payload)


async def get_cellpose_schemas(refresh: bool = False):
    if refresh or not _state.get("schemas"):
        await _refresh_service_schemas()
    return _state.get("schemas", {})


async def print_cellpose_runtime_docs(include_schemas: bool = True):
    docs = RUNTIME_GUIDE
    if include_schemas:
        schemas = await get_cellpose_schemas(refresh=True)
        if schemas:
            docs += "\n\nLive service schemas (from service.<function>.__schema__):\n"
            docs += json.dumps(schemas, indent=2, sort_keys=True)
        else:
            docs += (
                "\n\nLive service schemas are unavailable. "
                "Connect/select a service first, then call print_cellpose_runtime_docs()."
            )
    print(docs)


RUNTIME_GUIDE = """
You are a Cellpose training operations assistant for BioEngine.

Primary goal:
- Help users run reliable Cellpose fine-tuning workflows end-to-end: connect, select service, start training, monitor, troubleshoot, stop/restart, infer, export.

Recommended execution order:
1) Connect:
     - await connect_cellpose(server_url=None, workspace=None, token=None)
2) Discover and select service (important after redeploys):
     - await list_cellpose_services()
     - await select_cellpose_service(service_id=None)
3) Start training:
     - await start_cellpose_training(...)
4) Monitor:
     - await get_training_status(session_id)
     - optional: await list_training_sessions(status_types=[...])
5) Control:
     - await stop_training(session_id)
     - await restart_training(session_id, n_epochs=None)
6) Post-training:
     - await infer_cellpose(...)
     - await export_model(session_id, model_name=None, collection='bioimage-io/colab-annotations')

Function reference (wrapper-level):
- connect_cellpose(server_url=None, workspace=None, token=None)
    - Connects to Hypha server and caches server/workspace in _state.
    - If token missing, invokes interactive login().
    - Returns {server_url, workspace, user_id}.

- list_cellpose_services()
    - Lists candidate Cellpose service IDs from defaults + worker app status.
    - Returns [{service_id, available, error}].

- select_cellpose_service(service_id=None)
    - Selects explicit service_id or first available from list_cellpose_services().
    - Caches service + service_id in _state.
    - Refreshes live __schema__ docs.
    - Returns {service_id, workspace}.

- start_cellpose_training(**kwargs)
    - Forwards kwargs to service.start_training.
    - Required: artifact and either
        a) train_images + train_annotations, or
        b) metadata_dir, or
        c) BioImage Archive URL in artifact.

- get_training_status(session_id)
    - Returns latest status including status_type/message/progress/losses/metrics.

- list_training_sessions(status_types=None)
    - Lists known sessions, optionally filtered by status.

- stop_training(session_id)
    - Requests graceful stop for running session.

- restart_training(session_id, n_epochs=None)
    - Starts a new run from previous session checkpoint/params.

- infer_cellpose(**kwargs)
    - Forwards infer parameters to service.infer.

- export_model(session_id, model_name=None, collection='bioimage-io/colab-annotations')
    - Exports completed training to BioImage.IO artifact.

- get_cellpose_schemas(refresh=False)
    - Returns cached or live-refreshed method schemas from service.<function>.__schema__.

- print_cellpose_runtime_docs(include_schemas=True)
    - Prints this guide plus live schemas.

Training parameter guidance:
- split_mode: use 'auto' for BioImage Archive URL artifacts.
- train_split_ratio: used only in auto split (e.g. 0.8 for 80/20).
- n_samples:
    - if user gives percentage p%, pass n_samples=p/100.
    - valid decimal range: 0 < n_samples <= 1.
    - examples: 0.005 = 0.5%, 0.02 = 2%, 1.0 = 100%.
- validation metrics require test data (manual split) or auto split that creates test data.

Monitoring loop checklist:
- Poll every 2-5 seconds.
- Report: status_type, message, current_epoch/total_epochs, elapsed_seconds.
- Include latest available train/test losses and validation metrics when present.
- Stop polling when status_type is completed/failed/stopped.

Error triage checklist:
- Service resolution failure:
    - call list_cellpose_services(); pick available websocket service ID.
- Artifact/path mismatch:
    - verify artifact workspace and glob patterns.
- Token/workspace mismatch:
    - reconnect with correct token/workspace.
- Resource pressure (VRAM/OOM):
    - stop competing sessions, retry with smaller n_samples and/or lower load.

Examples:
1) Start from artifact paths:
     await start_cellpose_training(
             artifact='workspace/dataset',
             train_images='images/*/*.tif',
             train_annotations='annotations/*/*_mask.ome.tif',
             split_mode='auto',
             train_split_ratio=0.8,
             n_samples=0.005,
             model='cpsam',
             n_epochs=1,
             learning_rate=1e-5,
             min_train_masks=1,
             validation_interval=1,
     )

2) Start from metadata:
     await start_cellpose_training(
             artifact='workspace/dataset',
             metadata_dir='metadata/',
             model='cpsam',
             n_epochs=1,
     )

3) BioImage Archive URL:
     await start_cellpose_training(
             artifact='https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD1392',
             split_mode='auto',
             train_split_ratio=0.8,
             n_samples=0.005,
             n_epochs=1,
             model='cpsam',
     )

Schema usage note:
- To inspect exact callable schemas at runtime, run:
    await print_cellpose_runtime_docs(include_schemas=True)
"""

print(RUNTIME_GUIDE)
