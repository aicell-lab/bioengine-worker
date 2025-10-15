# Minimal deploy flow
import asyncio
import os

from hypha_rpc import connect_to_server, login


async def deploy():
    server_url = os.environ.get("HYPHA_SERVER_URL", "https://hypha.aicell.io")
    token = os.environ.get("HYPHA_TOKEN") or await login({"server_url": server_url})
    hypha = await connect_to_server({"server_url": server_url, "token": token})
    workspace = hypha.config.workspace
    print("Using workspace:", workspace)

    worker = await hypha.get_service("bioimage-io/bioengine-worker")

    app_id = await worker.deploy_application(
        artifact_id="ri-scale/cellpose-finetuning",
        application_id="cellpose-finetuning",
        hypha_token=token,
        version=None,  # latest
        disable_gpu=False,  # set True to force CPU-only
        max_ongoing_requests=1,  # keep at 1 for GPU
    )
    print("App ID:", app_id)

    status = await worker.get_status()
    service_ids = status["bioengine_apps"][app_id]["service_ids"]
    print("Services:", service_ids)
    return service_ids


asyncio.run(deploy())
