from typing import List, Callable
from hypha_rpc import login, connect_to_server
import asyncio

class Config:
    server_url = "https://hypha.aicell.io"
    workspace_name = "hpc-ray-cluster"
    service_id = "ray"
    client_id = "berzelius"
    service_name = "Ray"

class Hypha:
    async def _connect(token: str):
        return await connect_to_server(
        {
            "server_url": Config.server_url,
            "workspace": Config.workspace_name,
            "client_id": Config.client_id,
            "name": "Berzelius",
            "token": token,
        }
    )
    async def _login():
        return await login({"server_url": Config.server_url})
    
    async def authenticate():
        return await Hypha._connect(await Hypha._login())
    
    async def register_service(server_handle, callback: Callable):
        return await server_handle.register_service(
        {
            "name": Config.service_name,
            "id": Config.service_id,
            "config": {
                "visibility": "public",
                "require_context": True,  # TODO: only allow the service to be called by logged-in users
            },
            callback.__name__: callback,
        }, {"overwrite": True}
    )

    def print_service_info(service_info, callback: Callable):
        sid = service_info["id"]
        assert sid == f"{Config.workspace_name}/{Config.client_id}:{Config.service_id}"
        print(f"Registered service with ID: {sid}")
        print(f"Test the service at: {Config.server_url}/{Config.workspace_name}/services/{Config.client_id}:{Config.service_id}/{callback.__name__}")

async def register_service():
    def inference_task(context = None):
        return "my_inference_result"
    colab_client = await Hypha.authenticate()
    service_info = await Hypha.register_service(server_handle=colab_client, callback=inference_task)
    Hypha.print_service_info(service_info=service_info, callback=inference_task)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.create_task(register_service())
    loop.run_forever()
