from typing import List, Callable
from hypha_rpc import login, connect_to_server
from config import Config
import os

class Hypha:

    @staticmethod
    async def connect(token: str):
        result = None
        try:
            result = await connect_to_server(
                {
                    "server_url": Config.server_url,
                    "workspace": Config.workspace_name,
                    "client_id": Config.client_id,
                    "name": "Berzelius",
                    "token": token,
                }
            )
        except Exception as e:
            print("Connecting failed:", e)
        return result

    @staticmethod
    async def retrieve_token():
        return await login({"server_url": Config.server_url})
    
    @staticmethod
    async def authenticate():
        token = os.getenv(Config.TOKEN_VAR_NAME, None)
        if not token:
            print(f"Expected token from environment variable {Config.TOKEN_VAR_NAME}")
            return None
        return await Hypha.connect(token)

    @staticmethod
    def _get_services(callbacks: List[Callable]):
        services = {
            "name": Config.service_name,
            "id": Config.service_id,
            "config": {
                "visibility": "public",
                "require_context": True,
            }
        }
        for callback in callbacks:
            services[callback.__name__] = callback
        return services

    @staticmethod
    async def register_service(server_handle, callbacks: List[Callable]):
        return await server_handle.register_service(Hypha._get_services(callbacks), {"overwrite": True})

    @staticmethod
    def print_services(service_info, callbacks: List[Callable]):
        sid = service_info["id"]
        assert sid == f"{Config.workspace_name}/{Config.client_id}:{Config.service_id}"
        print(f"Registered service with ID: {sid}")
        for callback in callbacks:
            print(f"Test the service at: {Config.server_url}/{Config.workspace_name}/services/{Config.client_id}:{Config.service_id}/{callback.__name__}")

