import asyncio
import os
from typing import Any, Callable, Dict, List, Tuple, Union

import httpx
from hypha_rpc import connect_to_server, register_rtc_service
from ray import serve
from ray.serve.handle import DeploymentHandle


@serve.deployment
class RtcProxyDeployment:
    def __init__(
        self,
        application_id: str,
        application_name: str,
        entry_deployment: DeploymentHandle,
        exposed_methods: List[str],
        authorized_users: Union[List[str], str],
    ):
        # BioEngine application metadata
        self.application_id = application_id
        self.application_name = application_name
        self.entry_deployment = entry_deployment
        self.exposed_methods = exposed_methods
        self._check_exposed_methods()

        self.server_url = os.environ["HYPHA_SERVER_URL"]
        self.workspace = os.environ["HYPHA_WORKSPACE"]
        self.token = os.environ["HYPHA_TOKEN"]
        self.authorized_users = authorized_users

        self.service_task = asyncio.create_task(
            self.register_web_rtc_service(),
            name=f"rtc_proxy_service_{application_name}",
        )

    def _check_exposed_methods(self) -> None:
        """
        Check if the exposed methods are valid and exist in the deployment handle.

        Raises:
            ValueError: If any of the exposed methods do not exist in the deployment handle.
        """
        for method_name in self.exposed_methods:
            if not hasattr(self.entry_deployment, method_name):
                raise ValueError(
                    f"Method '{method_name}' is not defined in the entry deployment handle."
                )
            if not callable(getattr(self.entry_deployment, method_name)):
                raise ValueError(
                    f"Method '{method_name}' is not callable in the entry deployment handle."
                )

    async def _fetch_ice_servers(self):
        ice_servers = None

        # Fetch Hypha ICE servers
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(
                    "https://ai.imjoy.io/public/services/coturn/get_rtc_ice_servers"
                )
                if response.status_code == 200:
                    ice_servers = response.json()
                    print("Successfully fetched ICE servers:", ice_servers)
                    return ice_servers
                else:
                    print(
                        f"Failed to fetch ICE servers, status: {response.status_code}"
                    )
        except Exception as e:
            print(f"Error fetching ICE servers: {e}")

        # Fallback to default ICE servers if fetching fails
        if not ice_servers:
            print("Using fallback ICE servers")
            ice_servers = [{"urls": ["stun:stun.l.google.com:19302"]}]

        return ice_servers

    def _check_permissions(self, context: Dict[str, str]) -> None:
        """
        Check if the user in the context is authorized to access the deployment.

        Validates user permissions against the authorized users list for specific
        deployment operations.

        Args:
            context: Request context containing user information

        Returns:
            bool: True if user is authorized

        Raises:
            PermissionError: If user is not authorized to access the resource
        """
        if context is None or "user" not in context:
            raise PermissionError("Context is missing user information")
        user = context["user"]
        if isinstance(self.authorized_users, str):
            self.authorized_users = [self.authorized_users]
        if (
            "*" not in self.authorized_users
            and user["id"] not in self.authorized_users
            and user["email"] not in self.authorized_users
        ):
            raise PermissionError(
                f"User {user['id']} is not authorized to call application '{self.application_id}'."
            )

    def _create_deployment_function(self, method_name: str) -> Callable[..., Any]:
        """
        Create a deployment function for the specified method name.
        """

        async def deployment_function(*args, context=None, **kwargs) -> Any:
            # Check if the user is authorized to access the application
            await self._check_permissions(context)

            # Get the method from the deployment handle
            method = getattr(self.deployment, method_name)

            # Forward the request to the actual deployment
            return await method.remote(*args, **kwargs)

        return deployment_function

    async def register_web_rtc_service(self) -> None:
        """
        Register the WebRTC service and the deployment handle with the Hypha server.
        """
        try:
            # Connect to the Hypha server
            server = await connect_to_server(
                {
                    "server_url": self.server_url,
                    "workspace": self.workspace,
                    "token": self.token,
                }
            )

            # Fetch ICE servers
            # ice_servers = await self._fetch_ice_servers()

            # Register the WebRTC service
            # rtc_service_info = await register_rtc_service(
            #     server,
            #     service_id=f"{self.application_id}-rtc",
            #     config={
            #         "visibility": "public",
            #         "ice_servers": ice_servers,
            #     },
            # )
            # rtc_service_id = rtc_service_info["id"]
            # print(f"Registered WebRTC service for '{self.application_id}' with ID: {rtc_service_id}")

            # Register the service with the deployment handle
            service_functions = [
                self._create_deployment_function(method_name)
                for method_name in self.exposed_methods
            ]
            service_info = await server.register_service(
                {
                    "id": self.application_id,
                    "name": self.application_name,
                    "type": "bioengine-apps",
                    "description": "BioEngine application",
                    "config": {"visibility": "public", "require_context": True},
                    **service_functions,
                },
                {"overwrite": True},
            )
            service_id = service_info["id"]
            print(
                f"Registered service for '{self.application_id}' with ID: {service_id}"
            )

            # Keep the service running
            await server.serve()
        except Exception as e:
            # TODO: Crash the deployment if the service fails
            os._exit(1)
