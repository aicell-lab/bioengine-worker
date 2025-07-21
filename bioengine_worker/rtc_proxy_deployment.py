import asyncio
import logging
import uuid
from typing import Any, Callable, Dict, List, Optional

import httpx
from aiortc import RTCPeerConnection
from httpx import AsyncClient, HTTPStatusError, RequestError
from hypha_rpc import connect_to_server, register_rtc_service
from hypha_rpc.utils.schema import schema_function, schema_method
from ray.exceptions import RayTaskError
from ray.serve import deployment
from ray.serve.handle import DeploymentHandle
from starlette.requests import Request

from bioengine_worker.utils import get_pip_requirements

# Configure logging
logger = logging.getLogger("RtcProxyDeployment")

# Maximum number of ongoing requests to limit concurrency and prevent overload
MAX_ONGOING_REQUESTS = int(os.getenv("BIOENGINE_APPLICATION_MAX_ONGOING_REQUESTS", 10))


@deployment(
    ray_actor_options={
        "num_cpus": 1,
        "runtime_env": {
            "pip": get_pip_requirements(select=["aiortc", "httpx", "hypha-rpc"]),
        },
    },
    max_ongoing_requests=MAX_ONGOING_REQUESTS,  # Limit concurrent requests to avoid overload
    max_queued_requests=MAX_ONGOING_REQUESTS * 2,  # Allow some queuing to handle spikes
    autoscaling_config={
        "min_replicas": 1,  # Always keep at least 1 replica running
        "max_replicas": 10,  # Scale up to 10 replicas maximum
        "target_ongoing_requests": 5,  # Target 5 ongoing requests per replica
        "upscale_delay_s": 30.0,  # Wait 30s before scaling up
        "downscale_delay_s": 300.0,  # Wait 5 minutes before scaling down
        "upscale_smoothing_factor": 1.0,  # Aggressive upscaling
        "downscale_smoothing_factor": 0.5,  # Conservative downscaling
    },
    health_check_period_s=10,  # Check health every 10 seconds
    health_check_timeout_s=5,  # Timeout after 5 seconds
)
class RtcProxyDeployment:
    """
    Ray Serve deployment for BioEngine applications, acting as a proxy that registers both WebSocket and WebRTC services with Hypha to bridge Ray Serve applications and external clients.
    Enables efficient data handling via direct peer-to-peer WebRTC connections and standard RPC over WebSocket.

    Key features:
    - Registers services with Hypha for discovery and access
    - Supports authentication and user authorization
    - Monitors health and service registration status for Ray Serve
    - Handles load management and request throttling
    - WebRTC registration is optional; WebSocket is required for health

    Connection Types Supported:
    1. WebSocket Connections: Standard RPC over WebSocket for compatibility
       - Registered as main service with application_id
       - Always available if deployment is healthy
       - Handles all method calls through proxy functions
    2. WebRTC Connections: Direct peer-to-peer connections for high-performance data transfer
       - Registered as "{application_id}-rtc" service
       - Optional - deployment remains healthy if WebRTC registration fails
       - Uses custom ICE servers when available, falls back to defaults
       - Enables direct data channel communication bypassing server

    Architecture Overview:

    External Clients
            ↓
    (With WebSocket: Hypha Server )
            ↓
    RtcProxyDeployment - Ray Cluster
            ↓
    BioEngine Application - Ray Cluster

    """

    def __init__(
        self,
        application_id: str,
        application_name: str,
        application_description: str,
        entry_deployment_handle: DeploymentHandle,
        method_schemas: List[Dict[str, Any]],
        server_url: str,
        workspace: str,
        token: str,
        authorized_users: List[str],
        serve_http_url: str,
    ):
        """
        Initialize the RtcProxyDeployment with BioEngine application configuration.

        This constructor sets up the deployment state and initiates the background
        service registration process with the Hypha server. The deployment will
        automatically register both WebSocket and WebRTC services upon initialization.

        Args:
            application_id: Unique identifier for the BioEngine application.
                          Used as the primary service ID in Hypha registration.
                          Must be unique within the workspace.

            application_name: Human-readable name for the application.
                            Displayed in service discovery interfaces.

            application_description: Detailed description of the application's functionality.
                                   Used for documentation and service catalogs.

            entry_deployment_handle: Ray Serve deployment handle for the actual
                                   BioEngine application. All method calls will
                                   be forwarded to this deployment.

            method_schemas: List of JSON schema definitions for application methods.
                          Each schema must contain 'name', 'description', and 'parameters'.
                          Used for automatic proxy function generation and validation.

            server_url: Hypha server endpoint URL. Typically 'https://hypha.aicell.io'
                       for production or custom server for development.

            workspace: Hypha workspace identifier for service organization.
                      Can be None for default workspace. Used for multi-tenancy.

            token: Authentication token for Hypha server access.
                  Required for service registration and must have appropriate permissions.

            authorized_users: List of user identifiers (IDs or emails) allowed to access
                            this application. Use ["*"] for public access.
                            Empty list denies all access.
        """
        # BioEngine application metadata
        self.application_id = application_id
        self.application_name = application_name
        self.application_description = application_description
        self.entry_deployment_handle = entry_deployment_handle
        self.method_schemas = method_schemas
        self.max_ongoing_requests = MAX_ONGOING_REQUESTS

        # Hypha server connection parameters
        self.server_url = server_url
        self.workspace = workspace
        self.token = token
        self.authorized_users = authorized_users

        # Service state
        self.server = None
        self.rtc_service_id: Optional[str] = None
        self.service_id: Optional[str] = None
        self.service_semaphore = asyncio.Semaphore(self.max_ongoing_requests)

        # Store connection event handlers
        self._connection_handlers: List[Callable] = []

        # Store request events
        self.serve_http_url = serve_http_url
        self._request_events: Dict[str, asyncio.Event] = {}

    async def __call__(self, request: Request) -> Dict[str, Any]:
        """
        **RAY SERVE HTTP ENDPOINT - AUTOSCALING COORDINATION**

        Handle HTTP requests from Ray Serve's HTTP proxy to coordinate autoscaling.
        This endpoint waits for the corresponding WebRTC request to complete, ensuring
        Ray Serve can track request load for autoscaling decisions.

        Only accepts POST requests with X-Request-ID header from _mimic_request().

        Args:
            request: Starlette Request object from Ray Serve HTTP proxy

        Returns:
            JSON response with completion status
        """
        # Only accept POST requests for mimic coordination
        if request.method != "POST":
            return {
                "status": "error",
                "message": f"Method {request.method} not supported - only POST allowed",
            }

        request_id = request.headers.get("X-Request-ID")
        if not request_id:
            return {
                "status": "error",
                "message": "Missing X-Request-ID header",
            }

        # Wait for the corresponding request event
        event = self._request_events.get(request_id)
        if event:
            await event.wait()
            return {"status": "completed", "request_id": request_id}
        else:
            # Request may have already completed
            return {"status": "not_found", "request_id": request_id}

    # ===== Hypha Service Registration =====
    # Handles registration of WebSocket and WebRTC services with Hypha.

    async def _fetch_ice_servers(self) -> Optional[List[Dict[str, Any]]]:
        """
        **HYPHA SERVICE REGISTRATION - ICE SERVERS**

        Fetch custom ICE (Interactive Connectivity Establishment) servers from
        the Hypha infrastructure for WebRTC connections. These servers help
        establish peer-to-peer connections through NAT and firewall traversal.

        ICE servers are used by WebRTC to:
        1. Discover public IP addresses (STUN servers)
        2. Relay traffic when direct connection fails (TURN servers)
        3. Enable connections across different network configurations

        The method attempts to fetch custom ICE servers from the Hypha endpoint,
        but gracefully falls back to None if unavailable, allowing hypha-rpc
        to use its built-in default servers.

        ICE Server Format:
            [{"urls": "stun:stun.server.com:19302"},
             {"urls": "turn:turn.server.com:3478", "username": "...", "credential": "..."}]
        """
        try:
            async with AsyncClient(timeout=30) as client:
                response = await client.get(
                    "https://ai.imjoy.io/public/services/coturn/get_rtc_ice_servers"
                )
                response.raise_for_status()
                ice_servers = response.json()
                logger.info(
                    f"Successfully fetched ICE servers for {self.application_id}"
                )
                return ice_servers
        except HTTPStatusError as e:
            logger.error(
                f"HTTP error fetching ICE servers for {self.application_id}: {e}"
            )
        except RequestError as e:
            logger.error(
                f"Request error fetching ICE servers for {self.application_id}: {e}"
            )
        except Exception as e:
            logger.error(
                f"Unexpected error fetching ICE servers for {self.application_id}: {e}"
            )

        return None

    async def _on_webrtc_init(self, peer_connection: RTCPeerConnection) -> None:
        """
        ** Hypha Service Registration - WEBRTC CONNECTION LIFECYCLE MANAGEMENT**

        Callback handler invoked by hypha-rpc when a new WebRTC peer connection
        is established. Sets up connection state monitoring and integrates WebRTC
        connection events with the deployment's operational state.

        This method is registered as the "on_init" callback during WebRTC service
        registration and follows the hypha-rpc WebRTC connection initialization pattern.

        ## Connection State Monitoring
        Registers event handlers for WebRTC connection state changes:
        - "connected": Successful peer-to-peer connection established
        - "failed": Connection failed due to network, NAT, or other issues
        - "closed": Connection terminated by either peer
        - "disconnected": Connection interrupted, may be temporary

        ## Integration with Ray Serve Health System
        While WebRTC connection failures don't directly affect Ray Serve health checks,
        the connection state is monitored for:
        - Operational visibility and debugging
        - Future integration with advanced health criteria
        - Client connection quality metrics

        ## Connection Cleanup Management
        Stores connection event handlers in self._connection_handlers for proper
        cleanup during deployment shutdown, preventing memory leaks and ensuring
        graceful WebRTC resource management.
        """
        try:
            logger.info(
                f"WebRTC peer connection initialized for '{self.application_id}'"
            )

            # Set up connection state monitoring
            @peer_connection.on("connectionstatechange")
            def on_connection_state_change():
                state = peer_connection.connectionState
                logger.info(
                    f"WebRTC connection state changed to '{state}' for '{self.application_id}'"
                )

            # Store handler reference for cleanup
            self._connection_handlers.append(on_connection_state_change)

        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize WebRTC connection for '{self.application_id}': {e}"
            )

    async def _check_permissions(self, context: Optional[Dict[str, Any]]) -> None:
        """
        **HYPHA SERVICE REGISTRATION - PERMISSION VALIDATION**

        Check if the user in the request context is authorized to access this
        deployment. This method is called by proxy functions created during
        service registration to enforce access control.

        Permission Checking Process:
        1. Validates context contains user information
        2. Extracts user ID and email from context
        3. Checks against authorized_users list from deployment config
        4. Supports wildcard access ("*") for public deployments

        Authorization Methods:
        - Wildcard "*" in authorized_users allows all users
        - User ID match against authorized_users list
        - Email address match against authorized_users list

        Raises:
            PermissionError: If user is not authorized or context is invalid
        """
        if not isinstance(context, dict) or "user" not in context:
            raise PermissionError("Invalid context without user information")

        user = context["user"]
        if not isinstance(user, dict) or ("id" not in user and "email" not in user):
            raise PermissionError("Invalid user information in context")

        # Check authorization
        user_id = user["id"]
        user_email = user["email"]
        if (
            "*" not in self.authorized_users  # Wildcard access
            and user_id not in self.authorized_users
            and user_email not in self.authorized_users
        ):
            raise PermissionError(
                f"User '{user_id}' ({user_email}) is not authorized to access application '{self.application_id}'"
            )

    async def _mimic_request(self, request_id: str) -> None:
        """
        **HYPHA SERVICE REGISTRATION - AUTOSCALING TRIGGER HTTP REQUEST TO RAY SERVE**

        Send an HTTP request to Ray Serve to trigger autoscaling when users access
        the application through WebRTC connections. The request uses the same ID
        as the actual RPC call for coordination with __call__.

        Args:
            request_id: Unique identifier for correlating with the RPC call
        """
        try:
            timeout = httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0)

            async with httpx.AsyncClient(timeout=timeout) as client:
                await client.post(
                    f"{self.serve_http_url}/{self.application_id}/",
                    headers={"X-Request-ID": request_id},
                    json={"mimic_request": True},
                )

        except Exception as e:
            # Log but don't fail the user request
            logger.warning(
                f"Failed to send autoscaling trigger for '{self.application_id}' "
                f"request {request_id}: {e}"
            )

    def _create_deployment_function(
        self, method_schema: Dict[str, Any]
    ) -> Callable[..., Any]:
        """
        **HYPHA SERVICE REGISTRATION - PROXY FUNCTION CREATION**

        Create a proxy function that forwards Hypha RPC requests to the
        underlying Ray Serve deployment. This enables BioEngine applications
        to be accessed through Hypha's RPC system while maintaining proper
        authentication, logging, and error handling.

        Proxy Function Responsibilities:
        1. Validate user permissions via _check_permissions()
        2. Log method calls for audit and debugging
        3. Forward requests to the actual Ray deployment
        4. Handle Ray-specific errors and convert to appropriate exceptions
        5. Maintain method schema compliance for Hypha RPC

        The generated proxy function:
        - Uses @schema_function decorator for Hypha RPC integration
        - Forwards all arguments and keyword arguments to Ray deployment
        - Provides detailed error logging and user identification
        - Handles both RayTaskError and general exceptions

        Args:
            method_schema: Schema definition for the method containing:
                - name: Method name
                - description: Method description
                - parameters: Parameter schema for validation
        """
        method_name = method_schema["name"]

        async def deployment_function(*args, context, **kwargs) -> Any:
            async with self.service_semaphore:
                request_id = str(uuid.uuid4())
                try:
                    # Check user permissions
                    await self._check_permissions(context)

                    # Log the method call
                    user_info = context.get("user", {}) if context else {}
                    user_id = user_info.get("id", "unknown")
                    logger.info(
                        f"User {user_id} calling method '{method_name}' on app '{self.application_id}'"
                    )

                    # Mimic a request to trigger autoscaling (do not block)
                    self._request_events[request_id] = asyncio.Event()

                    # Create the mimic request task but don't await it to avoid blocking
                    mimic_task = asyncio.create_task(self._mimic_request(request_id))

                    # Add error handling for the mimic task (optional - runs in background)
                    def handle_mimic_error(task):
                        if task.exception():
                            logger.warning(
                                f"Mimic request task failed for '{self.application_id}' "
                                f"request ID {request_id}: {task.exception()}"
                            )

                    mimic_task.add_done_callback(handle_mimic_error)

                    # Get the method from the entry deployment handle
                    method = getattr(self.entry_deployment_handle, method_name, None)
                    if method is None:
                        raise AttributeError(
                            f"Method '{method_name}' not found on entry deployment"
                        )

                    # Forward the request to the actual deployment
                    try:
                        result = await method.remote(*args, **kwargs)
                        logger.debug(
                            f"Successfully executed method '{method_name}' for user {user_id}"
                        )
                        return result
                    except RayTaskError as e:
                        logger.error(f"Ray task error in method '{method_name}': {e}")
                        raise
                    except Exception as e:
                        logger.error(f"Unexpected error in method '{method_name}': {e}")
                        raise

                except PermissionError as e:
                    logger.warning(f"Permission denied for method '{method_name}': {e}")
                    raise
                except Exception as e:
                    logger.error(f"Error in proxy function '{method_name}': {e}")
                    raise
                finally:
                    # Remove the event to prevent memory leaks
                    event = self._request_events.pop(request_id, None)

                    # Signal that the request is complete
                    if event:
                        event.set()

        return schema_function(
            func=deployment_function,
            name=method_schema["name"],
            description=method_schema["description"],
            parameters=method_schema["parameters"],
        )

    @schema_method
    async def _get_load(self) -> float:
        """
        **HYPHA SERVICE REGISTRATION - SERVICE LOAD MONITORING**

        Returns the current load of the service as a float value between 0 and 1.
        This method is used by Hypha's load balancing system to distribute requests
        across multiple service instances.

        Load Calculation:
        - 0.0: No active requests, service is idle
        - 1.0: Maximum capacity reached, all semaphore slots occupied
        - Values between indicate partial load

        The load is calculated based on the number of active requests being processed
        through the service semaphore, which limits concurrent request processing.
        """
        # Calculate load based on available vs. total semaphore capacity
        available_slots = self.service_semaphore._value
        total_slots = self.max_ongoing_requests
        active_requests = total_slots - available_slots
        load = active_requests / total_slots

        logger.debug(
            f"Service load for '{self.application_id}': {load:.2f} "
            f"({active_requests}/{total_slots} active requests)"
        )

        return min(1.0, max(0.0, load))  # Ensure load is between 0 and 1

    async def _register_web_rtc_service(self) -> None:
        """
        **HYPHA SERVICE REGISTRATION - MAIN REGISTRATION**

        Register WebSocket and WebRTC services with Hypha.

        Connects to Hypha, fetches ICE servers, registers services, and sets up proxy functions.
        WebRTC registration is optional; main service registration is required for health.
        """
        # Connect to Hypha server
        try:
            logger.info(
                f"Connecting to Hypha server for application '{self.application_id}'"
            )

            # Connect to the Hypha server
            self.server = await connect_to_server(
                {
                    "server_url": self.server_url,
                    "token": self.token,
                    "workspace": self.workspace,
                }
            )
            logger.info(
                f"Successfully connected to Hypha server for '{self.application_id}'"
            )
        except Exception as e:
            self.server = None
            logger.error(
                f"Error connecting to Hypha server for '{self.application_id}': {e}"
            )
            raise

        # Register WebRTC service with custom ICE servers or fallback to defaults
        try:
            # Fetch custom ICE servers
            ice_servers = await self._fetch_ice_servers()

            # Prepare WebRTC config
            rtc_config = {
                "visibility": "public",
                "on_init": self._on_webrtc_init,  # Add WebRTC connection handler
            }

            # Add custom ICE servers if available, otherwise hypha-rpc will use defaults
            if ice_servers:
                rtc_config["ice_servers"] = ice_servers
                logger.info(
                    f"Using custom ICE servers for WebRTC service '{self.application_id}'"
                )
            else:
                logger.info(
                    f"Using default ICE servers for WebRTC service '{self.application_id}'"
                )

            # Register WebRTC service
            rtc_service_info = await register_rtc_service(
                self.server,
                service_id=f"{self.application_id}-rtc",
                config=rtc_config,
            )
            self.rtc_service_id = rtc_service_info["id"]
            logger.info(
                f"Registered WebRTC service for '{self.application_id}' with ID: {self.rtc_service_id}"
            )

        except Exception as e:
            logger.error(
                f"Failed to register WebRTC service for '{self.application_id}': {e}"
            )
            # Don't fail the entire deployment if WebRTC registration fails

        try:
            # Create service functions from method schemas
            service_functions = {}
            for method_schema in self.method_schemas:
                method_name = method_schema["name"]
                service_functions[method_name] = self._create_deployment_function(
                    method_schema
                )
                logger.debug(f"Created proxy function for method '{method_name}'")

            # Add load check function - for service load balancing (https://docs.amun.ai/#/service-load-balancing)
            service_functions["get_load"] = self._get_load

            logger.info(
                f"Registering service functions for '{self.application_id}': {list(service_functions.keys())}"
            )

            # Register the main service
            service_info = await self.server.register_service(
                {
                    "id": self.application_id,
                    "name": self.application_name,
                    "type": "bioengine-apps",
                    "description": self.application_description,
                    "config": {"visibility": "public", "require_context": True},
                    **service_functions,
                },
                {"overwrite": True},
            )

            self.service_id = service_info["id"]
            logger.info(
                f"Successfully registered service for '{self.application_id}' with ID: {self.service_id}"
            )

        except Exception as e:
            self.service_id = None
            logger.error(
                f"Error registering WebRTC service for '{self.application_id}': {e}"
            )
            raise

    # ===== BioEngine Worker Interface =====
    # Methods for external management and service discovery.

    @schema_method
    async def get_service_ids(self) -> Dict[str, str]:
        """
        **BIOENGINE WORKER INTERFACE - SERVICE DISCOVERY**

        Returns dictionary of registered service IDs:
        - "websocket_service_id": ID of the main WebSocket service
        - "webrtc_service_id": ID of the WebRTC service (if registered)

        Raises RuntimeError if registration failed.
        """
        if self.service_id is None:
            raise RuntimeError(
                f"Service registration failed for '{self.application_id}'"
            )

        return {
            "websocket_service_id": self.service_id,
            "webrtc_service_id": self.rtc_service_id,
        }

    # ===== Ray Serve Health Check =====
    # Implements periodic health checks for Ray Serve.

    async def check_health(self):
        """
        **RAY SERVE HEALTH CHECK**

        Ray Serve health check. Ensures Hypha connection and main service registration.
        Triggers registration if needed. Raises RuntimeError if unhealthy.
        Called during deployment initialization and periodically.
        """

        if not self.service_id:
            await self._register_web_rtc_service()

        # Check if Hypha server connection is established
        if self.server is None:
            raise RuntimeError("Hypha server connection not established")

        # Note: WebRTC service registration is optional and doesn't affect health

        # Check if service registration is complete
        if self.service_id is None:
            raise RuntimeError("Application service not registered")

        # All checks passed - deployment is healthy

    async def reconfigure(self, version) -> None:
        """
        **RAY SERVE RECONFIGURATION**

        Reconfigure the deployment with new settings. This method is called by Ray Serve
        when the deployment configuration is updated dynamically.

        Args:
            version: Configuration version for the reconfigure operation
        """
        # user_config: Config to pass to the reconfigure method of the deployment. This
        # can be updated dynamically without restarting the replicas of the
        # deployment. The user_config must be fully JSON-serializable.
        pass


if __name__ == "__main__":
    import os

    class MockMethod:
        async def remote(self, *args, **kwargs):
            return f"Mocked method called with args={args}, kwargs={kwargs}"

    class MockHandle:
        def __getattr__(self, name):
            return MockMethod()

    # Example usage of RtcProxyDeployment
    async def main():
        rtc_deployment_class = RtcProxyDeployment.func_or_class
        entry_deployment_handle = MockHandle()
        method_schema = {
            "name": "test_method",
            "description": "A test method for demonstration",
            "parameters": {
                "properties": {"test": {"description": "test", "type": "string"}},
                "required": ["test"],
                "type": "object",
            },
        }

        deployment = rtc_deployment_class(
            application_id="test-app",
            application_name="Test Application",
            application_description="A test application for demonstration",
            entry_deployment_handle=entry_deployment_handle,
            workspace=None,
            server_url="https://hypha.aicell.io",
            token=os.environ["HYPHA_TOKEN"],
            method_schemas=[method_schema],
            authorized_users=["*"],
        )

        await deployment.check_health()

        service_ids = await deployment.get_service_ids()
        print(f"Service IDs: {service_ids}")

    asyncio.run(main())
