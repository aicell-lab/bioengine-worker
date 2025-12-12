import asyncio
import logging
import os
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

import httpx
import ray
from httpx import AsyncClient, HTTPStatusError, RequestError
from hypha_rpc import connect_to_server, register_rtc_service
from hypha_rpc.rpc import RemoteService
from hypha_rpc.utils.schema import schema_function, schema_method
from pydantic import Field
from ray.exceptions import RayTaskError
from ray.serve import deployment, get_replica_context
from ray.serve.handle import DeploymentHandle
from starlette.requests import Request

from bioengine.utils import get_pip_requirements

logger = logging.getLogger("ray.serve")


@deployment(
    ray_actor_options={
        "num_cpus": 0,
        "runtime_env": {
            "pip": get_pip_requirements(
                select=["aiortc", "httpx", "hypha-rpc", "pydantic"],
                extras=["rtc"],
            ),
        },
    },
    max_ongoing_requests=10,  # Default limit concurrent requests to avoid overload
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
class BioEngineProxyDeployment:
    """
    A Ray Serve deployment that acts as a proxy for BioEngine applications.

    This deployment bridges BioEngine applications running in Ray Serve with external
    clients by registering both WebSocket and WebRTC services with the Hypha server.
    It enables efficient data handling through direct peer-to-peer WebRTC connections
    while also supporting standard RPC communication over WebSocket.

    Key Features:
    - Service discovery through Hypha registration
    - User authentication and authorization
    - Health monitoring for Ray Serve
    - Request load management and throttling
    - Dual connection support (WebSocket + WebRTC)

    Connection Types:

    1. WebSocket Connections
       - Standard RPC communication for broad compatibility
       - Always available when deployment is healthy
       - Registered using the application ID as service ID
       - All method calls routed through proxy functions

    2. WebRTC Connections
       - Direct peer-to-peer connections for high-performance data transfer
       - Optional feature - deployment works without it
       - Registered as "{application_id}-rtc" service
       - Uses custom ICE servers when available
       - Bypasses server for direct client communication

    Architecture:
    External Clients → Hypha Server → BioEngineProxyDeployment → BioEngine Application
                                           (Ray Cluster)             (Ray Cluster)
    """

    def __init__(
        self,
        application_id: str,
        application_name: str,
        application_description: str,
        entry_deployment_handle: DeploymentHandle,
        method_schemas: List[Dict[str, Any]],
        max_ongoing_requests: int,
        server_url: str,
        workspace: str,
        proxy_service_token: str,
        worker_client_id: str,
        authorized_users: List[str],
        serve_http_url: str,
        proxy_actor_name: str,
    ):
        """
        Initialize the BioEngine proxy deployment.

        Sets up the deployment state and prepares for service registration with the Hypha server.
        The deployment will automatically register WebSocket and WebRTC services when health
        checks are performed.

        Args:
            application_id: Unique identifier for the BioEngine application. Used as the
                          primary service ID in Hypha. Must be unique within the workspace.

            application_name: Human-readable name displayed in service discovery interfaces.

            application_description: Detailed description of the application's functionality
                                   for documentation and service catalogs.

            entry_deployment_handle: Ray Serve handle to the actual BioEngine application.
                                   All method calls are forwarded to this deployment.

            method_schemas: JSON schema definitions for application methods. Each schema
                          needs 'name', 'description', and 'parameters' fields for
                          automatic proxy function generation.

            server_url: Hypha server endpoint URL. Usually 'https://hypha.aicell.io'
                       for production or custom server for development.

            workspace: Hypha workspace identifier for service organization. Can be None
                      for default workspace. Used for multi-tenancy.

            proxy_service_token: Authentication token for Hypha server access. Required for service
                  registration with appropriate permissions.

            worker_client_id: Client ID of the worker that created this deployment.

            authorized_users: List of user identifiers (IDs or emails) allowed to access
                            this application. Use ["*"] for public access or empty list
                            to deny all access.

            serve_http_url: URL for Ray Serve HTTP endpoint used for autoscaling coordination.
            proxy_actor_name: Actor name of the BioEngineProxyActor for replica registration.
        """
        logger.info(
            f"🚀 Initializing BioEngineProxyDeployment for application: '{application_id}'"
        )
        logger.info(f"🔗 Server URL: {server_url}")
        logger.info(f"🏢 Workspace: '{workspace}'")
        logger.info(f"👥 Authorized users: {authorized_users}")
        logger.info(f"⚙️ Max ongoing requests: {max_ongoing_requests}")

        # Register this serve replica with the BioEngineProxyActor for tracking
        proxy_actor_handle = None
        replica_id = None

        try:
            proxy_actor_handle = ray.get_actor(
                name=proxy_actor_name, namespace="bioengine"
            )
            logger.debug(
                f"✅ Successfully retrieved BioEngineProxyActor '{proxy_actor_name}'"
            )
        except ValueError as e:
            logger.error(f"❌ BioEngineProxyActor '{proxy_actor_name}' not found: {e}")
        except Exception as e:
            logger.error(
                f"❌ Unexpected error getting BioEngineProxyActor '{proxy_actor_name}': {e}",
                exc_info=True,
            )

        if proxy_actor_handle:
            try:
                replica_context = get_replica_context()
                deployment_name = replica_context.deployment
                replica_id = replica_context.replica_tag

                # Get timezone at replica level
                replica_timezone = time.strftime("%Z")

                logger.debug(
                    f"Retrieved replica context: tag={replica_id}, "
                    f"deployment={replica_context.deployment}, "
                    f"app={replica_context.app_name}, "
                    f"timezone={replica_timezone}"
                )

                proxy_actor_handle.register_serve_replica.remote(
                    application_id=application_id,
                    deployment_name=deployment_name,
                    replica_id=replica_id,
                    timezone=replica_timezone,
                )
                logger.info(
                    f"✅ Registered replica '{replica_id}' with BioEngineProxyActor."
                )
            except ValueError as e:
                logger.error(f"❌ Invalid replica registration parameters: {e}")
            except Exception as e:
                logger.error(
                    f"❌ Unable to register replica with BioEngineProxyActor: {e}",
                    exc_info=True,
                )

        # BioEngine application metadata
        self.application_id = application_id
        self.application_name = application_name
        self.application_description = application_description
        self.entry_deployment_handle = entry_deployment_handle
        self.method_schemas = method_schemas
        self.max_ongoing_requests = max_ongoing_requests
        self.authorized_users = authorized_users

        # Hypha server connection parameters
        self.server_url = server_url
        self.workspace = workspace
        self.proxy_service_token = proxy_service_token

        if replica_id:
            self.client_id = f"{worker_client_id}-{replica_id}"
        else:
            # Fallback to generating a UUID-based ID for hypha client_id
            uuid_str = f"uuid-{str(uuid.uuid4())[:8]}"
            self.client_id = f"{worker_client_id}-{uuid_str}"
            logger.warning(
                f"⚠️ Using fallback client_id '{self.client_id}' due to missing replica ID."
            )

        # Store entry deployment readiness
        self.entry_deployment_ready = False

        # Service state
        self.server: RemoteService = None
        self.websocket_service_id: str = None
        self.mcp_service_id: str = None
        self.rtc_service_id: str = None
        self.service_semaphore = asyncio.Semaphore(self.max_ongoing_requests)

        # WebRTC peer connection tracking
        self._active_peer_connections: Dict[str, Dict[str, Any]] = {}

        # Store request events
        self.serve_http_url = serve_http_url
        self._request_events: Dict[str, Dict[str, Any]] = {}

        # Lock for service registration
        self._registration_lock = asyncio.Lock()

        # Start background cleanup task
        self._cleanup_task = None

    async def __call__(self, request: Request) -> Dict[str, Any]:
        """
        Handle HTTP requests for Ray Serve autoscaling coordination.

        This endpoint receives HTTP requests from Ray Serve's HTTP proxy and waits for
        corresponding WebRTC requests to complete. This ensures Ray Serve can properly
        track request load for autoscaling decisions.

        Only accepts POST requests with an X-Request-ID header that matches requests
        initiated by the _mimic_request() method.

        Args:
            request: HTTP request object from Ray Serve's HTTP proxy

        Returns:
            Dictionary with completion status and request ID
        """
        logger.info(
            f"🌐 Received '{request.method}' request to BioEngineProxyDeployment"
        )

        # Only accept POST requests for mimic coordination
        if request.method != "POST":
            logger.error(
                f"❌ Method '{request.method}' not supported - only POST allowed"
            )
            return {
                "status": "error",
                "message": f"Method '{request.method}' not supported - only POST allowed",
            }

        request_id = request.headers.get("X-Request-ID")
        if not request_id:
            logger.error(f"❌ Missing X-Request-ID header in request")
            return {
                "status": "error",
                "message": "Missing X-Request-ID header",
            }

        logger.info(f"⏳ Waiting for request: {request_id}")
        # Wait for the corresponding request event
        event_data = self._request_events.get(request_id)
        if event_data:
            try:
                await asyncio.wait_for(
                    event_data["event"].wait(), timeout=3600
                )  # free after 1 hour
                logger.info(f"✅ Request completed: {request_id}")
                return {"status": "completed", "request_id": request_id}
            except asyncio.TimeoutError:
                logger.info(f"⏱️ Request timed out after 1 hour: {request_id}")
                return {"status": "timeout", "request_id": request_id}
            finally:
                # Always remove the event to prevent memory leaks
                self._request_events.pop(request_id, None)
        else:
            logger.error(f"❌ Request not found: {request_id}")
            # Request may have already completed
            return {"status": "not_found", "request_id": request_id}

    # ===== Hypha Service Registration =====
    # Handles registration of WebSocket and WebRTC services with Hypha.

    async def _check_permissions(self, context: Dict[str, Any]) -> None:
        """
        Verify that a user is authorized to access this deployment.

        This method is called by proxy functions during service registration to
        enforce access control based on the authorized_users list configured
        during deployment initialization.

        Permission Checking Process:
        1. Validates that context contains user information
        2. Extracts user ID and email from the context
        3. Checks against the authorized_users list
        4. Supports wildcard access ("*") for public deployments

        Authorization Methods:
        - Wildcard "*" in authorized_users allows all users
        - User ID match against authorized_users list
        - Email address match against authorized_users list

        Args:
            context: Request context containing user information from Hypha

        Raises:
            PermissionError: If user is not authorized or context is invalid
        """
        logger.info(f"🔒 Checking permissions for application: {self.application_id}")

        if not isinstance(context, dict) or "user" not in context:
            logger.error(f"❌ Invalid context without user information")
            raise PermissionError("Invalid context without user information")

        user = context["user"]
        if not isinstance(user, dict) or ("id" not in user and "email" not in user):
            logger.error(f"❌ Invalid user information in context")
            raise PermissionError("Invalid user information in context")

        # Check authorization
        user_id = user["id"]
        user_email = user["email"]

        if "*" in self.authorized_users:
            logger.info(f"✅ Wildcard access granted for user: {user_id}")
            return

        if user_id in self.authorized_users or user_email in self.authorized_users:
            logger.info(f"✅ User authorized: {user_id} ({user_email})")
            return

        logger.error(f"❌ User not authorized: {user_id} ({user_email})")
        raise PermissionError(
            f"User '{user_id}' ({user_email}) is not authorized to access application '{self.application_id}'"
        )

    async def _mimic_request(self, request_id: str) -> None:
        """
        Send HTTP request to Ray Serve to trigger autoscaling.

        When users access the application through WebRTC connections, Ray Serve
        doesn't automatically detect the load for autoscaling. This method sends
        an HTTP request to the Ray Serve endpoint to mimic the load, ensuring
        proper autoscaling behavior.

        The request uses the same ID as the actual RPC call for coordination
        with the __call__ method.

        Args:
            request_id: Unique identifier for correlating with the RPC call
        """
        logger.info(f"📡 Sending autoscaling trigger for request: {request_id}")

        try:
            timeout = httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0)

            async with httpx.AsyncClient(timeout=timeout) as client:
                await client.post(
                    f"{self.serve_http_url}/{self.application_id}/",
                    headers={"X-Request-ID": request_id},
                    json={"mimic_request": True},
                )

            logger.info(
                f"✅ Autoscaling trigger sent successfully for request: {request_id}"
            )

        except Exception as e:
            # Log but don't fail the user request
            logger.error(
                f"❌ Failed to send autoscaling trigger for '{self.application_id}' request {request_id}: {e}"
            )
            # Clean up orphaned event since HTTP request failed
            self._request_events.pop(request_id, None)

    async def _cleanup_orphaned_events(self) -> None:
        """
        Periodically clean up orphaned events that were never completed.

        This background task runs every 5 minutes and removes events older than 2 hours
        to prevent unbounded memory growth from failed or abandoned requests.
        """
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes

                current_time = time.time()
                max_age = 7200  # 2 hours

                orphaned_ids = []
                for request_id, event_data in self._request_events.items():
                    age = current_time - event_data["created_at"]
                    if age > max_age:
                        orphaned_ids.append(request_id)

                if orphaned_ids:
                    logger.info(
                        f"🧹 Cleaning up {len(orphaned_ids)} orphaned events older than 2 hours"
                    )
                    for request_id in orphaned_ids:
                        self._request_events.pop(request_id, None)

            except asyncio.CancelledError:
                logger.info(f"🛑 Cleanup task cancelled")
                break
            except Exception as e:
                logger.error(f"❌ Error in cleanup task: {e}")

    def _create_deployment_function(
        self, method_schema: Dict[str, Any]
    ) -> Callable[..., Any]:
        """
        Create a proxy function that forwards RPC requests to the Ray deployment.

        This method generates proxy functions that enable BioEngine applications
        to be accessed through Hypha's RPC system while maintaining proper
        authentication, logging, and error handling.

        The generated proxy function:
        - Validates user permissions before processing requests
        - Logs method calls for auditing and debugging
        - Forwards requests to the actual Ray Serve deployment
        - Handles Ray-specific errors appropriately
        - Uses @schema_function decorator for Hypha RPC integration
        - Maintains method schema compliance for validation

        Args:
            method_schema: Schema definition containing:
                - name: Method name
                - description: Method description
                - parameters: Parameter schema for validation

        Returns:
            Callable proxy function decorated with @schema_function
        """
        method_name = method_schema["name"]

        async def deployment_function(*args, context: Dict[str, Any], **kwargs) -> Any:
            async with self.service_semaphore:
                request_id = str(uuid.uuid4())
                event_created = False
                try:
                    # Check user permissions
                    await self._check_permissions(context)

                    # Log the method call
                    user_info = context.get("user", {}) if context else {}
                    user_id = user_info.get("id", "unknown")
                    logger.info(
                        f"🎯 User '{user_id}' calling method '{method_name}' on app '{self.application_id}'"
                    )

                    # Get the method from the entry deployment handle
                    method = getattr(self.entry_deployment_handle, method_name, None)
                    if method is None:
                        logger.error(
                            f"❌ Method '{method_name}' not found on entry deployment"
                        )
                        raise AttributeError(
                            f"Method '{method_name}' not found on entry deployment"
                        )

                    # Create event with timestamp for tracking BEFORE starting mimic request
                    self._request_events[request_id] = {
                        "event": asyncio.Event(),
                        "created_at": time.time(),
                    }
                    event_created = True

                    # Create the mimic request task but don't await it to avoid blocking
                    mimic_task = asyncio.create_task(self._mimic_request(request_id))

                    # Add error handling for the mimic task (optional - runs in background)
                    def handle_mimic_error(task):
                        if task.exception():
                            logger.error(
                                f"❌ Mimic request task failed for '{self.application_id}' request ID {request_id}: {task.exception()}"
                            )
                            # Clean up orphaned event if mimic request failed
                            self._request_events.pop(request_id, None)

                    mimic_task.add_done_callback(handle_mimic_error)

                    # Forward the request to the actual deployment
                    try:
                        result = await method.remote(*args, **kwargs)
                        logger.info(
                            f"✅ Successfully executed method '{method_name}' for user {user_id}"
                        )
                        return result
                    except RayTaskError as e:
                        logger.error(
                            f"❌ Ray task error in method '{method_name}': {e}"
                        )
                        raise
                    except Exception as e:
                        logger.error(
                            f"❌ Unexpected error in method '{method_name}': {e}"
                        )
                        raise

                except PermissionError as e:
                    logger.warning(
                        f"⚠️ Permission denied for method '{method_name}': {e}"
                    )
                    # Clean up event if created before permission error
                    if event_created:
                        self._request_events.pop(request_id, None)
                    raise
                except Exception as e:
                    logger.error(f"❌ Error in proxy function '{method_name}': {e}")
                    # Clean up event if created before error
                    if event_created:
                        self._request_events.pop(request_id, None)
                    raise
                finally:
                    # Signal that the request is complete, but don't remove the event yet
                    # The event will be removed when the HTTP request completes in __call__
                    event_data = self._request_events.get(request_id)
                    if event_data:
                        event_data["event"].set()

        return schema_function(
            func=deployment_function,
            arbitrary_types_allowed=True,  # to support type Callable
            name=method_schema["name"],
            description=method_schema["description"],
            parameters=method_schema["parameters"],
        )

    async def _on_webrtc_init(self, peer_connection: "RTCPeerConnection") -> None:
        """
        Initialize and monitor a new WebRTC peer connection.

        This callback is invoked by hypha-rpc when a client establishes a WebRTC
        connection. It sets up connection monitoring, tracks active connections,
        and manages the connection lifecycle.

        Connection State Monitoring:
        The method registers handlers to track WebRTC connection state changes:
        - "connected": Peer-to-peer connection successfully established
        - "failed": Connection failed due to network, NAT, or other issues
        - "closed": Connection terminated by either peer
        - "disconnected": Connection interrupted (may be temporary)

        Connection Tracking:
        Active connections are tracked with:
        - Unique connection identifiers for debugging
        - State-based lifecycle management
        - Automatic cleanup when connections close or fail
        - Memory leak prevention through proper reference management

        Note: WebRTC connections are automatically cleaned up when the Ray Serve
        process terminates, so no explicit cleanup handlers are needed.

        Args:
            peer_connection: The WebRTC peer connection object from aiortc
        """
        try:
            # Generate unique connection ID
            connection_id = str(uuid.uuid4())
            current_time = time.time()

            logger.info(
                f"🔗 WebRTC peer connection initialized for '{self.application_id}' "
                f"(Connection ID: {connection_id[:8]}...)"
            )

            # Add to active connections tracking
            self._active_peer_connections[connection_id] = {
                "peer_connection": peer_connection,
                "created_at": current_time,  # TODO: call peer_connection.close() after a timeout
                "state": "new",
            }

            # Set up connection state monitoring
            @peer_connection.on("connectionstatechange")
            def on_connection_state_change():
                state = peer_connection.connectionState
                logger.info(
                    f"🔄 WebRTC connection state changed to '{state}' "
                    f"for '{self.application_id}' (Connection ID: {connection_id[:8]}...)"
                )

                # Update connection state
                if connection_id in self._active_peer_connections:
                    self._active_peer_connections[connection_id]["state"] = state

                    # Remove connection if it's closed or failed
                    if state in ["closed", "failed"]:
                        logger.info(
                            f"🚫 Removing WebRTC connection '{connection_id[:8]}...' "
                            f"for '{self.application_id}' (State: {state})"
                        )
                        del self._active_peer_connections[connection_id]

                        logger.info(
                            f"📊 Active WebRTC connections: {len(self._active_peer_connections)}"
                        )

            logger.info(
                f"📊 Active WebRTC connections: {len(self._active_peer_connections)}"
            )

        except Exception as e:
            logger.error(
                f"❌ Failed to initialize WebRTC connection for '{self.application_id}': {e}"
            )

    async def _fetch_ice_servers(self) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch custom ICE servers for WebRTC connections.

        ICE (Interactive Connectivity Establishment) servers help establish
        WebRTC connections through NAT and firewall traversal. This method
        attempts to fetch custom ICE servers from the Hypha infrastructure.

        ICE servers provide:
        1. STUN servers: Discover public IP addresses
        2. TURN servers: Relay traffic when direct connection fails
        3. Support for various network configurations

        The method gracefully falls back to None if custom servers are unavailable,
        allowing hypha-rpc to use its built-in default servers.

        Returns:
            List of ICE server configurations, or None to use defaults.

        Example ICE server format:
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
                    f"✅ Successfully fetched ICE servers for {self.application_id}"
                )
                return ice_servers
        except HTTPStatusError as e:
            logger.error(
                f"❌ HTTP error fetching ICE servers for {self.application_id}: {e}"
            )
        except RequestError as e:
            logger.error(
                f"❌ Request error fetching ICE servers for {self.application_id}: {e}"
            )
        except Exception as e:
            logger.error(
                f"❌ Unexpected error fetching ICE servers for {self.application_id}: {e}"
            )

        logger.warning(
            f"⚠️ Falling back to default ICE servers for {self.application_id}"
        )
        return None

    @schema_method
    async def _get_service_load(
        self,
        context: Dict[str, Any] = Field(
            ...,
            description="Authentication context containing user information, automatically provided by Hypha during service calls.",
        ),
    ) -> float:
        """
        Returns the current load of the BioEngine application service as a float value between 0 and 1.
        This method is used by Hypha's load balancing system to distribute requests
        across multiple service instances and for monitoring service capacity.

        Load Calculation:
        - 0.0: No active requests, service is idle and ready to handle new requests
        - 1.0: Maximum capacity reached, all semaphore slots occupied
        - Values between 0 and 1 indicate partial load based on active request ratio

        The load is calculated based on the number of active requests being processed
        through the service semaphore, which limits concurrent request processing to prevent overload.

        Returns:
            float: Current service load between 0.0 (idle) and 1.0 (at capacity)
        """
        # Calculate load based on available vs. total semaphore capacity
        available_slots = self.service_semaphore._value
        total_slots = self.max_ongoing_requests
        active_requests = total_slots - available_slots
        load = active_requests / total_slots

        return min(1.0, max(0.0, load))  # Ensure load is between 0 and 1

    @schema_method
    async def _get_num_pcs(
        self,
        context: Dict[str, Any] = Field(
            ...,
            description="Authentication context containing user information, automatically provided by Hypha during service calls.",
        ),
    ) -> int:
        """
        Returns the current number of active WebRTC peer connections for this BioEngine application.
        This method is used for monitoring connection status, debugging WebRTC connectivity issues,
        and understanding real-time usage patterns of the application.

        WebRTC peer connections enable direct peer-to-peer communication between clients and the
        BioEngine application, bypassing traditional server-mediated communication for better
        performance with large data transfers.

        Connection States Tracked:
        - Only counts connections in "connected" state
        - Excludes failed, closed, or disconnected connections
        - Updates automatically as connections are established or terminated

        Returns:
            int: Number of currently active WebRTC peer connections (0 or positive integer)
        """
        num_connections = len(self._active_peer_connections)

        return num_connections

    @schema_method
    async def _get_rtc_service_id(
        self,
        context: Dict[str, Any] = Field(
            ...,
            description="Authentication context containing user information, automatically provided by Hypha during service calls.",
        ),
    ) -> Optional[str]:
        """
        Returns the registered WebRTC service identifier if the WebRTC service is successfully registered.
        This method is used to verify WebRTC service availability and obtain the service ID for
        direct peer-to-peer connections to the BioEngine application.

        WebRTC Service Registration:
        - WebRTC services are registered separately from the main WebSocket service
        - Service ID follows the pattern: "{application_id}-rtc"
        - Registration may fail due to network issues or server constraints
        - Main application remains functional even if WebRTC registration fails

        Use Cases:
        - Verify WebRTC capability before attempting peer-to-peer connections
        - Troubleshoot WebRTC service registration issues
        - Provide service discovery information to clients

        Returns:
            Optional[str]: WebRTC service ID string if registered, None if registration failed or pending
        """
        return self.rtc_service_id

    async def _register_services(self) -> None:
        """
        Register WebSocket, WebRTC, and MCP services with the Hypha server.

        This method performs the complete service registration process:
        1. Connects to the Hypha server using the provided credentials
        2. Fetches custom ICE servers for WebRTC (optional)
        3. Registers WebRTC service for peer-to-peer connections (optional)
        4. Creates proxy functions from method schemas
        5. Registers the main WebSocket service (required)
        6. Registers MCP (Model Context Protocol) service

        WebRTC registration is optional - if it fails, the deployment remains
        functional with WebSocket-only communication. The main WebSocket service
        registration is required for the deployment to be considered healthy.
        """
        # Connect to Hypha server
        try:
            # Connect to the Hypha server
            self.server = await connect_to_server(
                {
                    "server_url": self.server_url,
                    "token": self.proxy_service_token,
                    "workspace": self.workspace,
                    "client_id": self.client_id,
                }
            )
            connected_workspace = self.server["config"]["workspace"]
            if connected_workspace != self.workspace:
                raise RuntimeError(
                    f"Workspace mismatch: expected '{self.workspace}', got '{connected_workspace}'"
                )
            registered_client_id = self.server["config"]["client_id"]
            if registered_client_id != self.client_id:
                raise RuntimeError(
                    f"Client ID mismatch: expected '{self.client_id}', got '{registered_client_id}'"
                )
            logger.info(
                f"✅ Successfully connected to Hypha server as client "
                f"'{self.client_id}' for '{self.application_id}'"
            )
        except Exception as e:
            self.server = None
            logger.error(
                f"❌ Error connecting to Hypha server for '{self.application_id}': {e}"
            )
            raise

        try:
            # Create service functions from method schemas
            service_functions = {}
            for method_schema in self.method_schemas:
                method_name = method_schema["name"]
                service_functions[method_name] = self._create_deployment_function(
                    method_schema
                )

            # Add load check function - for service load balancing (https://docs.amun.ai/#/service-load-balancing)
            service_functions["get_load"] = self._get_service_load

            # Add peer connection count function - for WebRTC connection monitoring
            service_functions["get_num_pcs"] = self._get_num_pcs

            # Add RTC service ID function - for WebRTC service ID retrieval
            service_functions["get_rtc_service_id"] = self._get_rtc_service_id

            # Register the main service
            websocket_service_info = await self.server.register_service(
                {
                    "id": self.application_id,
                    "name": self.application_name,
                    "description": self.application_description,
                    "type": "bioengine-apps",
                    "config": {"visibility": "public", "require_context": True},
                    **service_functions,
                }
            )
            self.websocket_service_id = (
                f"{self.workspace}/{self.client_id}:{self.application_id}"
            )
            if websocket_service_info["id"] != self.websocket_service_id:
                raise RuntimeError(
                    f"Service ID mismatch: expected '{self.websocket_service_id}', got '{websocket_service_info['id']}'"
                )
            logger.info(
                f"✅ Successfully registered WebSocket service for '{self.application_id}' "
                f"with ID: {self.websocket_service_id}"
            )

        except Exception as e:
            self.service_id = None
            logger.error(
                f"❌ Error registering WebSocket service for '{self.application_id}': {e}"
            )
            raise

        # Register WebRTC service with custom ICE servers or fallback to defaults
        try:
            # Prepare WebRTC config
            rtc_config = {
                "visibility": "public",
                "require_context": True,
                "on_init": self._on_webrtc_init,  # Add WebRTC connection handler
            }

            # Fetch custom ICE servers
            ice_servers = await self._fetch_ice_servers()

            # Add custom ICE servers if available, otherwise hypha-rpc will use defaults
            if ice_servers:
                rtc_config["ice_servers"] = ice_servers

            # Register WebRTC service
            rtc_service_info = await register_rtc_service(
                self.server,
                service_id=f"{self.application_id}-rtc",
                config=rtc_config,
            )
            self.rtc_service_id = (
                f"{self.workspace}/{self.client_id}:{self.application_id}-rtc"
            )
            if rtc_service_info["id"] != self.rtc_service_id:
                raise RuntimeError(
                    f"RTC Service ID mismatch: expected '{self.rtc_service_id}', got '{rtc_service_info['id']}'"
                )
            logger.info(
                f"✅ Registered WebRTC service for '{self.application_id}' with ID: {self.rtc_service_id}"
            )

        except Exception as e:
            logger.warning(
                f"⚠️ Warning: Failed to register WebRTC service for '{self.application_id}': {e}"
            )
            # Don't fail the entire deployment if WebRTC registration fails

        # Log registered service functions
        logger.info(
            f"📋 Service functions registered: {list(service_functions.keys())}"
        )

    # ===== Ray Serve Health Check =====
    # Implements periodic health checks for Ray Serve.

    async def check_health(self):
        """
        Perform Ray Serve health check for this deployment.

        This method is called by Ray Serve during deployment initialization and
        periodically thereafter to ensure the deployment is healthy and ready
        to serve requests.

        Health Check Process:
        1. Verifies the underlying BioEngine application deployment is healthy
        2. Ensures Hypha server connection is established
        3. Registers services with Hypha if not already done
        4. Tests Hypha server connectivity with a ping

        Raises:
            RuntimeError: If any health check fails, indicating the deployment is unhealthy
        """
        # Wait for the entry deployment to be ready and healthy
        if not self.entry_deployment_ready:
            await self.entry_deployment_handle.check_health.remote()
            self.entry_deployment_ready = True

        # Register services if not already done (with lock to prevent concurrent registration)
        if not self.server or not self.websocket_service_id:
            async with self._registration_lock:
                # Double-check after acquiring lock
                if not self.server or not self.websocket_service_id:
                    await self._register_services()

                    # Start cleanup task on first successful registration
                    if self._cleanup_task is None or self._cleanup_task.done():
                        self._cleanup_task = asyncio.create_task(
                            self._cleanup_orphaned_events()
                        )

        # Check if Hypha server can be reached
        try:
            await self.server.echo("ping")
        except Exception as e:
            logger.error(
                f"❌ Hypha server connection failed for '{self.application_id}': {e}"
            )
            # Reset server connection to trigger re-connection on next call
            self.server = None
            # Clear orphaned events since the connection is lost
            self._request_events.clear()
            raise RuntimeError("Hypha server connection failed")

        # Check if the WebSocket service can be reached
        try:
            websocket_service = await self.server.get_service(self.websocket_service_id)
            await websocket_service.get_load()
            await websocket_service.get_num_pcs()
        except Exception as e:
            logger.error(
                f"❌ WebSocket service connection failed for '{self.application_id}': {e}"
            )
            # Reset service ID to trigger re-registration on next call
            self.websocket_service_id = None
            # Clear orphaned events since the service is lost
            self._request_events.clear()
            raise RuntimeError("WebSocket service connection failed")

        # All checks passed - deployment is healthy


if __name__ == "__main__":

    class MockMethod:
        def __init__(self, name: str):
            self.name = name

        async def remote(self, *args, **kwargs):
            print(
                f"Mocked method '{self.name}' called with args={args}, kwargs={kwargs}"
            )
            return {"status": "success", "data": "mocked data"}

    class MockHandle:
        def __getattr__(self, name):
            return MockMethod(name)

    # Example usage of BioEngineProxyDeployment
    async def test_proxy_deployment():
        rtc_deployment_class = BioEngineProxyDeployment.func_or_class
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

        server_url = "https://hypha.aicell.io"
        token = os.environ["HYPHA_TOKEN"]
        worker_client = await connect_to_server(
            {
                "server_url": server_url,
                "token": token,
            }
        )
        workspace = worker_client["config"]["workspace"]
        worker_client_id = worker_client["config"]["client_id"]

        deployment = rtc_deployment_class(
            application_id="test-app",
            application_name="Test Application",
            application_description="A test application for demonstration",
            entry_deployment_handle=entry_deployment_handle,
            method_schemas=[method_schema],
            server_url=server_url,
            workspace=workspace,
            proxy_service_token=token,
            worker_client_id=worker_client_id,
            authorized_users=["*"],
            serve_http_url="not_used_in_mock",
        )

        await deployment.check_health()

        print("Deployment is healthy and services are registered")

    asyncio.run(test_proxy_deployment())
