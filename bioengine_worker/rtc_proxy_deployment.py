import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional

import httpx
from hypha_rpc import connect_to_server, register_rtc_service
from hypha_rpc.utils.schema import schema_function
from ray import serve
from ray.exceptions import RayTaskError
from ray.serve.handle import DeploymentHandle

# Configure logging
logger = logging.getLogger("RtcProxyDeployment")


@serve.deployment(
    ray_actor_options={"num_cpus": 1},
    health_check_period_s=10,  # Check health every 10 seconds
    health_check_timeout_s=5,  # Timeout after 5 seconds
)
class RtcProxyDeployment:
    def __init__(
        self,
        application_id: str,
        application_name: str,
        application_description: str,
        entry_deployment: DeploymentHandle,
        method_schemas: List[Dict[str, Any]],
        server_url: str,
        workspace: str,
        token: str,
        authorized_users: List[str],
    ):
        # BioEngine application metadata
        self.application_id = application_id
        self.application_name = application_name
        self.application_description = application_description
        self.entry_deployment = entry_deployment
        self.method_schemas = method_schemas

        # Hypha server connection parameters
        self.server_url = server_url
        self.workspace = workspace
        self.token = token
        self.authorized_users = authorized_users

        # Service state
        self.service_id: Optional[str] = None
        self.server = None
        self.rtc_service_id: Optional[str] = None  # Track WebRTC service separately
        self._shutdown_event = asyncio.Event()
        self._service_ready = asyncio.Event()
        self._last_error: Optional[str] = None
        self._connection_handlers: List[Callable] = (
            []
        )  # Store connection event handlers

        # Start the service registration task
        self.service_task = asyncio.create_task(
            self._register_web_rtc_service(),
            name=f"rtc_proxy_service_{application_id}",
        )

    # ===== Hypha Service Registration =====
    # Methods responsible for registering and managing Hypha RPC services
    # These methods handle both WebSocket and WebRTC service registration with the Hypha server

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

        Error Handling:
            - HTTP errors (4xx, 5xx responses)
            - Network connectivity issues
            - Request timeouts (30 second timeout)
            - JSON parsing errors

        Returns:
            Optional[List[Dict[str, Any]]]: List of ICE server configurations
                                          or None if fetching fails

        ICE Server Format:
            [{"urls": "stun:stun.server.com:19302"},
             {"urls": "turn:turn.server.com:3478", "username": "...", "credential": "..."}]

        Used By:
            - _register_web_rtc_service() during WebRTC service registration
        """
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(
                    "https://ai.imjoy.io/public/services/coturn/get_rtc_ice_servers"
                )
                response.raise_for_status()
                ice_servers = response.json()
                logger.info(
                    f"Successfully fetched ICE servers for {self.application_id}"
                )
                return ice_servers
        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error fetching ICE servers for {self.application_id}: {e}"
            )
        except httpx.RequestError as e:
            logger.error(
                f"Request error fetching ICE servers for {self.application_id}: {e}"
            )
        except Exception as e:
            logger.error(
                f"Unexpected error fetching ICE servers for {self.application_id}: {e}"
            )

        return None

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

        Args:
            context: Request context containing user information from Hypha

        Raises:
            PermissionError: If user is not authorized or context is invalid

        Context Format Expected:
            {"user": {"id": "user123", "email": "user@example.com"}}

        Used By:
            - Proxy functions created by _create_deployment_function()
            - All registered service methods for access control
        """
        if context is None or "user" not in context:
            raise PermissionError("Context is missing user information")

        user = context["user"]
        if not isinstance(user, dict):
            raise PermissionError("Invalid user information in context")

        user_id = user.get("id")
        user_email = user.get("email")

        if not user_id and not user_email:
            raise PermissionError("User context missing both ID and email")

        # Check authorization
        if "*" in self.authorized_users:
            return  # Wildcard access

        if user_id and user_id in self.authorized_users:
            return

        if user_email and user_email in self.authorized_users:
            return

        raise PermissionError(
            f"User {user_id or user_email} is not authorized to access application '{self.application_id}'"
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

        Returns:
            Callable: Async proxy function ready for Hypha service registration

        Proxy Function Signature:
            async def deployment_function(*args, context, **kwargs) -> Any

        Used By:
            - _register_web_rtc_service() during service function creation
        """
        method_name = method_schema["name"]

        @schema_function(
            name=method_schema["name"],
            description=method_schema["description"],
            parameters=method_schema["parameters"],
        )
        async def deployment_function(*args, context, **kwargs) -> Any:
            try:
                # Check user permissions
                await self._check_permissions(context)

                # Log the method call
                user_info = context.get("user", {}) if context else {}
                user_id = user_info.get("id", "unknown")
                logger.info(
                    f"User {user_id} calling method '{method_name}' on app '{self.application_id}'"
                )

                # Get the method from the entry deployment handle
                if not hasattr(self.entry_deployment, method_name):
                    raise AttributeError(
                        f"Method '{method_name}' not found on entry deployment"
                    )

                method = getattr(self.entry_deployment, method_name)

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

        return deployment_function

    async def _register_web_rtc_service(self) -> None:
        """
        **HYPHA SERVICE REGISTRATION - MAIN REGISTRATION PROCESS**

        Register both WebSocket and WebRTC services with the Hypha server,
        establishing the complete RPC service infrastructure for this BioEngine
        application deployment.

        Registration Process:
        1. Connect to Hypha server using provided credentials
        2. Fetch custom ICE servers for WebRTC (with fallback to defaults)
        3. Register WebRTC service for peer-to-peer connections
        4. Create proxy functions for all application methods
        5. Register main WebSocket service with all proxy functions
        6. Signal service readiness for health checks
        7. Start connection monitoring loop

        Service Types Registered:
        - WebRTC Service: For direct peer-to-peer connections (optional)
        - WebSocket Service: For standard RPC connections (required)

        The method handles failures gracefully:
        - WebRTC registration failure doesn't prevent WebSocket service
        - Connection monitoring continues even if some services fail
        - Error states are recorded for health check system

        Service Configuration:
        - Visibility: "public" (accessible to all authorized users)
        - Context: Required (for user authentication)
        - Type: "bioengine-apps" (for service discovery)

        Raises:
            Exception: If critical service registration fails

        Side Effects:
        - Sets self.server (Hypha connection)
        - Sets self.service_id (main WebSocket service ID)
        - Sets self.rtc_service_id (WebRTC service ID, may be None)
        - Sets self._service_ready event
        - May set self._last_error on failure

        Used By:
        - Constructor via asyncio.create_task() during deployment initialization
        """
        try:
            logger.info(
                f"Connecting to Hypha server for application '{self.application_id}'"
            )

            # Connect to the Hypha server
            self.server = await connect_to_server(
                {
                    "server_url": self.server_url,
                    "workspace": self.workspace,
                    "token": self.token,
                }
            )
            logger.info(
                f"Successfully connected to Hypha server for '{self.application_id}'"
            )

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

            # Create service functions from method schemas
            service_functions = {}
            for method_schema in self.method_schemas:
                try:
                    method_name = method_schema["name"]
                    service_functions[method_name] = self._create_deployment_function(
                        method_schema
                    )
                    logger.debug(f"Created proxy function for method '{method_name}'")
                except Exception as e:
                    logger.error(
                        f"Failed to create proxy function for method '{method_schema.get('name', 'unknown')}': {e}"
                    )
                    continue

            if not service_functions:
                raise RuntimeError("No valid service functions could be created")

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

            # Signal that the service is ready
            self._service_ready.set()
            logger.info(f"Service '{self.application_id}' is now healthy and ready")

            # Keep the service running and monitor connection health
            await self._run_with_connection_monitoring()

        except Exception as e:
            logger.error(
                f"Error registering WebRTC service for '{self.application_id}': {e}"
            )
            # Set error state and signal ready (to unblock waiters)
            self._last_error = f"Service registration failed: {str(e)}"
            self._service_ready.set()  # Unblock waiters even on failure
            raise

    async def _on_webrtc_init(self, peer_connection) -> None:
        """
        **HYPHA SERVICE REGISTRATION - WEBRTC CONNECTION HANDLER**

        Handler called when a new WebRTC peer connection is established through
        the registered WebRTC service. Sets up connection monitoring and event
        handlers to integrate WebRTC connection state with the health check system.

        This method is registered as the "on_init" callback during WebRTC service
        registration and is automatically called by hypha-rpc when a client
        establishes a WebRTC peer connection to this service.

        Connection State Monitoring:
        - "connected": Clears any WebRTC-related error states
        - "failed": Sets error state that will be caught by health checks
        - "closed": Logs closure but doesn't set error state

        Integration with Health System:
        - Failed connections set self._last_error for health check detection
        - Successful connections clear WebRTC-related errors via _clear_error_state()
        - Connection handlers are stored for proper cleanup during shutdown

        Args:
            peer_connection: RTCPeerConnection object from aiortc library

        Side Effects:
        - Registers connection state change handler on peer_connection
        - May set self._last_error on connection failures
        - May call self._clear_error_state() on successful connections
        - Stores handler reference in self._connection_handlers for cleanup

        Called By:
        - hypha-rpc WebRTC service infrastructure during peer connection setup
        - Registered as "on_init" callback in WebRTC service configuration

        Used For:
        - Real-time monitoring of WebRTC connection health
        - Integration with Ray Serve health check system
        - Automatic error recovery when connections are restored
        """
        try:
            logger.info(
                f"WebRTC peer connection initialized for '{self.application_id}'"
            )

            # Set up connection state monitoring
            def on_connection_state_change():
                state = peer_connection.connectionState
                logger.info(
                    f"WebRTC connection state changed to '{state}' for '{self.application_id}'"
                )

                if state == "failed":
                    logger.error(
                        f"WebRTC connection failed for '{self.application_id}'"
                    )
                    self._last_error = "WebRTC connection failed"
                elif state == "closed":
                    logger.info(f"WebRTC connection closed for '{self.application_id}'")
                elif state == "connected":
                    logger.info(
                        f"WebRTC connection established for '{self.application_id}'"
                    )
                    # Clear any connection-related errors
                    if self._last_error and "webrtc" in self._last_error.lower():
                        self._clear_error_state()

            # Register the connection state change handler
            peer_connection.on_connectionstatechange = on_connection_state_change

            # Store handler reference for cleanup
            self._connection_handlers.append(on_connection_state_change)

        except Exception as e:
            logger.error(
                f"Error setting up WebRTC connection monitoring for '{self.application_id}': {e}"
            )
            self._last_error = f"WebRTC initialization failed: {str(e)}"

    async def _run_with_connection_monitoring(self) -> None:
        """
        Keep the service running while periodically monitoring the connection.
        """
        last_connection_check = 0
        connection_check_interval = 120  # Check connection every 2 minutes (less frequent due to WebRTC monitoring)

        while not self._shutdown_event.is_set():
            try:
                # Wait for shutdown with timeout for periodic checks
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(), timeout=connection_check_interval
                    )
                    break  # Shutdown was signaled
                except asyncio.TimeoutError:
                    pass  # Timeout reached, do periodic check

                # Periodic connection health check
                current_time = asyncio.get_event_loop().time()
                if current_time - last_connection_check >= connection_check_interval:
                    last_connection_check = current_time

                    # Test connection health internally
                    connection_ok = await self._test_hypha_connection()
                    if not connection_ok:
                        logger.error(
                            f"Connection health check failed for '{self.application_id}'"
                        )
                    else:
                        # Clear any connection-related errors if connection is working
                        if self._last_error and (
                            "connection" in self._last_error.lower()
                            or "webrtc" in self._last_error.lower()
                        ):
                            logger.info(
                                f"Connection restored for '{self.application_id}'"
                            )
                            self._clear_error_state()

            except Exception as e:
                logger.error(
                    f"Error in connection monitoring for '{self.application_id}': {e}"
                )
                self._last_error = f"Connection monitoring error: {str(e)}"

    # ===== BioEngine Worker Interface =====
    # Methods exposed to the BioEngine worker for deployment management
    # These methods provide the external API for interacting with this deployment

    async def async_init(self) -> None:
        """
        **BIOENGINE WORKER INTERFACE - DEPLOYMENT INITIALIZATION**

        Initialize the entry deployment if it supports async initialization.
        This method is called by the BioEngine worker during application deployment
        to perform any required async initialization on the underlying Ray deployment.

        The method attempts to call the `_async_init` method on the entry deployment,
        but gracefully handles cases where the method doesn't exist, as async
        initialization is optional for BioEngine applications.

        Called By:
            - AppsManager during application deployment
            - BioEngine worker during service startup

        Behavior:
            - Calls entry_deployment._async_init.remote() if available
            - Logs success or gracefully handles missing method
            - Propagates actual errors (not "method not found" errors)

        Raises:
            Exception: If async initialization fails (not if method is missing)

        Returns:
            None: Completion indicates successful initialization or graceful skip
        """
        try:
            await self.entry_deployment._async_init.remote()
            logger.info(
                f"Successfully initialized entry deployment for '{self.application_id}'"
            )
        except RayTaskError as e:
            if "Tried to call a method '_async_init' that does not exist." not in str(
                e
            ):
                logger.error(
                    f"Error during async init for '{self.application_id}': {e}"
                )
                raise e
            # Method doesn't exist, which is fine
            logger.debug(
                f"Entry deployment for '{self.application_id}' does not support async_init"
            )
        except Exception as e:
            logger.error(
                f"Unexpected error during async init for '{self.application_id}': {e}"
            )
            raise

    async def test_application(self) -> Any:
        """
        **BIOENGINE WORKER INTERFACE - APPLICATION TESTING**

        Test the entry deployment if it supports testing functionality.
        This method is called by the BioEngine worker to verify that the
        deployed application is working correctly and can handle requests.

        The method attempts to call the `_test_application` method on the entry
        deployment, but gracefully handles cases where testing is not implemented,
        as application testing is optional for BioEngine applications.

        Called By:
            - AppsManager during application health verification
            - BioEngine worker during deployment validation
            - Manual testing workflows

        Behavior:
            - Calls entry_deployment._test_application.remote() if available
            - Returns test results or None if testing not supported
            - Propagates actual test failures (not "method not found" errors)

        Returns:
            Any: Test results from the application, or None if testing not supported

        Raises:
            Exception: If application testing fails (not if method is missing)
        """
        try:
            result = await self.entry_deployment._test_application.remote()
            logger.info(f"Successfully tested application '{self.application_id}'")
            return result
        except RayTaskError as e:
            if (
                "Tried to call a method '_test_application' that does not exist."
                not in str(e)
            ):
                logger.error(
                    f"Error during application test for '{self.application_id}': {e}"
                )
                raise e
            # Method doesn't exist, which is fine
            logger.debug(
                f"Entry deployment for '{self.application_id}' does not support test_application"
            )
            return None
        except Exception as e:
            logger.error(
                f"Unexpected error during application test for '{self.application_id}': {e}"
            )
            raise

    async def get_service_ids(self, timeout: int = 60) -> Dict[str, str]:
        """
        **BIOENGINE WORKER INTERFACE - SERVICE ID RETRIEVAL**

        Get both WebSocket and WebRTC service IDs, waiting for registration
        to complete if necessary. This method provides the BioEngine worker
        with the registered service identifiers needed for client connections.

        The method waits for service registration to complete (via _service_ready
        event) and returns both the main WebSocket service ID and the optional
        WebRTC service ID for peer-to-peer connections.

        Called By:
            - AppsManager to register services with BioEngine catalog
            - BioEngine worker to provide connection info to clients
            - Service discovery and monitoring systems

        Service Types Returned:
            - websocket_service_id: Main Hypha service for WebSocket connections
            - webrtc_service_id: Optional WebRTC service for P2P connections (may be None)

        Args:
            timeout: Maximum time in seconds to wait for service registration

        Returns:
            Dict[str, str]: Dictionary containing:
                - "websocket_service_id": Main service ID (always present)
                - "webrtc_service_id": WebRTC service ID (may be None)

        Raises:
            TimeoutError: If service registration doesn't complete within timeout
            RuntimeError: If service registration failed completely
        """
        try:
            # Wait for service to be ready
            await asyncio.wait_for(self._service_ready.wait(), timeout=timeout)

            if self.service_id is None:
                raise RuntimeError(
                    f"Service registration failed for '{self.application_id}'"
                )

            return {
                "websocket_service_id": self.service_id,
                "webrtc_service_id": self.rtc_service_id,
            }
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"Timed out waiting for service registration for '{self.application_id}'"
            )

    # ===== Ray Serve Health Check =====
    # Methods required for Ray Serve health monitoring and status checking
    #
    # Health Check State Variables (defined in __init__):
    # - self._last_error: Optional[str] = None
    #   Records any error state that should cause health check to fail.
    #   Set by connection monitoring, cleared by _clear_error_state().
    #
    # - self._service_ready: asyncio.Event()
    #   Event that signals when service registration is complete.
    #   Set by _register_web_rtc_service() when both Hypha connection and
    #   service registration succeed.
    #
    # - self.server: Optional[hypha_rpc.Server] = None
    #   Hypha server connection object. Must be non-None for health.
    #   Set during _register_web_rtc_service() connection phase.
    #
    # - self.service_id: Optional[str] = None
    #   Main WebSocket service ID from Hypha registration.
    #   Required for health (WebRTC service_id is optional).
    #
    # Ray Serve Configuration (in @serve.deployment decorator):
    # - health_check_period_s=10: Calls check_health() every 10 seconds
    # - health_check_timeout_s=5: Health check must complete within 5 seconds

    def check_health(self):
        """
        **REQUIRED FOR RAY SERVE HEALTH MONITORING**

        Primary health check method called periodically by Ray Serve to determine
        if this deployment replica is healthy and should continue serving requests.

        This method is automatically invoked by Ray Serve based on the deployment
        configuration parameters:
        - health_check_period_s=10: Health check runs every 10 seconds
        - health_check_timeout_s=5: Health check must complete within 5 seconds

        Health Check Criteria:
        1. No recorded internal errors (_last_error must be None)
        2. Hypha server connection must be established (self.server is not None)
        3. Service registration must be complete (_service_ready must be set)
        4. Main service ID must be available (WebSocket service)

        Note: WebRTC service registration is optional and does not affect health status.
        The deployment can serve requests via WebSocket even if WebRTC fails.

        Raises:
            RuntimeError: If any health check criterion fails, causing Ray Serve
                         to mark this replica as unhealthy and potentially restart it

        Returns:
            None: Implicit success if no exception is raised
        """
        # Check if we have a last recorded error
        if self._last_error:
            raise RuntimeError(f"Deployment unhealthy: {self._last_error}")

        # Check if Hypha server connection is established
        if self.server is None:
            raise RuntimeError("Hypha server connection not established")

        # Check if service is registered and ready
        if not self._service_ready.is_set():
            raise RuntimeError("Service not registered or ready")

        if self.service_id is None:
            raise RuntimeError("Service ID not available")

        # All checks passed - deployment is healthy
        # Note: WebRTC service registration is optional and doesn't affect health

    def _clear_error_state(self) -> None:
        """
        **USED BY RAY SERVE HEALTH RECOVERY**

        Clears any recorded error state, allowing the deployment to become healthy
        again after connection restoration or error resolution.

        This method is called internally when:
        - Hypha connection is restored after a temporary failure
        - WebRTC connection issues are resolved
        - Other transient errors are cleared during connection monitoring

        The method enables automatic recovery without requiring a full deployment
        restart, improving service availability and reducing downtime.

        Usage Pattern:
            # Connection monitoring detects restoration
            if connection_restored and self._last_error:
                self._clear_error_state()  # Allow health check to pass again

        Side Effects:
            - Sets self._last_error to None
            - Logs the error state that was cleared for debugging
        """
        if self._last_error:
            logger.info(
                f"Cleared error state for '{self.application_id}': {self._last_error}"
            )
            self._last_error = None

    async def _test_hypha_connection(self) -> bool:
        """
        **USED BY RAY SERVE HEALTH MONITORING**

        Performs active testing of the Hypha server connection to detect network
        issues, server downtime, or service registration problems before they
        affect the main health check.

        This method is called periodically by the connection monitoring loop
        (_run_with_connection_monitoring) to proactively detect issues and set
        error states that will be caught by the main check_health() method.

        Connection Test Process:
        1. Verifies Hypha server connection exists
        2. Tests connection with lightweight server.list_services() call
        3. Optionally verifies service registration is still valid
        4. Sets _last_error if any issues are detected

        This proactive approach allows Ray Serve health checks to detect connection
        issues immediately rather than waiting for actual request failures.

        Returns:
            bool: True if connection is healthy, False if issues detected
                 When False is returned, _last_error is set with details

        Side Effects:
            - May set self._last_error on connection failure
            - Logs warnings for connection issues
            - Does not raise exceptions (returns status instead)

        Used By:
            - _run_with_connection_monitoring() for periodic health testing
            - Connection restoration logic for validating recovery
        """
        if self.server is None:
            return False

        try:
            # Test the connection by trying to list services
            # This is a lightweight operation that validates the connection
            await self.server.list_services()

            # If we have a registered service, verify it's still accessible
            if self.service_id:
                try:
                    service_info = await self.server.get_service_info(self.service_id)
                    if not service_info:
                        logger.warning(
                            f"Service '{self.service_id}' not found on server"
                        )
                        self._last_error = "Service not found on server"
                        return False
                except Exception as e:
                    logger.warning(f"Failed to verify service '{self.service_id}': {e}")
                    # Don't mark as failed just for service info check

            return True
        except Exception as e:
            logger.warning(
                f"Hypha connection test failed for '{self.application_id}': {e}"
            )
            self._last_error = f"Connection test failed: {str(e)}"
            return False
