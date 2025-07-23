"""
BioEngine Worker End-to-End Testing Suite

Comprehensive end-to-end tests for BioEngine Worker startup deployments,
service registration, and application lifecycle management. These tests
validate the complete worker functionality in a real Hypha server environment.

Usage:
    conda activate bioengine-worker
    pytest tests/end_to_end/ -v
"""

import asyncio
from typing import Dict, List

import pytest
from hypha_rpc import get_rtc_service
from hypha_rpc.rpc import RemoteService

from bioengine_worker.worker import BioEngineWorker


@pytest.mark.asyncio
async def test_worker_initialization_and_readiness(
    bioengine_worker: BioEngineWorker, worker_ready_timeout: int
):
    """Test that the BioEngine worker initializes correctly and becomes ready."""

    # Wait for worker to be ready
    await asyncio.wait_for(
        bioengine_worker.is_ready.wait(), timeout=worker_ready_timeout
    )

    # Worker should be ready
    assert (
        bioengine_worker.is_ready.is_set()
    ), "Worker should be ready after initialization"

    # Worker should have a server connection
    assert (
        bioengine_worker._server is not None
    ), "Worker should have a server connection"

    # Worker should have admin context
    assert (
        bioengine_worker._admin_context is not None
    ), "Worker should have admin context"

    # Worker should have a service ID from service registration
    assert (
        bioengine_worker.full_service_id is not None
    ), "Worker should have a full service ID"

    # Worker should have a monitoring task
    assert (
        bioengine_worker._monitoring_task is not None
    ), "Worker should have a monitoring task"


@pytest.mark.asyncio
async def test_worker_service_registration(
    bioengine_worker: BioEngineWorker,
    hypha_client: RemoteService,
    worker_ready_timeout: int,
):
    """Test that the worker registers itself as a service in Hypha."""

    # Wait for worker to be ready
    await asyncio.wait_for(
        bioengine_worker.is_ready.wait(), timeout=worker_ready_timeout
    )

    # Find the worker service
    services = await hypha_client.list_services({"type": "bioengine-worker"})
    assert services, "No BioEngine Worker services found"
    assert len(services) >= 1, "Expected at least one BioEngine Worker service"

    # Get the worker service
    service_id = services[0].id
    worker_service = await hypha_client.get_service(service_id)
    assert worker_service, f"Could not retrieve worker service {service_id}"

    # Test worker readiness through service
    is_ready = await worker_service.is_ready()
    assert is_ready, "BioEngine Worker service reports not ready"


@pytest.mark.asyncio
async def test_startup_applications_deployment(
    bioengine_worker: BioEngineWorker,
    hypha_client: RemoteService,
    startup_applications: List[Dict],
    worker_ready_timeout: int,
):
    """Test that startup applications are deployed successfully."""

    # Wait for worker to be ready
    await asyncio.wait_for(
        bioengine_worker.is_ready.wait(), timeout=worker_ready_timeout
    )

    # Get worker service
    services = await hypha_client.list_services({"type": "bioengine-worker"})
    worker_service = await hypha_client.get_service(services[0].id)

    # Get worker status
    worker_status = await worker_service.get_status()
    bioengine_apps = worker_status["bioengine_apps"]

    # Verify each startup application
    for app_config in startup_applications:
        app_id = app_config["application_id"]

        # Check application is deployed
        assert (
            app_id in bioengine_apps
        ), f"Application {app_id} not found in worker status"

        app_status = bioengine_apps[app_id]
        assert (
            app_status["status"] == "RUNNING"
        ), f"Application {app_id} is not running: {app_status['status']}"

        # Verify service IDs are present
        service_ids = app_status.get("service_ids", {})
        assert (
            "webrtc_service_id" in service_ids
        ), f"WebRTC service ID missing for {app_id}"
        assert (
            "websocket_service_id" in service_ids
        ), f"WebSocket service ID missing for {app_id}"

        # Verify available methods
        available_methods = app_status.get("available_methods", [])
        assert (
            len(available_methods) > 0
        ), f"No available methods for application {app_id}"
        assert (
            "ping" in available_methods
        ), f"'ping' method not found for application {app_id}"


@pytest.mark.asyncio
async def test_application_websocket_connectivity(
    bioengine_worker: BioEngineWorker,
    hypha_client: RemoteService,
    startup_applications: List[Dict],
    worker_ready_timeout: int,
    application_check_timeout: int,
):
    """Test WebSocket connectivity to deployed applications."""

    # Wait for worker to be ready
    await asyncio.wait_for(
        bioengine_worker.is_ready.wait(), timeout=worker_ready_timeout
    )

    # Get worker service and status
    services = await hypha_client.list_services({"type": "bioengine-worker"})
    worker_service = await hypha_client.get_service(services[0].id)
    worker_status = await worker_service.get_status()
    bioengine_apps = worker_status["bioengine_apps"]

    for app_config in startup_applications:
        app_id = app_config["application_id"]
        app_status = bioengine_apps[app_id]

        # Get WebSocket service
        websocket_service_id = app_status["service_ids"]["websocket_service_id"]

        # Validate service ID format
        assert (
            "/" in websocket_service_id
        ), f"Invalid WebSocket service ID format: {websocket_service_id}"
        workspace, client_service = websocket_service_id.split("/", 1)
        assert ":" in client_service, f"Invalid client service format: {client_service}"
        client_id, service_name = client_service.split(":", 1)
        assert (
            service_name == app_id
        ), f"Service name mismatch: expected {app_id}, got {service_name}"

        # Connect to application service
        app_service = await hypha_client.get_service(websocket_service_id)
        assert (
            app_service
        ), f"Could not connect to WebSocket service {websocket_service_id}"

        # Test ping method
        try:
            result = await asyncio.wait_for(
                app_service.ping(), timeout=application_check_timeout
            )
            assert result, f"Ping failed for application {app_id}"
        except asyncio.TimeoutError:
            pytest.fail(f"Ping timeout for application {app_id}")


@pytest.mark.asyncio
async def test_application_webrtc_connectivity(
    bioengine_worker: BioEngineWorker,
    hypha_client: RemoteService,
    startup_applications: List[Dict],
    worker_ready_timeout: int,
    application_check_timeout: int,
):
    """Test WebRTC connectivity to deployed applications."""

    # Wait for worker to be ready
    await asyncio.wait_for(
        bioengine_worker.is_ready.wait(), timeout=worker_ready_timeout
    )

    # Get worker service and status
    services = await hypha_client.list_services({"type": "bioengine-worker"})
    worker_service = await hypha_client.get_service(services[0].id)
    worker_status = await worker_service.get_status()
    bioengine_apps = worker_status["bioengine_apps"]

    for app_config in startup_applications:
        app_id = app_config["application_id"]
        app_status = bioengine_apps[app_id]

        # Get WebRTC service
        webrtc_service_id = app_status["service_ids"]["webrtc_service_id"]
        assert webrtc_service_id.endswith(
            "rtc"
        ), f"Invalid WebRTC service ID: {webrtc_service_id}"

        # Connect to WebRTC service
        peer_connection = await get_rtc_service(hypha_client, webrtc_service_id)
        assert (
            peer_connection
        ), f"Could not connect to WebRTC service {webrtc_service_id}"

        try:
            # Get peer service
            peer_service = await peer_connection.get_service(app_id)
            assert peer_service, f"Could not get peer service {app_id} from WebRTC"

            # Test ping with admin context
            result = await asyncio.wait_for(
                peer_service.ping(context=bioengine_worker._admin_context),
                timeout=application_check_timeout,
            )
            assert result, f"WebRTC ping failed for application {app_id}"

        finally:
            # Clean up WebRTC connection
            await peer_connection.disconnect()
