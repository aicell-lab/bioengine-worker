import time
from pathlib import Path

import pytest

from bioengine_worker.worker import BioEngineWorker


@pytest.mark.asyncio
async def test_worker_initialization_and_readiness(
    cache_dir: Path,
    data_dir: Path,
    server_url: str,
    hypha_token: str,
):
    """Test comprehensive BioEngine worker initialization, readiness, and component functionality."""
    # Generate unique client ID for test isolation
    test_client_id = f"test_worker_{int(time.time())}"
    
    bioengine_worker = BioEngineWorker(
        mode="single-machine",
        admin_users=None,
        cache_dir=cache_dir,
        data_dir=data_dir,
        startup_applications=None,
        monitoring_interval_seconds=10,
        server_url=server_url,
        workspace=None,
        token=hypha_token,
        client_id=test_client_id,
        ray_cluster_config={
            "head_num_cpus": 1,
            "head_num_gpus": 0,
            "head_memory_in_gb": 1,
        },
        dashboard_url="https://bioimage.io/#/bioengine",
        log_file="off",
        debug=False,
        graceful_shutdown_timeout=60,
    )

    # ========================================================================
    # Test 1: Basic Worker Configuration and Initialization
    # ========================================================================
    
    # Verify worker configuration attributes are properly set
    assert bioengine_worker.cache_dir == cache_dir, "Cache directory should match configuration"
    assert bioengine_worker.data_dir == data_dir, "Data directory should match configuration"
    assert bioengine_worker.server_url == server_url, "Server URL should match configuration"
    assert bioengine_worker.client_id == test_client_id, "Client ID should match configuration"
    assert bioengine_worker.service_id == "bioengine-worker", "Service ID should be 'bioengine-worker'"
    assert bioengine_worker.graceful_shutdown_timeout == 60, "Graceful shutdown timeout should match configuration"
    
    # Verify initial state before starting
    assert not bioengine_worker.is_ready.is_set(), "Worker should not be ready before starting"
    assert bioengine_worker._server is None, "Server connection should be None before starting"
    assert bioengine_worker._admin_context is None, "Admin context should be None before starting"
    assert bioengine_worker.full_service_id is None, "Full service ID should be None before starting"
    assert bioengine_worker._monitoring_task is None, "Monitoring task should be None before starting"
    assert bioengine_worker.start_time is None, "Start time should be None before starting"

    # ========================================================================
    # Test 2: Component Managers Initialization
    # ========================================================================
    
    # Verify Ray cluster component is properly initialized
    assert bioengine_worker.ray_cluster is not None, "Ray cluster should be initialized"
    assert bioengine_worker.ray_cluster.mode == "single-machine", "Ray cluster mode should match configuration"
    assert not bioengine_worker.ray_cluster.is_ready.is_set(), "Ray cluster should not be ready before starting"
    
    # Verify apps manager component is properly initialized
    assert bioengine_worker.apps_manager is not None, "Apps manager should be initialized"
    assert bioengine_worker.apps_manager.ray_cluster == bioengine_worker.ray_cluster, "Apps manager should reference the same Ray cluster"
    assert bioengine_worker.apps_manager.startup_applications is None, "No startup applications should be configured"
    assert bioengine_worker.apps_manager.app_builder is not None, "Apps manager should have app builder"
    
    # Verify dataset manager component is properly initialized
    assert bioengine_worker.dataset_manager is not None, "Dataset manager should be initialized"
    assert hasattr(bioengine_worker.dataset_manager, '_datasets'), "Dataset manager should have datasets registry"

    # ========================================================================
    # Test 3: Worker Startup and Service Registration
    # ========================================================================
    
    # Record time before startup for validation
    startup_time = time.time()
    
    # Start the worker in non-blocking mode
    await bioengine_worker.start(blocking=False)

    # ========================================================================
    # Test 4: Post-Startup State Validation
    # ========================================================================
    
    # Worker should be ready and properly initialized
    assert bioengine_worker.is_ready.is_set(), "Worker should be ready after starting"
    assert bioengine_worker.start_time is not None, "Start time should be set after starting"
    assert bioengine_worker.start_time >= startup_time, "Start time should be reasonable"
    
    # Hypha server connection should be established
    assert bioengine_worker._server is not None, "Worker should have a server connection"
    assert bioengine_worker._server.config is not None, "Server config should be available"
    assert bioengine_worker._server.config.client_id == test_client_id, "Server client ID should match"
    
    # Admin context should be created from authenticated user
    assert bioengine_worker._admin_context is not None, "Worker should have admin context"
    assert "user" in bioengine_worker._admin_context, "Admin context should contain user information"
    assert "id" in bioengine_worker._admin_context["user"], "Admin context should contain user ID"
    assert "email" in bioengine_worker._admin_context["user"], "Admin context should contain user email"
    
    # Service registration should be completed
    assert bioengine_worker.full_service_id is not None, "Worker should have a full service ID"
    assert bioengine_worker.service_id in bioengine_worker.full_service_id, "Full service ID should contain service ID"
    
    # Monitoring task should be running
    assert bioengine_worker._monitoring_task is not None, "Worker should have a monitoring task"
    assert not bioengine_worker._monitoring_task.done(), "Monitoring task should be running"

    # ========================================================================
    # Test 5: Ray Cluster Status and Connectivity
    # ========================================================================
    
    # Ray cluster should be ready and connected
    assert bioengine_worker.ray_cluster.is_ready.is_set(), "Ray cluster should be ready after worker startup"
    assert bioengine_worker.ray_cluster.start_time is not None, "Ray cluster should have start time"
    assert bioengine_worker.ray_cluster.head_node_address is not None, "Ray cluster should have head node address"
    assert bioengine_worker.ray_cluster.serve_http_url is not None, "Ray cluster should have serve HTTP URL"
    
    # Ray cluster status should be available
    ray_cluster_status = bioengine_worker.ray_cluster.status
    assert ray_cluster_status is not None, "Ray cluster status should be available"
    assert "cluster" in ray_cluster_status, "Ray cluster status should contain cluster information"
    assert "nodes" in ray_cluster_status, "Ray cluster status should contain nodes information"
    
    # Cluster should have head node with configured resources
    cluster_info = ray_cluster_status["cluster"]
    assert cluster_info["total_cpu"] >= 1, "Cluster should have at least 1 CPU"
    assert cluster_info["total_gpu"] == 0, "Cluster should have 0 GPUs as configured"
    
    # ========================================================================
    # Test 6: Component Managers Initialization Status
    # ========================================================================
    
    # Apps manager should be initialized with server connection
    assert bioengine_worker.apps_manager.server is not None, "Apps manager should have server connection"
    assert bioengine_worker.apps_manager.admin_users is not None, "Apps manager should have admin users list"
    assert bioengine_worker.apps_manager.artifact_manager is not None, "Apps manager should have artifact manager"
    
    # Apps manager status should be available
    apps_status = await bioengine_worker.apps_manager.get_status()
    assert isinstance(apps_status, dict), "Apps manager status should be a dictionary"
    
    # Dataset manager should be initialized with server connection
    assert bioengine_worker.dataset_manager.server is not None, "Dataset manager should have server connection"
    assert bioengine_worker.dataset_manager.admin_users is not None, "Dataset manager should have admin users list"
    
    # Dataset manager status should be available
    dataset_status = await bioengine_worker.dataset_manager.get_status()
    assert isinstance(dataset_status, dict), "Dataset manager status should be a dictionary"

    # ========================================================================
    # Test 7: Worker Status API Functionality
    # ========================================================================
    
    # Worker status should be comprehensive and accurate
    worker_status = await bioengine_worker.get_status(context=bioengine_worker._admin_context)
    assert isinstance(worker_status, dict), "Worker status should be a dictionary"
    
    # Verify all expected status fields are present
    expected_status_fields = [
        "service_start_time", "service_uptime", "worker_mode", "workspace", 
        "client_id", "ray_cluster", "bioengine_apps", "bioengine_datasets",
        "admin_users", "is_ready"
    ]
    for field in expected_status_fields:
        assert field in worker_status, f"Worker status should contain '{field}' field"
    
    # Verify status field values
    assert worker_status["service_start_time"] == bioengine_worker.start_time, "Status start time should match worker start time"
    assert worker_status["service_uptime"] > 0, "Service uptime should be positive"
    assert worker_status["worker_mode"] == "single-machine", "Worker mode should match configuration"
    assert worker_status["client_id"] == test_client_id, "Status client ID should match configuration"
    assert worker_status["is_ready"] is True, "Worker status should indicate ready state"
    assert len(worker_status["admin_users"]) > 0, "Admin users list should not be empty"
    
    # Ray cluster status should be embedded in worker status
    assert isinstance(worker_status["ray_cluster"], dict), "Ray cluster status should be embedded"
    
    # Component statuses should be embedded
    assert isinstance(worker_status["bioengine_apps"], dict), "Apps manager status should be embedded"
    assert isinstance(worker_status["bioengine_datasets"], dict), "Dataset manager status should be embedded"

    # ========================================================================
    # Test 8: Admin Users Configuration
    # ========================================================================
    
    # Admin users should include the authenticated user
    assert bioengine_worker.admin_users is not None, "Admin users list should be set"
    assert len(bioengine_worker.admin_users) > 0, "Admin users list should not be empty"
    
    # The authenticated user should be in the admin users list
    user_id = bioengine_worker._server.config.user["id"]
    user_email = bioengine_worker._server.config.user["email"]
    assert user_id in bioengine_worker.admin_users or user_email in bioengine_worker.admin_users, \
        "Authenticated user should be in admin users list"

    # ========================================================================
    # Test 9: Graceful Shutdown and Cleanup
    # ========================================================================
    
    # Store monitoring task reference before shutdown (it may be set to None during cleanup)
    monitoring_task = bioengine_worker._monitoring_task
    assert monitoring_task is not None, "Monitoring task should exist before shutdown"
    assert not monitoring_task.done(), "Monitoring task should be running before shutdown"
    
    # Initiate graceful shutdown
    await bioengine_worker._stop(blocking=True)
    
    # Verify shutdown state
    assert monitoring_task.done(), "Monitoring task should be completed after shutdown"
    assert bioengine_worker._shutdown_event.is_set(), "Shutdown event should be set after graceful shutdown"
    
    # Ray cluster should be cleaned up (Note: is_ready may still be set during cleanup)
    # We don't test Ray cluster shutdown state here as it may vary based on cleanup timing
