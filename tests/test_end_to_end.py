"""End-to-end integration test for BioEngine worker."""

import os
import asyncio
import tempfile
import time
import pytest
import ray
from bioengine_worker.worker import BioEngineWorker

@pytest.mark.end_to_end
def test_register_and_ray_operations():
    """Test full registration with real Hypha server and Ray operations.
    
    This test will:
    1. Connect to the real Hypha server
    2. Register the BioEngine worker service
    3. Start a real Ray cluster head node
    4. Test Ray cluster operations
    5. Clean up both Ray cluster and service
    
    Note: This test requires HYPHA_TOKEN in environment and should be run
    only in a controlled environment.
    """
    # Skip if HYPHA_TOKEN is not available
    token = os.environ.get("HYPHA_TOKEN")
    if not token:
        pytest.skip("HYPHA_TOKEN not available")
    
    # Create a temporary config for this test
    with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as temp_file:
        temp_file_path = temp_file.name
    
    try:
        # Create a worker with real parameters
        worker = BioEngineWorker(
            config_path=temp_file_path,
            num_gpu=1,
            dataset_paths=["/tmp"],  # Temporary path for testing
            trusted_models=["ghcr.io/aicell-lab/tabula:0.1.1"],
        )
        
        # Setup event loop
        loop = asyncio.get_event_loop()
        
        # Register with Hypha
        result = loop.run_until_complete(worker.register())
        assert result["success"] is True
        assert "service_url" in result
        
        print(f"Service registered successfully at {result['service_url']}")
        
        # Get worker status - cluster should be inactive initially
        status = worker.get_worker_status()
        assert not status["ray_cluster"]["head_node_running"]
        
        # Start a real Ray cluster
        print("Starting real Ray cluster...")
        start_result = worker.ray_manager.start_cluster()
        assert start_result["success"] is True
        print(f"Ray cluster started: {start_result['message']}")
        
        # Give the cluster a moment to stabilize
        time.sleep(2)
        
        # Check cluster status - now it should be running
        status = worker.get_worker_status()
        assert status["ray_cluster"]["head_node_running"] is True
        
        # Test Ray is actually running and accessible
        assert ray.is_initialized()
        
        # Get Ray runtime info
        runtime_context = ray.get_runtime_context()
        gcs_address = runtime_context.gcs_address
        print(f"Ray GCS address: {gcs_address}")
        
        # Try to get Ray nodes info
        nodes = ray.nodes()
        assert len(nodes) > 0
        print(f"Found {len(nodes)} Ray nodes")
        
        # Shut down Ray cluster
        print("Shutting down Ray cluster...")
        shutdown_result = worker.ray_manager.shutdown_cluster()
        assert shutdown_result["success"] is True
        print("Ray cluster shut down successfully")
        
        # Verify Ray is shut down
        assert not ray.is_initialized()
        
        # Final cleanup
        cleanup_result = loop.run_until_complete(worker.cleanup())
        assert cleanup_result is not None
        
    finally:
        # Ensure Ray is always shut down
        if ray.is_initialized():
            ray.shutdown()
            
        # Clean up config file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
