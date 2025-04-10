"""Integration tests for RayClusterManager using real Ray instances."""

import os
import time
import ray
import pytest
import platform
import subprocess
from unittest.mock import patch
from bioengine_worker.ray_cluster_manager import RayClusterManager

def is_running_in_vscode():
    """Check if test is running in VSCode's test runner."""
    return "VSCODE_PID" in os.environ or "TERM_PROGRAM" in os.environ and os.environ["TERM_PROGRAM"] == "vscode"

@pytest.fixture(scope="module")
def ray_shutdown_after_module():
    """Make sure Ray is shut down after all tests in this module."""
    yield
    # Always make sure Ray is shut down at the end
    if ray.is_initialized():
        ray.shutdown()
    
    # In some cases, we need to force cleanup by calling ray stop
    try:
        subprocess.run(["ray", "stop"], timeout=10, capture_output=True)
    except:
        pass

@pytest.fixture
def real_ray_manager_cleanup():
    """Fixture that provides a RayClusterManager and ensures Ray cleanup."""
    # Create the manager
    manager = RayClusterManager()
    
    yield manager
    
    # Ensure Ray is shut down after the test
    if ray.is_initialized():
        ray.shutdown()


@pytest.mark.integration
def test_real_ray_cluster_lifecycle(ray_shutdown_after_module):
    """Test a complete Ray cluster lifecycle with real Ray instances."""
    # Skip in VSCode test runner to prevent crashes
    if is_running_in_vscode():
        pytest.skip("Skipping real Ray test in VSCode test runner")
    
    # Create the manager
    manager = RayClusterManager()
    
    try:
        # Initial state should be not connected
        assert not ray.is_initialized()
        
        # Check cluster status - should be not running
        status = manager.check_cluster()
        assert not status["head_running"]
        assert status["worker_count"] == 0
        
        # Start the cluster
        start_result = manager.start_cluster()
        assert start_result["success"] is True
        
        # Verify cluster is running
        assert ray.is_initialized()
        
        # Get runtime info to verify connection
        runtime_context = ray.get_runtime_context()
        gcs_address = runtime_context.gcs_address
        
        # Check status again - should be running
        status = manager.check_cluster()
        assert status["head_running"] is True
        
        # Shut down the cluster
        shutdown_result = manager.shutdown_cluster()
        assert shutdown_result["success"] is True
        
        # Verify shutdown
        assert not ray.is_initialized()
        status = manager.check_cluster()
        assert not status["head_running"]
        
    except Exception as e:
        pytest.fail(f"Test failed with exception: {str(e)}")
    finally:
        # Make sure Ray is always shut down
        if ray.is_initialized():
            try:
                ray.shutdown()
            except:
                pass


@pytest.mark.integration
def test_real_cluster_check(ray_shutdown_after_module):
    """Test checking a real Ray cluster."""
    # Skip in VSCode test runner to prevent crashes
    if is_running_in_vscode():
        pytest.skip("Skipping real Ray test in VSCode test runner")
    
    manager = RayClusterManager()
    
    try:
        # First make sure Ray is not running
        if ray.is_initialized():
            ray.shutdown()
        
        # Now start a Ray cluster with explicit parameters to avoid resource issues
        ray.init(
            address="local",
            num_cpus=1,
            num_gpus=0,
            resources={"custom_resource": 1},
            include_dashboard=False,
            ignore_reinit_error=True,
            logging_level=40  # ERROR level
        )
        
        # Give Ray a moment to stabilize
        time.sleep(1)
        
        # Check the cluster status
        status = manager.check_cluster()
        
        # Verify the cluster is detected properly
        assert status["head_running"] is True
        assert isinstance(status["worker_count"], int)
        
    except Exception as e:
        pytest.fail(f"Test failed with exception: {str(e)}")
    finally:
        # Always clean up
        try:
            if ray.is_initialized():
                ray.shutdown()
        except:
            pass
