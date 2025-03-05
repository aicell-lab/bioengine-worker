"""Tests for HpcWorker class."""

import os
import time
import tempfile
from unittest.mock import MagicMock, patch
import pytest
import yaml
import asyncio
from datetime import timezone

from hpc_worker.hpc_worker import HpcWorker


@pytest.fixture
def hpc_worker(temp_config_file):
    """Create an HpcWorker instance with a temporary config.
    
    This uses minimal mocking - only mocking load_dataset_info since
    datasets aren't available in the test environment.
    """
    # Create a worker instance with the temp config file
    worker = HpcWorker(config_path=temp_config_file)
    
    # Mock the load_dataset_info method only
    worker.load_dataset_info = MagicMock(return_value={
        'name': 'test_dataset',
        'samples': 1000
    })
    
    return worker


def test_hpc_worker_init(temp_config_file):
    """Test initialization of HpcWorker."""
    worker = HpcWorker(config_path=temp_config_file)
    
    assert worker.config_path == temp_config_file
    assert worker.server_url == "https://hypha.aicell.io"
    assert worker.ray_manager is not None


def test_hpc_worker_create_config():
    """Test creation of worker configuration file."""
    with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as temp_file:
        config_path = temp_file.name
    
    try:
        # Create worker with config parameters
        worker = HpcWorker(
            config_path=config_path,
            num_gpu=2,
            dataset_paths=["/path/to/data1", "/path/to/data2"],
            trusted_models=["model:1.0", "model:2.0"]
        )
        
        # Verify config file was created
        assert os.path.exists(config_path)
        
        # Load and verify config contents
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        assert config['max_gpus'] == 2
        assert config['dataset_paths'] == ["/path/to/data1", "/path/to/data2"]
        assert config['trusted_models'] == ["model:1.0", "model:2.0"]
        
    finally:
        # Clean up
        if os.path.exists(config_path):
            os.unlink(config_path)


def test_hpc_worker_init_with_params():
    """Test initialization of HpcWorker with direct parameters."""
    with tempfile.TemporaryDirectory() as temp_dir:
        worker = HpcWorker(
            config_dir=temp_dir,
            num_gpu=2,
            dataset_paths=["/path/to/data1"],
            trusted_models=["model:1.0"],
            server_url="https://custom.server",
        )
        
        # Verify config file was created
        config_path = os.path.join(temp_dir, "worker_config.yaml")
        assert os.path.exists(config_path)
        
        # Load and verify config contents
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        assert config['max_gpus'] == 2
        assert config['dataset_paths'] == ["/path/to/data1"]
        assert config['trusted_models'] == ["model:1.0"]
        assert config['server_url'] == "https://custom.server"


def test_hpc_worker_init_missing_params():
    """Test initialization fails when required params are missing."""
    with pytest.raises(ValueError):
        # Missing trusted_models
        worker = HpcWorker(
            num_gpu=2,
            dataset_paths=["/path/to/data"],
            trusted_models=None,
        )


def test_format_time(hpc_worker):
    """Test formatting timestamp with duration."""
    # Test with current timestamp minus 1 hour
    timestamp = int(time.time()) - 3600
    
    time_info = hpc_worker.format_time(timestamp)
    
    assert "timestamp" in time_info
    assert "timezone" in time_info
    assert "duration_since" in time_info
    assert "1h" in time_info["duration_since"]


def test_format_time_complex_duration(hpc_worker):
    """Test formatting timestamp with complex duration."""
    # Test with current timestamp minus 3 days, 5 hours, 10 minutes, 15 seconds
    seconds = 3*86400 + 5*3600 + 10*60 + 15
    timestamp = int(time.time()) - seconds
    
    time_info = hpc_worker.format_time(timestamp)
    
    assert "3d" in time_info["duration_since"]
    assert "5h" in time_info["duration_since"]
    assert "10m" in time_info["duration_since"]
    assert "15s" in time_info["duration_since"]


def test_process_model_info(hpc_worker):
    """Test processing Docker image string into model info."""
    # Test with standard image format
    image = "ghcr.io/aicell-lab/tabula:0.1.1"
    
    model_info = hpc_worker.process_model_info(image)
    
    assert model_info['name'] == "tabula"
    assert model_info['image'] == image
    assert model_info['version'] == "0.1.1"
    
    # Test with another format
    image = "pytorch/pytorch:2.0.1-cuda11.7"
    
    model_info = hpc_worker.process_model_info(image)
    
    assert model_info['name'] == "pytorch"
    assert model_info['image'] == image
    assert model_info['version'] == "2.0.1-cuda11.7"


def test_get_worker_status(hpc_worker, temp_config_file):
    """Test getting complete worker status with minimal mocking."""
    # Update the config file with test data
    with open(temp_config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update config with test values - using fake dataset paths
    test_dataset_paths = ["/path/to/dataset1", "/path/to/dataset2"]
    config['dataset_paths'] = test_dataset_paths
    config['machine_name'] = 'test-machine'
    config['max_gpus'] = 4
    config['server_url'] = 'https://test.server'
    
    with open(temp_config_file, 'w') as f:
        yaml.dump(config, f)
    
    # Patch os.path.exists to return True for our test dataset paths
    def mock_path_exists(path):
        return path in test_dataset_paths or os.path.exists(path)
    
    # When calling get_worker_status, need to patch both path.exists and check_cluster
    with patch('os.path.exists', side_effect=mock_path_exists):
        with patch.object(hpc_worker.ray_manager, 'check_cluster') as mock_check_cluster:
            # Set up the mock to return a known status
            mock_check_cluster.return_value = {
                "head_running": True,
                "worker_count": 3
            }
            
            status = hpc_worker.get_worker_status()
            
            # Verify machine info
            assert status['machine']['name'] == 'test-machine'
            assert status['machine']['max_gpus'] == 4
            
            # Verify server URL is included
            assert status['server_url'] == 'https://test.server'
            
            # Verify ray cluster status
            assert status['ray_cluster']['head_node_running'] is True
            assert status['ray_cluster']['active_workers'] == 3
            
            # Verify dataset info (using mock)
            assert len(status['datasets']) == 2  # Should have two datasets from config
            assert status['datasets'][0]['name'] == 'test_dataset'
            assert status['datasets'][0]['samples'] == 1000
            assert status['datasets'][1]['name'] == 'test_dataset'
            assert status['datasets'][1]['samples'] == 1000
            
            # Verify registration time info
            assert 'registration' in status
            assert 'registered_at' in status['registration']
            assert 'uptime' in status['registration']


def test_register_service_sync():
    """Test registering the worker service with Hypha using real token."""
    # Skip if HYPHA_TOKEN is not available
    token = os.environ.get("HYPHA_TOKEN")
    if not token:
        pytest.skip("HYPHA_TOKEN not available")
    
    # Create a temporary config for this test
    with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as temp_file:
        temp_file_path = temp_file.name
    
    try:
        # Create worker instance
        worker = HpcWorker(
            config_path=temp_file_path,
            num_gpu=1,
            dataset_paths=["/path/to/data"],
            trusted_models=["model:1.0"],
            server_url="https://hypha.aicell.io"  # Use the production URL
        )
        
        # Call register with asyncio event loop - using the real Hypha connection
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(worker.register())
        
        # Verify result
        assert result['success'] is True
        assert 'service_id' in result
        assert 'service_url' in result
        
        # Verify service was registered (service info should be available)
        assert worker.service_info is not None
        assert worker.server is not None
        
        # Verify connection to Hypha server
        assert worker.server.config.workspace is not None
                
    finally:
        # Clean up
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)


def test_cleanup_sync(hpc_worker):
    """Test cleanup method properly shuts down resources."""
    # Mock the ray_manager methods
    hpc_worker.ray_manager.shutdown_cluster = MagicMock(return_value={"success": True})
    hpc_worker.ray_manager.cancel_worker_jobs = MagicMock(return_value={
        "success": True,
        "cancelled_jobs": 2
    })
    
    # Run async method using event loop
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(hpc_worker.cleanup())
    
    # Verify ray manager methods were called
    assert hpc_worker.ray_manager.shutdown_cluster.called
    assert hpc_worker.ray_manager.cancel_worker_jobs.called
    
    # Verify results structure
    assert "ray_shutdown" in result
    assert "jobs_cancel" in result
    assert result["ray_shutdown"]["success"] is True
    assert result["jobs_cancel"]["success"] is True
    assert result["jobs_cancel"]["cancelled_jobs"] == 2

# Remove or comment out previous async-marked tests if you're having troubles
# @pytest.mark.asyncio
# async def test_register_service():
#    ...

# @pytest.mark.asyncio
# async def test_cleanup(hpc_worker):
#    ...
