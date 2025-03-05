"""Common fixtures and mocks for HPC Worker tests."""

import os
import sys
import tempfile
from unittest.mock import MagicMock, patch
import pytest
import yaml
import numpy as np
import logging

# Add the project root directory to Python's path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))


@pytest.fixture
def mock_ray():
    """Mock Ray module for testing."""
    with patch('hpc_worker.ray_cluster_manager.ray') as mock_ray:
        # Mock ray.is_initialized() to return False by default
        mock_ray.is_initialized.return_value = False
        
        # Mock ray.init()
        mock_ray.init = MagicMock()
        
        # Mock ray.shutdown()
        mock_ray.shutdown = MagicMock()
        
        # Mock ray.get_runtime_context()
        mock_context = MagicMock()
        mock_context.gcs_address = "127.0.0.1:6379"
        mock_ray.get_runtime_context.return_value = mock_context
        
        # Mock ray.nodes() - Make sure IsSyncPoint is correctly set:
        # Head node has IsSyncPoint=True, workers have IsSyncPoint=False
        mock_ray.nodes.return_value = [
            {"Alive": True, "NodeID": "head_node", "IsSyncPoint": True},  # Head node 
            {"Alive": True, "NodeID": "worker_node1", "IsSyncPoint": False},  # Worker 1
            {"Alive": True, "NodeID": "worker_node2", "IsSyncPoint": False},  # Worker 2
        ]
        
        yield mock_ray


@pytest.fixture
def mock_subprocess():
    """Mock subprocess module for testing."""
    with patch('hpc_worker.ray_cluster_manager.subprocess') as mock_subprocess:
        # Mock successful command execution
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = "Submitted batch job 12345"
        mock_process.stderr = ""
        
        # Mock run method
        mock_subprocess.run.return_value = mock_process
        mock_subprocess.CalledProcessError = Exception
        
        yield mock_subprocess


@pytest.fixture
def temp_config_file():
    """Create a temporary worker configuration file."""
    # Create a temporary file path but don't open it yet
    with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as temp_file:
        temp_file_path = temp_file.name
    
    # Create the config
    config = {
        'machine_name': 'test-machine',
        'max_gpus': 4,
        'dataset_paths': ['/path/to/dataset1', '/path/to/dataset2'],
        'trusted_models': ['ghcr.io/aicell-lab/tabula:0.1.1', 'ghcr.io/aicell-lab/model:1.0.0'],
        'server_url': 'https://hypha.aicell.io',
    }
    
    # Write the config to the file in text mode
    with open(temp_file_path, 'w') as f:
        yaml.dump(config, f)
    
    yield temp_file_path
    
    # Clean up
    try:
        os.unlink(temp_file_path)
    except:
        pass


@pytest.fixture
def mock_dataset():
    """Create a mock dataset with info.npz file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create mock info.npz file
        info_data = {'length': np.array([1000])}
        np.savez(os.path.join(temp_dir, 'info.npz'), **info_data)
        
        yield temp_dir


@pytest.fixture
def real_ray_manager():
    """Create a real (non-mocked) RayClusterManager for testing.
    
    This allows testing with a real Ray manager when appropriate,
    but without connecting to actual Ray clusters.
    """
    from hpc_worker.ray_cluster_manager import RayClusterManager
    
    # Create a logger for the test
    logger = logging.getLogger("test_ray_manager")
    logger.setLevel(logging.INFO)
    
    # Create the manager
    manager = RayClusterManager(logger=logger)
    
    # Patch only the check_cluster method to prevent it from trying 
    # to connect to a real Ray cluster during tests
    manager.check_cluster = MagicMock(return_value={"head_running": False, "worker_count": 0})
    
    return manager
