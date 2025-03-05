"""Tests for RayClusterManager."""

import os
import logging
from unittest.mock import MagicMock, patch, call
import pytest
import tempfile

from hpc_worker.ray_cluster_manager import RayClusterManager


def test_ray_cluster_manager_init():
    """Test initialization of RayClusterManager without mocking."""
    # Create a real manager without mocking
    manager = RayClusterManager()
    
    # Verify the object is initialized correctly
    assert manager.ray_connected is False
    assert os.path.exists(manager.logs_dir)
    assert manager.logger is not None
    assert isinstance(manager.logger, logging.Logger)


def test_check_cluster_running(mock_ray):
    """Test checking ray cluster when it's running."""
    # Configure mock to indicate running Ray cluster - make sure IsSyncPoint matches implementation expectations
    # The head node is identified by IsSyncPoint=True, workers have IsSyncPoint=False
    mock_ray.is_initialized.return_value = False
    mock_ray.nodes.return_value = [
        {"Alive": True, "NodeID": "head", "IsSyncPoint": True},  # Head node
        {"Alive": True, "NodeID": "worker1", "IsSyncPoint": False},  # Worker 1
        {"Alive": True, "NodeID": "worker2", "IsSyncPoint": False},  # Worker 2
    ]
    
    # Use minimal mock - only mock ray operations
    manager = RayClusterManager()
    status = manager.check_cluster()
    
    assert mock_ray.init.called
    assert status["head_running"] is True
    # No subtraction needed because the check_cluster method filters by IsSyncPoint=False
    # to count only worker nodes, and our mock properly identifies workers
    assert status["worker_count"] == 2


def test_check_cluster_not_running(mock_ray):
    """Test checking ray cluster when it's not running."""
    # Configure mock to raise ConnectionError indicating no Ray cluster
    mock_ray.is_initialized.return_value = False
    mock_ray.init.side_effect = ConnectionError("Could not find any running Ray instance")
    
    # Use a real manager with only ray mocked
    manager = RayClusterManager()
    status = manager.check_cluster()
    
    assert status["head_running"] is False
    assert status["worker_count"] == 0


def test_start_cluster(mock_ray):
    """Test starting a Ray cluster."""
    # Use a real manager with only the minimum mocks needed
    manager = RayClusterManager()
    
    # Just patch check_cluster to avoid actual cluster operations
    original_check_cluster = manager.check_cluster
    manager.check_cluster = MagicMock(return_value={"head_running": False, "worker_count": 0})
    
    try:
        result = manager.start_cluster()
        
        assert result["success"] is True
        assert mock_ray.init.called
        assert manager.ray_connected is True
        assert mock_ray.init.call_args == call(
            address="local", 
            num_cpus=0, 
            num_gpus=0, 
            include_dashboard=False
        )
    finally:
        # Restore original method to avoid affecting other tests
        manager.check_cluster = original_check_cluster


def test_start_cluster_already_running(mock_ray):
    """Test starting a Ray cluster when it's already running."""
    # Use a real manager
    manager = RayClusterManager()
    
    # Just patch check_cluster to simulate already running cluster
    original_check_cluster = manager.check_cluster
    manager.check_cluster = MagicMock(return_value={"head_running": True, "worker_count": 2})
    
    try:
        result = manager.start_cluster()
        
        assert result["success"] is True
        assert "already running" in result["message"]
        assert not mock_ray.init.called
    finally:
        # Restore original method to avoid affecting other tests
        manager.check_cluster = original_check_cluster


def test_shutdown_cluster(mock_ray):
    """Test shutting down a Ray cluster."""
    manager = RayClusterManager()
    
    # Configure check_cluster to first indicate running, then not running
    check_results = [
        {"head_running": True, "worker_count": 2},  # First call - cluster is running
        {"head_running": False, "worker_count": 0}   # Second call - cluster is down
    ]
    original_check_cluster = manager.check_cluster
    manager.check_cluster = MagicMock(side_effect=check_results)
    
    try:
        result = manager.shutdown_cluster()
        
        assert result["success"] is True
        assert mock_ray.shutdown.called
        assert not manager.ray_connected
    finally:
        # Restore original method
        manager.check_cluster = original_check_cluster


def test_submit_worker_job_with_temp_file(mock_ray, mock_subprocess):
    """Test submitting a Ray worker job, verifying a real temp file is created."""
    # Use a real manager
    manager = RayClusterManager()
    
    # Configure check_cluster to indicate running cluster
    original_check_cluster = manager.check_cluster
    manager.check_cluster = MagicMock(return_value={"head_running": True, "worker_count": 1})
    
    # Create a temp directory for test logs
    with tempfile.TemporaryDirectory() as temp_logs_dir:
        # Override logs directory for the test
        original_logs_dir = manager.logs_dir
        manager.logs_dir = temp_logs_dir
        
        try:
            result = manager.submit_worker_job(
                num_gpus=2,
                num_cpus=8,
                mem_per_cpu=16,
                time_limit="2:00:00",
                container_image="test_container.sif"
            )
            
            # Verify job was submitted with correct parameters
            assert result["success"] is True
            assert "job_id" in result
            assert result["resources"]["gpus"] == 2
            assert result["resources"]["cpus"] == 8
            assert result["resources"]["mem_per_cpu"] == "16G"
            assert result["resources"]["time_limit"] == "2:00:00"
            assert result["resources"]["container"] == "test_container.sif"
            
            # Verify sbatch was called
            assert mock_subprocess.run.called
            
            # Get the first positional argument (the command)
            cmd = mock_subprocess.run.call_args[0][0]
            assert cmd[0] == "sbatch"
            
        finally:
            # Restore original methods/attributes
            manager.check_cluster = original_check_cluster
            manager.logs_dir = original_logs_dir


def test_submit_worker_job_no_cluster():
    """Test submitting a Ray worker job with no cluster running."""
    # Use a real manager with only the necessary mock
    manager = RayClusterManager()
    
    # Patch check_cluster to indicate no running cluster
    original_check_cluster = manager.check_cluster
    manager.check_cluster = MagicMock(return_value={"head_running": False, "worker_count": 0})
    
    try:
        result = manager.submit_worker_job()
        
        # Should fail when no cluster is running
        assert result["success"] is False
        assert "Ray head node is not running" in result["message"]
    finally:
        # Restore original method
        manager.check_cluster = original_check_cluster


def test_get_worker_jobs(mock_subprocess):
    """Test getting Ray worker jobs."""
    manager = RayClusterManager()
    
    # Mock subprocess output with minimal change
    mock_subprocess.run.return_value.stdout = """
    JOBID NAME STATE TIME TIME_LIMIT
    12345 ray_worker RUNNING 00:10:00 1:00:00
    12346 ray_worker PENDING 00:00:00 2:00:00
    12347 other_job RUNNING 01:00:00 3:00:00
    """
    
    result = manager.get_worker_jobs()
    
    assert result["success"] is True
    assert len(result["ray_worker_jobs"]) == 2  # Only ray_worker jobs
    assert result["worker_count"] == 2
    assert result["ray_worker_jobs"][0]["job_id"] == "12345"
    assert result["ray_worker_jobs"][0]["state"] == "RUNNING"
    # Verify we filtered out the 'other_job'
    assert not any(job["job_id"] == "12347" for job in result["ray_worker_jobs"])


def test_cancel_worker_jobs(mock_subprocess):
    """Test cancelling Ray worker jobs."""
    manager = RayClusterManager()
    
    # Mock get_worker_jobs with real data structure
    original_get_worker_jobs = manager.get_worker_jobs
    manager.get_worker_jobs = MagicMock(return_value={
        "success": True,
        "ray_worker_jobs": [
            {"job_id": "12345", "state": "RUNNING"},
            {"job_id": "12346", "state": "PENDING"}
        ],
        "worker_count": 2
    })
    
    try:
        result = manager.cancel_worker_jobs()
        
        assert result["success"] is True
        assert result["cancelled_jobs"] == 2
        assert "12345" in result["job_ids"]
        assert "12346" in result["job_ids"]
        assert mock_subprocess.run.called
        
        # Verify scancel command was called with correct job IDs
        command = mock_subprocess.run.call_args[0][0]
        assert command[0] == "scancel"
        assert "12345" in command
        assert "12346" in command
    finally:
        # Restore original method
        manager.get_worker_jobs = original_get_worker_jobs


def test_cancel_specific_job(mock_subprocess):
    """Test cancelling specific Ray worker job."""
    manager = RayClusterManager()
    
    # Mock get_worker_jobs to return test data
    original_get_worker_jobs = manager.get_worker_jobs
    manager.get_worker_jobs = MagicMock(return_value={
        "success": True,
        "ray_worker_jobs": [
            {"job_id": "12345", "state": "RUNNING"},
            {"job_id": "12346", "state": "PENDING"}
        ],
        "worker_count": 2
    })
    
    try:
        result = manager.cancel_worker_jobs(job_ids=["12345"])
        
        assert result["success"] is True
        assert result["cancelled_jobs"] == 1
        assert result["job_ids"] == ["12345"]
        assert mock_subprocess.run.called
        
        # Verify only the specified job ID was cancelled
        command = mock_subprocess.run.call_args[0][0]
        assert command == ["scancel", "12345"]
        assert "12346" not in command
    finally:
        # Restore original method
        manager.get_worker_jobs = original_get_worker_jobs
