import asyncio
import logging
import time
import ray
from typing import Dict, List, Optional, Union
import numpy as np

class RayAutoscaler:
    """Autoscaler for Ray clusters in HPC environments.
    
    This class monitors Ray cluster resource usage and auto-scales 
    worker nodes based on configured policies.
    """
    
    def __init__(
        self, 
        ray_manager,
        logger: Optional[logging.Logger] = None,
        # Autoscaling configuration
        min_workers: int = 0,
        max_workers: int = 4,
        target_gpu_utilization: float = 0.7,
        scale_up_cooldown_seconds: int = 60,
        scale_down_cooldown_seconds: int = 300,
        metrics_interval_seconds: int = 10,
        scale_up_threshold_seconds: int = 30,
        scale_down_threshold_seconds: int = 180,
        worker_startup_time_estimate_seconds: int = 120,
        gpus_per_worker: int = 1,
    ):
        """Initialize the Ray autoscaler.
        
        Args:
            ray_manager: RayClusterManager instance for cluster operations
            logger: Optional logger instance
            min_workers: Minimum number of worker nodes to maintain
            max_workers: Maximum number of worker nodes to allow
            target_gpu_utilization: Target GPU utilization (0-1)
            scale_up_cooldown_seconds: Minimum time between scaling up operations
            scale_down_cooldown_seconds: Minimum time between scaling down operations
            metrics_interval_seconds: Interval for collecting metrics
            scale_up_threshold_seconds: Time over threshold before scaling up
            scale_down_threshold_seconds: Time under threshold before scaling down
            worker_startup_time_estimate_seconds: Estimated time for worker startup
        """
        # Set up logging
        self.logger = logger or logging.getLogger("ray_autoscaler")
        if not logger:
            self.logger.setLevel(logging.INFO)
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            
        # Store ray manager reference
        self.ray_manager = ray_manager
        
        # Store configuration
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.target_gpu_utilization = target_gpu_utilization
        self.scale_up_cooldown = scale_up_cooldown_seconds
        self.scale_down_cooldown = scale_down_cooldown_seconds
        self.metrics_interval = metrics_interval_seconds
        self.scale_up_threshold = scale_up_threshold_seconds
        self.scale_down_threshold = scale_down_threshold_seconds
        self.worker_startup_estimate = worker_startup_time_estimate_seconds
        self.gpus_per_worker = gpus_per_worker
        
        # Initialize state variables
        self.last_scale_up_time = 0
        self.last_scale_down_time = 0
        self.utilization_history = {
            'timestamp': [],
            'gpu': [],
            'memory': [],
            'worker_count': [],
            'pending_tasks': []
        }
        
        # Background task
        self.monitoring_task = None
        self.is_running = False
        self.jobs_starting = {}  # Track jobs that have been submitted but may not be active yet
        
    async def start(self):
        """Start the autoscaler monitoring task."""
        if self.is_running:
            self.logger.info("Autoscaler is already running")
            return
            
        self.logger.info(
            f"Starting Ray autoscaler with config: "
            f"min_workers={self.min_workers}, max_workers={self.max_workers}, "
            f"target_gpu_util={self.target_gpu_utilization}, target_cpu_util={self.target_cpu_utilization}"
        )
        
        self.is_running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
    async def stop(self):
        """Stop the autoscaler monitoring task."""
        if not self.is_running:
            return
            
        self.logger.info("Stopping Ray autoscaler")
        self.is_running = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            self.monitoring_task = None
    
    async def _monitoring_loop(self):
        """Main monitoring loop for the autoscaler."""
        self.logger.info("Autoscaler monitoring loop started")
        
        try:
            while self.is_running:
                try:
                    # Collect and analyze metrics
                    await self._collect_metrics()
                    await self._make_scaling_decisions()
                    
                except Exception as e:
                    self.logger.error(f"Error in autoscaler monitoring loop: {str(e)}")
                
                # Sleep for the metrics interval
                await asyncio.sleep(self.metrics_interval)
                
        except asyncio.CancelledError:
            self.logger.info("Autoscaler monitoring loop cancelled")
        except Exception as e:
            self.logger.error(f"Unexpected error in monitoring loop: {str(e)}")
    
    async def _collect_metrics(self):
        """Collect resource utilization metrics from the Ray cluster."""
        # Verify Ray cluster is running
        ray_status = self.ray_manager.check_cluster()
        if not ray_status["head_running"]:
            self.logger.warning("Ray cluster is not running, skipping metrics collection")
            return
            
        # Get worker job status
        worker_jobs = self.ray_manager.get_worker_jobs()
        active_workers = worker_jobs.get("worker_count", 0)
        
        # Add pending workers that were recently submitted but may not be visible yet
        current_time = time.time()
        pending_jobs = 0
        for job_id, submit_time in list(self.jobs_starting.items()):
            if current_time - submit_time > self.worker_startup_estimate:
                # Job should be started by now, remove from tracking
                self.jobs_starting.pop(job_id, None)
            else:
                pending_jobs += 1
        
        # Estimate total workers including pending ones
        total_workers = active_workers + pending_jobs
        
        # Connect to Ray if needed
        was_connected = ray.is_initialized()
        if not was_connected:
            try:
                ray.init(address="auto")
            except Exception as e:
                self.logger.error(f"Failed to connect to Ray: {str(e)}")
                return
        
        try:
            # Get resource metrics
            resources = ray.cluster_resources()
            available = ray.available_resources()
            
            # Calculate utilization percentages
            metrics = {
                "timestamp": current_time,
                "worker_count": total_workers,
                "cpu": 0.0,
                "gpu": 0.0,
                "memory": 0.0,
                "pending_tasks": 0
            }
            
            # CPU utilization
            if "CPU" in resources and "CPU" in available:
                total_cpu = resources["CPU"]
                free_cpu = available.get("CPU", 0)
                if total_cpu > 0:
                    metrics["cpu"] = (total_cpu - free_cpu) / total_cpu
            
            # GPU utilization
            if "GPU" in resources and "GPU" in available:
                total_gpu = resources["GPU"]
                free_gpu = available.get("GPU", 0)
                if total_gpu > 0:
                    metrics["gpu"] = (total_gpu - free_gpu) / total_gpu
            
            # Memory utilization (use object store memory as proxy)
            if "object_store_memory" in resources and "object_store_memory" in available:
                total_mem = resources["object_store_memory"]
                free_mem = available.get("object_store_memory", 0)
                if total_mem > 0:
                    metrics["memory"] = (total_mem - free_mem) / total_mem
            
            # Get pending tasks from Ray dashboard
            # TODO: Implement this using Ray API when available
            metrics["pending_tasks"] = 0
            
            # Store metrics in history
            for key, value in metrics.items():
                if key in self.utilization_history:
                    self.utilization_history[key].append(value)
                    # Keep a reasonable history length
                    if len(self.utilization_history[key]) > 1000:
                        self.utilization_history[key] = self.utilization_history[key][-1000:]
            
            self.logger.debug(
                f"Metrics collected: workers={metrics['worker_count']}, "
                f"CPU={metrics['cpu']:.2f}, GPU={metrics['gpu']:.2f}, "
                f"memory={metrics['memory']:.2f}, pending_tasks={metrics['pending_tasks']}"
            )
                
        except Exception as e:
            self.logger.error(f"Error collecting Ray metrics: {str(e)}")
        
        finally:
            # Disconnect from Ray if we connected here
            if not was_connected and ray.is_initialized():
                ray.shutdown()
    
    async def _make_scaling_decisions(self):
        """Analyze metrics and make scaling decisions."""
        # Skip if we don't have enough history
        min_history = max(3, self.scale_up_threshold // self.metrics_interval)
        if len(self.utilization_history['timestamp']) < min_history:
            return
            
        current_time = time.time()
        recent_worker_count = self.utilization_history['worker_count'][-1]
        
        # Get recent GPU utilization average
        window_size = min(len(self.utilization_history['timestamp']), 5)
        recent_gpu_util = np.mean(self.utilization_history['gpu'][-window_size:])
        
        # Check if we need to scale up
        scale_up_window = self.scale_up_threshold // self.metrics_interval
        scale_up_window = min(scale_up_window, len(self.utilization_history['gpu']))
        
        gpu_high = np.mean(self.utilization_history['gpu'][-scale_up_window:]) > self.target_gpu_utilization
        
        should_scale_up = gpu_high and recent_worker_count < self.max_workers
        
        # Check cooldown period
        can_scale_up = (current_time - self.last_scale_up_time) > self.scale_up_cooldown
        
        if should_scale_up and can_scale_up:
            await self._scale_up()
        
        # Check if we need to scale down
        scale_down_window = self.scale_down_threshold // self.metrics_interval
        scale_down_window = min(scale_down_window, len(self.utilization_history['gpu']))
        
        gpu_low = np.mean(self.utilization_history['gpu'][-scale_down_window:]) < (self.target_gpu_utilization * 0.5)
        
        should_scale_down = gpu_low and recent_worker_count > self.min_workers
        
        # Check cooldown period
        can_scale_down = (current_time - self.last_scale_down_time) > self.scale_down_cooldown
        
        if should_scale_down and can_scale_down:
            await self._scale_down()
    
    async def _scale_up(self):
        """Scale up the Ray cluster by adding a worker."""
        self.logger.info("Initiating scale up operation")
        
        try:
            # Submit a worker job with the configured resources
            job_result = self.ray_manager.submit_worker_job(
                num_gpus=self.gpus_per_worker
            )
            
            if job_result["success"]:
                job_id = job_result["job_id"]
                self.logger.info(
                    f"Scaled up: Submitted worker job {job_id} "
                    f"with {self.gpus_per_worker} GPUs"
                )
                
                # Track this job as starting
                self.jobs_starting[job_id] = time.time()
                
                # Update last scale up time
                self.last_scale_up_time = time.time()
                
                return True
            else:
                self.logger.error(f"Failed to scale up: {job_result.get('message', 'Unknown error')}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error during scale up: {str(e)}")
            return False
    
    async def _scale_down(self):
        """Scale down the Ray cluster by removing a worker."""
        self.logger.info("Initiating scale down operation")
        
        try:
            # Get current worker jobs
            result = self.ray_manager.get_worker_jobs()
            
            if not result["success"] or not result["ray_worker_jobs"]:
                self.logger.warning("No worker jobs to remove during scale down")
                return False
            
            # Find the worker job with the shortest remaining time or least resources
            worker_jobs = result["ray_worker_jobs"]
            if worker_jobs:
                # For now, just take the first job (we could implement smarter selection)
                job_to_remove = worker_jobs[0]["job_id"]
                
                # Cancel the selected job
                cancel_result = self.ray_manager.cancel_worker_jobs(job_ids=[job_to_remove])
                
                if cancel_result["success"]:
                    self.logger.info(f"Scaled down: Removed worker job {job_to_remove}")
                    
                    # Update last scale down time
                    self.last_scale_down_time = time.time()
                    
                    return True
                else:
                    self.logger.error(f"Failed to scale down: {cancel_result.get('message', 'Unknown error')}")
                    return False
            else:
                self.logger.warning("No suitable worker jobs found for scale down")
                return False
                
        except Exception as e:
            self.logger.error(f"Error during scale down: {str(e)}")
            return False
    
    def get_autoscaler_status(self):
        """Get the current status of the autoscaler.
        
        Returns:
            Dict with autoscaler status information
        """
        # Calculate recent utilization metrics
        recent_metrics = {}
        for key in ['gpu', 'memory', 'worker_count']:
            if self.utilization_history[key]:
                window_size = min(5, len(self.utilization_history[key]))
                recent_metrics[key] = np.mean(self.utilization_history[key][-window_size:])
            else:
                recent_metrics[key] = 0.0
        
        # Get scaling cooldown information
        current_time = time.time()
        scale_up_cooldown_remaining = max(0, self.scale_up_cooldown - (current_time - self.last_scale_up_time))
        scale_down_cooldown_remaining = max(0, self.scale_down_cooldown - (current_time - self.last_scale_down_time))
        
        # Return formatted status
        return {
            "is_running": self.is_running,
            "config": {
                "min_workers": self.min_workers,
                "max_workers": self.max_workers,
                "target_gpu_utilization": self.target_gpu_utilization,
                "target_cpu_utilization": self.target_cpu_utilization,
                "scale_up_cooldown": self.scale_up_cooldown,
                "scale_down_cooldown": self.scale_down_cooldown,
            },
            "current_metrics": recent_metrics,
            "scaling_status": {
                "scale_up_cooldown_remaining": int(scale_up_cooldown_remaining),
                "scale_down_cooldown_remaining": int(scale_down_cooldown_remaining),
                "pending_workers": len(self.jobs_starting),
            }
        }

if __name__ == "__main__":
    """Test the RayAutoscaler class"""
    import asyncio
    from ray_cluster_manager import RayClusterManager
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    async def test_autoscaler():
        logger.info("Starting autoscaler test")
        
        # Create a RayClusterManager
        cluster_manager = RayClusterManager()
        
        # Start Ray cluster if not running
        cluster_status = cluster_manager.check_cluster()
        if not cluster_status["head_running"]:
            logger.info("Starting Ray cluster")
            start_result = cluster_manager.start_cluster()
            if not start_result["success"]:
                logger.error(f"Failed to start Ray cluster: {start_result}")
                return
            logger.info("Ray cluster started successfully")
        
        # Create and start the autoscaler
        autoscaler = RayAutoscaler(
            cluster_manager, 
            min_workers=1, 
            max_workers=3,
            metrics_interval_seconds=5  # Use shorter interval for testing
        )
        
        await autoscaler.start()
        
        try:
            # Run for some time to test scaling behavior
            logger.info("Autoscaler running - will test for 60 seconds")
            
            # Check status every 10 seconds
            for i in range(6):
                await asyncio.sleep(10)
                status = autoscaler.get_autoscaler_status()
                logger.info(f"Autoscaler status: workers={status['current_metrics']['worker_count']}, "
                           f"CPU={status['current_metrics']['cpu']:.2f}, "
                           f"GPU={status['current_metrics']['gpu']:.2f}")
            
        finally:
            # Stop the autoscaler
            await autoscaler.stop()
            logger.info("Autoscaler stopped")
            
            # Optionally stop the Ray cluster
            # cluster_manager.shutdown_cluster()
    
    # Run the test
    asyncio.run(test_autoscaler())
