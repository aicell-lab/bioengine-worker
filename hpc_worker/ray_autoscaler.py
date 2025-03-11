import asyncio
import logging
import time
import ray
from ray.experimental.state.api import list_tasks
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
        # Autoscaling configuration parameters
        min_workers: int = 0,
        max_workers: int = 4,
        target_gpu_utilization: float = 0.7,
        scale_up_cooldown_seconds: int = 60,
        scale_down_cooldown_seconds: int = 300,
        metrics_interval_seconds: int = 10,
        scale_up_threshold_seconds: int = 30,
        scale_down_threshold_seconds: int = 180,
        idle_threshold: float = 0.1,  # Node is considered idle below this utilization
        node_grace_period_seconds: int = 300,  # Give new nodes time before considering for scale-down
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
            idle_threshold: Node is considered idle below this utilization
            node_grace_period_seconds: Give new nodes time before considering for scale-down
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
        self.autoscale_config = {
            'min_workers': min_workers,
            'max_workers': max_workers,
            'target_gpu_utilization': target_gpu_utilization,
            'scale_up_cooldown': scale_up_cooldown_seconds,
            'scale_down_cooldown': scale_down_cooldown_seconds,
            'metrics_interval': metrics_interval_seconds,
            'scale_up_threshold': scale_up_threshold_seconds,
            'scale_down_threshold': scale_down_threshold_seconds,
            'idle_threshold': idle_threshold,
            'node_grace_period': node_grace_period_seconds,
        }
        
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
        
        # Add node-specific tracking
        self.node_metrics = {}  # Track per-node metrics
        self.node_start_times = {}  # Track when nodes were added
        
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
            f"Starting Ray autoscaler with config {self.autoscale_config}"
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
                await asyncio.sleep(self.autoscale_config['metrics_interval'])
                
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
        worker_jobs_result = self.ray_manager.get_worker_jobs()
        worker_jobs = worker_jobs_result.get("ray_worker_jobs", [])
        
        # Count active and pending workers
        active_workers = sum(1 for job in worker_jobs if job["state"] == "RUNNING")
        pending_workers = sum(1 for job in worker_jobs if job["state"] in ["PENDING", "CONFIGURING"])
        total_workers = active_workers + pending_workers
        
        # Connect to Ray if needed
        was_connected = ray.is_initialized()
        if not was_connected:
            try:
                ray.init(address="auto")
            except Exception as e:
                self.logger.error(f"Failed to connect to Ray: {str(e)}")
                return
        
        try:
            # Get current tasks
            tasks = list_tasks()
            pending_tasks = sum(1 for task in tasks if task["state"] == "PENDING")
            
            # Enhanced task analysis
            gpu_tasks = sum(1 for task in tasks if task.get("required_resources", {}).get("GPU", 0) > 0)
            cpu_intensive_tasks = sum(1 for task in tasks if task.get("required_resources", {}).get("CPU", 0) >= 2)
            memory_intensive_tasks = sum(1 for task in tasks if 
                                        task.get("required_resources", {}).get("object_store_memory", 0) > 1e9)
            
            # Get resource metrics
            resources = ray.cluster_resources()
            available = ray.available_resources()
            
            # Calculate utilization percentages
            current_time = time.time()
            metrics = {
                "timestamp": current_time,
                "worker_count": total_workers,
                "active_workers": active_workers,
                "pending_workers": pending_workers,
                "pending_tasks": pending_tasks,
                "gpu_tasks": gpu_tasks,
                "cpu_intensive_tasks": cpu_intensive_tasks,
                "memory_intensive_tasks": memory_intensive_tasks,
                "cpu": 0.0,
                "gpu": 0.0,
                "memory": 0.0,
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
            
            # Memory utilization
            if "object_store_memory" in resources and "object_store_memory" in available:
                total_mem = resources["object_store_memory"]
                free_mem = available.get("object_store_memory", 0)
                if total_mem > 0:
                    metrics["memory"] = (total_mem - free_mem) / total_mem
            
            # Store metrics in history
            for key, value in metrics.items():
                if key in self.utilization_history:
                    self.utilization_history[key].append(value)
                    # Keep a reasonable history length
                    if len(self.utilization_history[key]) > 1000:
                        self.utilization_history[key] = self.utilization_history[key][-1000:]
                else:
                    # Initialize new metrics in history
                    self.utilization_history[key] = [value]
            
            self.logger.debug(
                f"Metrics collected: active_workers={active_workers}, "
                f"pending_workers={pending_workers}, "
                f"pending_tasks={pending_tasks}, GPU_tasks={gpu_tasks}, "
                f"CPU={metrics['cpu']:.2f}, GPU={metrics['gpu']:.2f}, "
                f"memory={metrics['memory']:.2f}"
            )
            
            # Get per-node metrics
            nodes = ray.nodes()
            current_node_ids = set()
            
            for node in nodes:
                node_id = node["NodeID"]
                current_node_ids.add(node_id)
                
                # Skip head node, we're only interested in worker nodes
                if node.get("IsHead", False):
                    continue
                    
                # Calculate node-specific metrics
                node_resources = node.get("Resources", {})
                node_available = node.get("AvailableResources", {})
                
                # Initialize this node's metrics if first time seen
                if node_id not in self.node_metrics:
                    self.node_metrics[node_id] = {
                        "timestamps": [],
                        "cpu_util": [],
                        "gpu_util": [],
                        "memory_util": [],
                        "ip_address": node.get("NodeManagerAddress"),
                        "last_active_time": current_time,
                        "idle_streak": 0
                    }
                    # Record node start time
                    self.node_start_times[node_id] = current_time
                    
                # CPU utilization
                node_cpu_util = 0.0
                if "CPU" in node_resources and "CPU" in node_available:
                    total_cpu = node_resources["CPU"]
                    free_cpu = node_available.get("CPU", 0)
                    if total_cpu > 0:
                        node_cpu_util = (total_cpu - free_cpu) / total_cpu
                        
                # GPU utilization
                node_gpu_util = 0.0
                if "GPU" in node_resources and "GPU" in node_available:
                    total_gpu = node_resources["GPU"]
                    free_gpu = node_available.get("GPU", 0)
                    if total_gpu > 0:
                        node_gpu_util = (total_gpu - free_gpu) / total_gpu
                
                # Memory utilization
                node_memory_util = 0.0
                if "object_store_memory" in node_resources and "object_store_memory" in node_available:
                    total_mem = node_resources["object_store_memory"]
                    free_mem = node_available.get("object_store_memory", 0)
                    if total_mem > 0:
                        node_memory_util = (total_mem - free_mem) / total_mem
                        
                # Store node metrics in history
                self.node_metrics[node_id]["timestamps"].append(current_time)
                self.node_metrics[node_id]["cpu_util"].append(node_cpu_util)
                self.node_metrics[node_id]["gpu_util"].append(node_gpu_util)
                self.node_metrics[node_id]["memory_util"].append(node_memory_util)
                
                # Limit history size
                max_history = 100
                if len(self.node_metrics[node_id]["timestamps"]) > max_history:
                    for key in ["timestamps", "cpu_util", "gpu_util", "memory_util"]:
                        self.node_metrics[node_id][key] = self.node_metrics[node_id][key][-max_history:]
                        
                # Track active/idle status
                # Node is considered active if GPU utilization is above threshold (removed CPU check)
                is_active = node_gpu_util > self.autoscale_config['idle_threshold']
                            
                if is_active:
                    self.node_metrics[node_id]["last_active_time"] = current_time
                    self.node_metrics[node_id]["idle_streak"] = 0
                else:
                    self.node_metrics[node_id]["idle_streak"] += 1
                    
            # Clean up metrics for nodes no longer in the cluster
            for node_id in list(self.node_metrics.keys()):
                if node_id not in current_node_ids:
                    self.node_metrics.pop(node_id, None)
                    self.node_start_times.pop(node_id, None)
                
        except Exception as e:
            self.logger.error(f"Error collecting Ray metrics: {str(e)}")
        
        finally:
            # Disconnect from Ray if we connected here
            if not was_connected and ray.is_initialized():
                ray.shutdown()
    
    async def _make_scaling_decisions(self):
        """Analyze metrics and make scaling decisions based primarily on GPU utilization."""
        # Skip if we don't have enough history
        min_history = max(3, self.autoscale_config['scale_up_threshold'] // self.autoscale_config['metrics_interval'])
        if len(self.utilization_history['timestamp']) < min_history:
            return
            
        current_time = time.time()
        recent_worker_count = self.utilization_history['worker_count'][-1]
        
        # Calculate both immediate and trend metrics
        window_size = min(len(self.utilization_history['timestamp']), 5)
        long_window = min(len(self.utilization_history['timestamp']), 10)
        
        # Recent metrics (short window average)
        recent_gpu_util = np.mean(self.utilization_history['gpu'][-window_size:])
        recent_pending_tasks = np.mean(self.utilization_history['pending_tasks'][-window_size:])
        
        # Calculate GPU trend indicators (if enough history exists)
        gpu_trend = 0
        if len(self.utilization_history['gpu']) >= long_window:
            gpu_now = np.mean(self.utilization_history['gpu'][-window_size:])
            gpu_past = np.mean(self.utilization_history['gpu'][-(long_window):-(long_window-window_size)])
            gpu_trend = gpu_now - gpu_past
        
        # Get task type information if available
        recent_gpu_tasks = np.mean(self.utilization_history.get('gpu_tasks', [0])[-window_size:])
        
        # SCALE UP LOGIC
        # -----------------------------
        # Calculate scale up signals
        scale_up_window = self.autoscale_config['scale_up_threshold'] // self.autoscale_config['metrics_interval']
        scale_up_window = min(scale_up_window, len(self.utilization_history['gpu']))
        
        # GPU-based signals
        gpu_high = np.mean(self.utilization_history['gpu'][-scale_up_window:]) > self.autoscale_config['target_gpu_utilization']
        
        # Task-based signals
        has_pending_tasks = recent_pending_tasks > 0
        has_gpu_tasks = recent_gpu_tasks > 0
        
        # Trend signals - detect rapidly increasing GPU load
        rapid_increase = gpu_trend > 0.1 and recent_gpu_util > 0.4
        
        # Combined decision logic - focusing primarily on GPU metrics
        should_scale_up = (gpu_high or (has_pending_tasks and has_gpu_tasks) or rapid_increase) and \
                         recent_worker_count < self.autoscale_config['max_workers']
        
        # Determine scale-up quantity based on load
        scale_up_quantity = 1  # Default
        if (recent_pending_tasks > 3 or recent_gpu_util > 0.9) and recent_worker_count < self.autoscale_config['max_workers'] - 1:
            # If many tasks are pending or GPU utilization is very high, consider scaling by 2
            scale_up_quantity = min(2, self.autoscale_config['max_workers'] - recent_worker_count)
        
        # Check cooldown period
        can_scale_up = (current_time - self.last_scale_up_time) > self.autoscale_config['scale_up_cooldown']
        
        if should_scale_up and can_scale_up:
            # Log the reasoning behind scaling decision
            reasons = []
            if gpu_high:
                reasons.append(f"high GPU utilization ({recent_gpu_util:.2f})")
            if has_pending_tasks and has_gpu_tasks:
                reasons.append(f"pending GPU tasks ({recent_pending_tasks})")
            if rapid_increase:
                reasons.append(f"rapidly increasing GPU load (trend: {gpu_trend:.2f})")
            
            self.logger.info(f"Scaling up due to: {', '.join(reasons)}")
            
            # Scale up by the determined quantity
            for _ in range(scale_up_quantity):
                success = await self._scale_up()
                if not success:
                    break
        
        # SCALE DOWN LOGIC
        # -----------------------------
        # Calculate scale down signals
        scale_down_window = self.autoscale_config['scale_down_threshold'] // self.autoscale_config['metrics_interval']
        scale_down_window = min(scale_down_window, len(self.utilization_history['gpu']))
        
        # GPU-based signals - use a lower threshold for scaling down
        target_low = self.autoscale_config['target_gpu_utilization'] * 0.5
        gpu_low = np.mean(self.utilization_history['gpu'][-scale_down_window:]) < target_low
        
        # Task-based signals
        no_pending_tasks = recent_pending_tasks == 0
        
        # Trend analysis - ensure we're not in a temporary lull
        stable_or_decreasing = gpu_trend <= 0
        
        # Combined decision logic with safety buffer - focus on GPU metrics
        should_scale_down = gpu_low and no_pending_tasks and stable_or_decreasing and \
                            recent_worker_count > self.autoscale_config['min_workers']
        
        # Check cooldown period
        can_scale_down = (current_time - self.last_scale_down_time) > self.autoscale_config['scale_down_cooldown']
        
        if should_scale_down and can_scale_down:
            # Log the reasoning behind scaling decision
            self.logger.info(
                f"Scaling down due to low GPU utilization: GPU={recent_gpu_util:.2f}, "
                f"no pending tasks, stable/decreasing trend"
            )
            await self._scale_down_idle_node()

    async def _scale_down_idle_node(self):
        """Scale down by removing the most GPU-idle worker node."""
        self.logger.info("Finding GPU-idle worker node for scale down")
        
        try:
            # Get current worker jobs
            result = self.ray_manager.get_worker_jobs()
            
            if not result["success"] or not result["ray_worker_jobs"]:
                self.logger.warning("No worker jobs available for scale down")
                return False
            
            worker_jobs = result["ray_worker_jobs"]
            current_time = time.time()
            
            # Find idle nodes - focus primarily on GPU idleness
            idle_nodes = []
            for node_id, metrics in self.node_metrics.items():
                # Skip nodes in their grace period
                node_age = current_time - self.node_start_times.get(node_id, 0)
                if node_age < self.autoscale_config['node_grace_period']:
                    self.logger.debug(f"Node {node_id} is too new to scale down ({node_age:.0f}s < {self.autoscale_config['node_grace_period']}s)")
                    continue
                    
                # Calculate recent GPU utilization
                history_len = min(5, len(metrics["gpu_util"]))
                if history_len < 3:  # Need minimum data points
                    continue
                    
                recent_gpu_util = np.mean(metrics["gpu_util"][-history_len:])
                idle_time = current_time - metrics["last_active_time"]
                
                # Node is considered idle primarily based on GPU utilization
                if recent_gpu_util < self.autoscale_config['idle_threshold'] and idle_time > 60:
                    # Store node with its GPU idleness score (higher means more idle)
                    idle_score = idle_time * (1.0 - recent_gpu_util)
                    idle_nodes.append((node_id, idle_score, metrics["ip_address"]))
            
            if not idle_nodes:
                self.logger.info("No GPU-idle nodes found for scale down")
                return False
                
            # Sort nodes by idle score (most idle first)
            idle_nodes.sort(key=lambda x: x[1], reverse=True)
            most_idle_node = idle_nodes[0]
            self.logger.info(f"Selected GPU-idle node {most_idle_node[0]} (IP: {most_idle_node[2]}) "
                           f"for scale down, idle score: {most_idle_node[1]:.2f}")
            
            # Find the job matching this node
            # TODO: Implement proper node_id to job_id mapping
            job_to_remove = worker_jobs[0]["job_id"]
            
            # Cancel the selected job
            cancel_result = self.ray_manager.cancel_worker_jobs(job_ids=[job_to_remove])
            
            if cancel_result["success"]:
                self.logger.info(f"Scaled down: Removed GPU-idle worker job {job_to_remove}")
                
                # Update last scale down time
                self.last_scale_down_time = current_time
                
                # Clean up node tracking
                self.node_metrics.pop(most_idle_node[0], None)
                self.node_start_times.pop(most_idle_node[0], None)
                
                return True
            else:
                self.logger.error(f"Failed to scale down: {cancel_result.get('message', 'Unknown error')}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error during selective scale down: {str(e)}")
            return False
    
    async def _scale_up(self):
        """Scale up the Ray cluster by adding a worker."""
        self.logger.info("Initiating scale up operation")
        
        try:
            # Submit a worker job with the configured resources
            job_result = self.ray_manager.submit_worker_job()
            
            if job_result["success"]:
                # Track this job as starting
                self.jobs_starting[job_result["job_id"]] = time.time()
                
                # Update last scale up time
                self.last_scale_up_time = time.time()
                
                return True
            else:
                self.logger.error(f"Failed to scale up: {job_result.get('message', 'Unknown error')}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error during scale up: {str(e)}")
            return False
    
    def get_autoscaler_status(self):
        """Get the current status of the autoscaler.
        
        Returns:
            Dict with autoscaler status information
        """
        # Calculate recent utilization metrics
        recent_metrics = {
            "worker_count": 0,
            "active_workers": 0,
            "pending_workers": 0,
            "pending_tasks": 0,
            "cpu": 0.0,
            "gpu": 0.0,
            "memory": 0.0,
        }        
        for key in recent_metrics.keys():
            if self.utilization_history[key]:
                if key in ["cpu", "gpu", "memory"]:
                    window_size = min(5, len(self.utilization_history[key]))
                    recent_metrics[key] = np.mean(self.utilization_history[key][-window_size:])
                else:
                    recent_metrics[key] = self.utilization_history[key][-1]
        
        # Get scaling cooldown information
        current_time = time.time()
        scale_up_cooldown_remaining = max(0, self.autoscale_config['scale_up_cooldown'] - 
                                        (current_time - self.last_scale_up_time))
        scale_down_cooldown_remaining = max(0, self.autoscale_config['scale_down_cooldown'] - 
                                          (current_time - self.last_scale_down_time))
        
        # Add node-specific metrics to the status
        node_status = {}
        current_time = time.time()
        
        for node_id, metrics in self.node_metrics.items():
            if not metrics["timestamps"]:
                continue
                
            # Calculate recent utilization
            history_len = min(5, len(metrics["gpu_util"]))
            recent_gpu_util = np.mean(metrics["gpu_util"][-history_len:]) if history_len > 0 else 0
            recent_cpu_util = np.mean(metrics["cpu_util"][-history_len:]) if history_len > 0 else 0
            recent_memory_util = np.mean(metrics["memory_util"][-history_len:]) if history_len > 0 else 0
            
            # Calculate idle time
            idle_time = current_time - metrics["last_active_time"]
            
            # Calculate node age
            node_age = current_time - self.node_start_times.get(node_id, current_time)
            
            node_status[node_id] = {
                "ip_address": metrics["ip_address"],
                "gpu_utilization": recent_gpu_util,
                "idle_time_seconds": idle_time,
                "age_seconds": node_age,
                "is_idle": recent_gpu_util < self.autoscale_config['idle_threshold'],
                "in_grace_period": node_age < self.autoscale_config['node_grace_period']
            }
        
        # Return formatted status
        return {
            "is_running": self.is_running,
            "config": self.autoscale_config,  # Return entire config dict instead of subset
            "current_metrics": recent_metrics,
            "scaling_status": {
            "scale_up_cooldown_remaining": int(scale_up_cooldown_remaining),
            "scale_down_cooldown_remaining": int(scale_down_cooldown_remaining),
            "pending_workers": len(self.jobs_starting),
            },
            "nodes": node_status
        }

if __name__ == "__main__":
    """Test the RayAutoscaler class"""
    import asyncio
    from hpc_worker.ray_cluster_manager import RayClusterManager
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    async def test_autoscaler():
        logger.info("Starting autoscaler test")
        
        # Create a RayClusterManager
        cluster_manager = RayClusterManager(
            logger=logger,
            # Use shorter time for testing
            time_limit="1:00:00",
        )
        
        # Start Ray cluster if not running
        logger.info("Starting Ray cluster")
        start_result = cluster_manager.start_cluster()
        if not start_result["success"]:
            logger.error(f"Failed to start Ray cluster: {start_result}")
            return
        logger.info("Ray cluster started successfully")
        
        # Create and start the autoscaler
        autoscaler = RayAutoscaler(
            cluster_manager,
            logger=logger,
            # Use shorter time for testing
            metrics_interval_seconds=5,  
            scale_up_threshold_seconds=10,
            scale_down_threshold_seconds=20,
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
            
            # Shutdown the Ray cluster
            cluster_manager.cancel_worker_jobs()
            cluster_manager.shutdown_cluster()
    
    # Run the test
    asyncio.run(test_autoscaler())
