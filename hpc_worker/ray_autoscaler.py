import asyncio
import logging
import time
from typing import Optional

import numpy as np
import ray
from ray.util.state import list_tasks


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
        metrics_interval_seconds: int = 10,
        gpu_idle_threshold: float = 0.1,
        cpu_idle_threshold: float = 0.1,
        scale_down_threshold_seconds: int = 180,
        scale_up_cooldown_seconds: int = 60,
        scale_down_cooldown_seconds: int = 300,
        node_grace_period_seconds: int = 300,
    ):
        """Initialize the Ray autoscaler.

        Args:
            ray_manager: RayClusterManager instance for cluster operations
            logger: Optional logger instance
            min_workers: Minimum number of worker nodes to maintain
            max_workers: Maximum number of worker nodes to allow
            metrics_interval_seconds: Interval for collecting metrics
            gpu_idle_threshold: Node is considered idle below this GPU utilization
            cpu_idle_threshold: Node is considered idle below this CPU utilization
            scale_down_threshold_seconds: Idle time under threshold before scaling down
            scale_up_cooldown_seconds: Minimum time between scaling up operations
            scale_down_cooldown_seconds: Minimum time between scaling down operations
            node_grace_period_seconds: Give new nodes time before considering for scale-down
        """
        # Set up logging
        self.logger = logger or logging.getLogger("RayAutoscaler")
        if not logger:
            self.logger.setLevel(logging.INFO)
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "\033[36m%(asctime)s\033[0m - \033[32m%(name)s\033[0m - \033[1;33m%(levelname)s\033[0m - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # Store ray manager reference
        self.ray_manager = ray_manager

        # Store configuration
        self.autoscale_config = {
            "min_workers": min_workers,
            "max_workers": max_workers,
            "metrics_interval": metrics_interval_seconds,
            "gpu_idle_threshold": gpu_idle_threshold,
            "cpu_idle_threshold": cpu_idle_threshold,
            "scale_down_threshold": scale_down_threshold_seconds,
            "scale_up_cooldown": scale_up_cooldown_seconds,
            "scale_down_cooldown": scale_down_cooldown_seconds,
            "node_grace_period": node_grace_period_seconds,
        }

        # Initialize state variables
        self.last_scale_up_time = 0
        self.last_scale_down_time = 0
        self.node_metrics = {}  # Track per-node metrics

        # Background task
        self.monitoring_task = None
        self.is_running = False

    async def start(self):
        """Start the autoscaler monitoring task."""
        if self.is_running:
            self.logger.info("Autoscaler is already running")
            return

        if not ray.is_initialized():
            self.logger.error("Ray is not initialized, cannot start autoscaler")
            return

        self.logger.info(f"Starting Ray autoscaler with config {self.autoscale_config}")

        self.is_running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())

    async def stop(self):
        """Stop the autoscaler monitoring task."""
        if not self.is_running:
            return

        self.logger.info("Stopping Ray autoscaler")
        self.is_running = False

        # Clear statistics or perform any cleanup necessary
        self.node_metrics.clear()

        # Cancel monitoring task
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
                    # Check if Ray is still initialized
                    if not ray.is_initialized():
                        self.logger.error("Ray is not initialized, stopping autoscaler")
                        await self.stop()
                        break

                    # Collect and analyze metrics
                    node_metrics, dead_node_ids = await self._collect_metrics(self.node_metrics.copy())
                    node_metrics = await self._make_scaling_decisions(node_metrics)
                    await self._cleanup_dead_nodes(dead_node_ids)

                    # Update node metrics
                    self.node_metrics = node_metrics

                    # Sleep for the metrics interval
                    await asyncio.sleep(self.autoscale_config["metrics_interval"])
                except Exception as e:
                    self.logger.error(f"Error in monitoring loop: {str(e)}")
                    # Add delay after errors to prevent rapid error loops
                    await asyncio.sleep(min(self.autoscale_config["metrics_interval"], 5))
        except asyncio.CancelledError:
            self.logger.info("Autoscaler monitoring loop cancelled")
            raise  # Re-raise to properly propagate cancellation

    async def _collect_metrics(self, node_metrics: dict):
        """Collect resource utilization metrics from worker nodes using cluster status."""

        # Get current cluster status
        current_time = time.time()
        cluster_status = self.ray_manager.cluster_status()
        active_nodes = cluster_status["worker_nodes"]["Alive"]
        dead_nodes = cluster_status["worker_nodes"]["Dead"]

        # Remove metrics for nodes that are no longer in cluster
        current_worker_ids = {node["WorkerID"] for node in active_nodes}
        for worker_id in list(node_metrics.keys()):
            if worker_id not in current_worker_ids:
                node_metrics.pop(worker_id, None)

        # Update metrics for active nodes
        for node in active_nodes:
            worker_id = node["WorkerID"]

            # Initialize metrics for new nodes
            if worker_id not in node_metrics:
                node_metrics[worker_id] = {
                    "ip_address": node["NodeIP"],
                    "node_id": node["NodeID"],
                    "start_time": current_time,
                    "last_active_time": current_time,
                    "timestamps": [],
                    "gpu_util": [],
                    "cpu_util": [],
                    "memory_util": [],
                }

            # Calculate utilization based on available resources
            if node["Total GPU"] > 0:
                used_gpu = node["Total GPU"] - node["Available GPU"]
                gpu_util = used_gpu / node["Total GPU"]
            else:
                gpu_util = 0

            if node["Total CPU"] > 0:
                used_cpu = node["Total CPU"] - node["Available CPU"]
                cpu_util = used_cpu / node["Total CPU"]
            else:
                cpu_util = 0

            if node["Total Memory"] > 0:
                used_memory = node["Total Memory"] - node["Available Memory"]
                memory_util = used_memory / node["Total Memory"]
            else:
                memory_util = 0

            node_metrics[worker_id]["timestamps"].append(current_time)
            node_metrics[worker_id]["gpu_util"].append(gpu_util)
            node_metrics[worker_id]["cpu_util"].append(cpu_util)
            node_metrics[worker_id]["memory_util"].append(memory_util)

            # Update active/idle status based on both GPU and CPU utilization
            if (
                gpu_util >= self.autoscale_config["gpu_idle_threshold"]
                or cpu_util >= self.autoscale_config["cpu_idle_threshold"]
            ):
                node_metrics[worker_id]["last_active_time"] = current_time

            # Limit history size
            max_history = 100
            if len(node_metrics[worker_id]["timestamps"]) > max_history:
                for key in ["timestamps", "gpu_util", "cpu_util", "memory_util"]:
                    node_metrics[worker_id][key] = node_metrics[worker_id][
                        key
                    ][-max_history:]

        # Collect dead nodes for cleanup
        if dead_nodes:
            dead_node_ids = {node["WorkerID"] for node in dead_nodes}
        else:
            dead_node_ids = set()

        return node_metrics, dead_node_ids

    def _get_pending_tasks(self):
        """Get pending tasks that need resources."""
        head_node_ip = self.ray_manager.ray_cluster_config["head_node_ip"]
        head_node_port = self.ray_manager.ray_cluster_config["head_node_port"]
        address = f"{head_node_ip}:{head_node_port}"
        pending_tasks = list_tasks(
            address=address,
            filters=[("state", "=", "PENDING_NODE_ASSIGNMENT")]
        )
        if len(pending_tasks) < 0:
            self.logger.error("Error getting pending tasks")
        return pending_tasks

    def _get_num_worker_jobs(self):
        """Get number of SLURM workers in the cluster (configuring, pending or running)."""
        result = self.ray_manager.get_worker_jobs()
        return len(result["ray_worker_jobs"])

    def _scale_up(
        self,
        num_gpus: int = 1,
        num_cpus: int = 4,
        mem_per_cpu: int = 8,
        time_limit: str = "4:00:00",
    ):
        """Scale up the cluster by adding a new worker node."""
        job_result = self.ray_manager.submit_worker_job(
            num_gpus=num_gpus,
            num_cpus=num_cpus,
            mem_per_cpu=mem_per_cpu,
            time_limit=time_limit,
        )
        if job_result["success"]:
            self.last_scale_up_time = time.time()

    async def _make_scaling_decisions(self, node_metrics: dict):
        """Make scaling decisions based on resource utilization and pending tasks."""
        current_time = time.time()
        n_worker_jobs = self._get_num_worker_jobs()
        pending_tasks = self._get_pending_tasks()

        # If at least one task needs resources, check scale up, otherwise check scale down
        if len(pending_tasks) > 0:

            # SCALE UP LOGIC
            can_scale_up = (
                # Cooldown period has passed
                (current_time - self.last_scale_up_time)
                > self.autoscale_config["scale_up_cooldown"]
                # Not at max worker limit
                and n_worker_jobs
                < self.autoscale_config["max_workers"]
            )
            if not can_scale_up:
                return node_metrics

            # Determine resource needs based on pending tasks (simplified example)

            # Default configuration
            num_gpus = 1
            num_cpus = 4
            mem_per_cpu = 8
            time_limit = "4:00:00"

            # Check if any pending task needs more resources
            # TODO: check why resources are not specified with `@ray.remote(num_cpus=1, num_gpus=1)`
            for task in pending_tasks:
                task_req = task.get("required_resources", {}) or {}
                if task_req.get("GPU", 0) > num_gpus:
                    num_gpus = max(num_gpus, int(task_req.get("GPU", 1)))
                if task_req.get("CPU", 0) > num_cpus:
                    num_cpus = max(num_cpus, int(task_req.get("CPU", 4)))

            self.logger.info(
                f"Scaling up with {num_gpus} GPU(s) and {num_cpus} CPU(s) due to {len(pending_tasks)} pending task(s)"
            )
            self._scale_up(
                num_gpus=num_gpus,
                num_cpus=num_cpus,
                mem_per_cpu=mem_per_cpu,
                time_limit=time_limit,
            )
        else:
            # SCALE DOWN LOGIC
            longest_idle_time = 0
            longest_idle_worker = None
            for worker_id, metrics in node_metrics.items():
                node_age = current_time - metrics["start_time"]

                # Skip nodes in grace period
                if node_age < self.autoscale_config["node_grace_period"]:
                    continue

                # Check how long node has been idle
                idle_time = current_time - metrics["last_active_time"]

                can_scale_down = (
                    # Node has been idle for too long
                    idle_time > self.autoscale_config["scale_down_threshold"]
                    # Cooldown period has passed
                    and (current_time - self.last_scale_down_time)
                    > self.autoscale_config["scale_down_cooldown"]
                    # Not at min worker limit
                    and n_worker_jobs > self.autoscale_config["min_workers"]
                    # Longest idle worker
                    and idle_time > longest_idle_time
                )
                if can_scale_down:
                    longest_idle_time = idle_time
                    longest_idle_worker = worker_id

            if longest_idle_worker:
                self.logger.info(
                    f"Scaling down node {longest_idle_worker} due to inactivity for {longest_idle_time}s"
                )
                result = self.ray_manager.shutdown_worker_node(longest_idle_worker)
                if result["success"]:
                    self.last_scale_down_time = current_time
                    node_metrics.pop(longest_idle_worker, None)

        return node_metrics

    async def _cleanup_dead_nodes(self, dead_node_ids: set):
        if dead_node_ids:
            self.logger.info(f"Cleaning up {len(dead_node_ids)} dead nodes from cluster: {dead_node_ids}")
            for worker_id in dead_node_ids:
                self.ray_manager.shutdown_worker_node(worker_id)

    def get_autoscaler_status(self, history_len: int = 5) -> dict:
        """Get the current status of the autoscaler.

        Args:
            history_len: Number of recent data points to use for averaging. Must be > 0.

        Returns:
            Dict containing cluster status, node metrics and scaling information
        """
        current_time = time.time()
        history_len = max(1, min(history_len, 100))  # Ensure valid history length

        node_metrics = self.node_metrics.copy()
        num_ray_workers = len(node_metrics)

        # Get pending jobs and tasks
        n_worker_jobs = self._get_num_worker_jobs()
        pending_workers = n_worker_jobs - num_ray_workers
        pending_tasks = len(self._get_pending_tasks())

        # Initialize metrics
        recent_metrics = {
            "worker_count": n_worker_jobs,
            "active_workers": 0,
            "idle_workers": 0,
            "pending_workers": pending_workers,
            "pending_tasks": pending_tasks,
            "average_cpu": 0.0,
            "average_gpu": 0.0,
            "average_memory": 0.0,
        }

        node_status = {}
        if node_metrics:
            total_metrics = {"cpu": 0.0, "gpu": 0.0, "memory": 0.0}
            
            for worker_id, metrics in node_metrics.items():
                if not metrics["timestamps"]:
                    recent_metrics["idle_workers"] += 1
                    continue

                # Get recent metrics using numpy for efficiency
                recent_slice = slice(-history_len, None)
                recent_cpu_util = float(np.mean(metrics["cpu_util"][recent_slice]))
                recent_gpu_util = float(np.mean(metrics["gpu_util"][recent_slice]))
                recent_memory_util = float(np.mean(metrics["memory_util"][recent_slice]))

                total_metrics["cpu"] += recent_cpu_util
                total_metrics["gpu"] += recent_gpu_util
                total_metrics["memory"] += recent_memory_util

                # Determine if node is idle based on current utilization
                cpu_is_idle = metrics["cpu_util"][-1] < self.autoscale_config["cpu_idle_threshold"]
                gpu_is_idle = metrics["gpu_util"][-1] < self.autoscale_config["gpu_idle_threshold"]
                
                if cpu_is_idle and gpu_is_idle:
                    recent_metrics["idle_workers"] += 1
                    idle_time = current_time - metrics["last_active_time"]
                else:
                    recent_metrics["active_workers"] += 1
                    idle_time = 0

                node_age = current_time - metrics["start_time"]
                
                node_status[worker_id] = {
                    "ip_address": metrics["ip_address"],
                    "node_id": metrics["node_id"],
                    "age_seconds": int(node_age),
                    "in_grace_period": node_age < self.autoscale_config["node_grace_period"],
                    "idle_time_seconds": int(idle_time),
                    "cpu_is_idle": cpu_is_idle,
                    "gpu_is_idle": gpu_is_idle,
                    "average_cpu_utilization": round(recent_cpu_util, 3),
                    "average_gpu_utilization": round(recent_gpu_util, 3),
                    "average_memory_utilization": round(recent_memory_util, 3),
                }

            # Calculate cluster-wide averages safely
            recent_metrics["average_cpu"] = round(total_metrics["cpu"] / num_ray_workers, 3)
            recent_metrics["average_gpu"] = round(total_metrics["gpu"] / num_ray_workers, 3)
            recent_metrics["average_memory"] = round(total_metrics["memory"] / num_ray_workers, 3)

        # Calculate cooldown times
        scale_up_cooldown = max(0, self.autoscale_config["scale_up_cooldown"] - 
                              (current_time - self.last_scale_up_time))
        scale_down_cooldown = max(0, self.autoscale_config["scale_down_cooldown"] - 
                                (current_time - self.last_scale_down_time))

        self.logger.info(
            f"Autoscaler status: {recent_metrics['worker_count']} worker(s) "
            f"({recent_metrics['active_workers']} active, {recent_metrics['idle_workers']} idle, "
            f"{recent_metrics['pending_workers']} pending), {recent_metrics['pending_tasks']} pending task(s), "
            f"{scale_up_cooldown:.1f}s until scale up, "
            f"{scale_down_cooldown:.1f}s until scale down"
        )

        return {
            "is_running": self.is_running,
            "current_metrics": recent_metrics,
            "scaling_status": {
                "scale_up_cooldown_remaining": int(scale_up_cooldown),
                "scale_down_cooldown_remaining": int(scale_down_cooldown),
            },
            "nodes": node_status,
        }


if __name__ == "__main__":
    """Test the RayAutoscaler class with Ray Serve deployment"""
    import asyncio
    import time

    from hpc_worker.ray_cluster_manager import RayClusterManager

    print("===== Testing Ray Autoscaler class =====", end="\n\n")

    @ray.remote(num_cpus=1, num_gpus=1)
    def test_remote():
        import time

        time.sleep(10)
        return "Successfully run a task on the worker node!"

    async def test_autoscaler():
        try:
            # Create RayClusterManager with test configuration
            cluster_manager = RayClusterManager()
            # Start Ray cluster
            start_result = cluster_manager.start_cluster()
            if not start_result["success"]:
                return

            # Create and start the autoscaler with shorter thresholds for quicker testing
            autoscaler = RayAutoscaler(
                cluster_manager,
                # Use shorter times for faster testing
                max_workers=3,
                metrics_interval_seconds=1,
                scale_down_threshold_seconds=15,  # 15 seconds idle before scale down
                scale_up_cooldown_seconds=10,  # 10 seconds between scale up
                scale_down_cooldown_seconds=10,  # 10 seconds between scale down
                node_grace_period_seconds=10,
            )
            await autoscaler.start()
            # Test autoscaler status
            autoscaler.get_autoscaler_status()

            # Submit some test tasks
            obj_refs = [test_remote.remote() for _ in range(5)]

            # Observe autoscaler status for a while
            for _ in range(20):
                await asyncio.sleep(3)
                autoscaler.get_autoscaler_status()

            results = await asyncio.gather(*obj_refs)
            print(results)

        except Exception as e:
            print(f"An error occurred: {str(e)}")
        finally:
            await autoscaler.stop()
            cluster_manager.shutdown_cluster()

    asyncio.run(test_autoscaler())
