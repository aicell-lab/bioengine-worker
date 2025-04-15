import asyncio
import logging
import time
from typing import Optional

import numpy as np
import ray
from ray.util.state import list_actors, list_tasks

from bioengine_worker.ray_cluster_manager import RayClusterManager
from bioengine_worker.utils.logger import create_logger


class RayAutoscaler:
    """Autoscaler for Ray clusters in HPC environments.

    This class monitors Ray cluster resource usage and auto-scales
    worker nodes based on configured policies.
    """

    def __init__(
        self,
        cluster_manager: RayClusterManager,
        # Default resource parameters
        default_num_gpus: int = 1,
        default_num_cpus: int = 8,
        default_mem_per_cpu: int = 16,
        default_time_limit: str = "4:00:00",
        # Autoscaling configuration parameters
        min_workers: int = 0,
        max_workers: int = 4,
        metrics_interval_seconds: int = 60,  # Higher value to reduce monitoring overhead
        gpu_idle_threshold: float = 0.05,
        cpu_idle_threshold: float = 0.1,
        scale_down_threshold_seconds: int = 300,  # 5 minutes of idleness before scale down
        scale_up_cooldown_seconds: int = 120,  # 2 minutes between scale ups
        scale_down_cooldown_seconds: int = 600,  # 10 minutes between scale downs
        node_grace_period_seconds: int = 600,  # 10 minutes grace period for new nodes
        # Logger
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize the Ray autoscaler.

        Args:
            default_num_gpus: Default number of GPUs per worker
            default_num_cpus: Default number of CPUs per worker
            default_mem_per_cpu: Default memory per CPU in GB
            default_time_limit: Default time limit for workers in HH:MM:SS format
            min_workers: Minimum number of worker nodes to maintain
            max_workers: Maximum number of worker nodes to allow
            metrics_interval_seconds: Interval for collecting metrics
            gpu_idle_threshold: Node is considered idle below this GPU utilization
            cpu_idle_threshold: Node is considered idle below this CPU utilization
            scale_down_threshold_seconds: Idle time under threshold before scaling down
            scale_up_cooldown_seconds: Minimum time between scaling up operations
            scale_down_cooldown_seconds: Minimum time between scaling down operations
            node_grace_period_seconds: Give new nodes time before considering for scale-down
            logger: Optional logger instance
        """
        # Store ray manager reference
        self.cluster_manager = cluster_manager
        if not ray.is_initialized():
            raise RuntimeError(
                "Ray is not initialized. Please start the Ray cluster first."
            )

        # Store configuration
        self.autoscale_config = {
            "default_num_gpus": default_num_gpus,
            "default_num_cpus": default_num_cpus,
            "default_mem_per_cpu": default_mem_per_cpu,
            "default_time_limit": default_time_limit,
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
        self.last_time_collected_metrics = 0
        self.node_metrics = {}  # Track per-node metrics
        self.last_scale_up_time = 0
        self.last_scale_down_time = 0

        # Background task
        self.monitoring_task = None
        self.is_running = False

        # Set up logging
        self.logger = logger or create_logger("RayAutoscaler")

    @property
    def pending_tasks(self) -> list:
        """Get pending tasks that need resources."""
        if ray.is_initialized():
            pending_tasks = list_tasks(
                address=self.cluster_manager.head_node_address,
                filters=[("state", "=", "PENDING_NODE_ASSIGNMENT")],
            )
            if pending_tasks:
                self.logger.debug(f"Found {len(pending_tasks)} pending task(s)")

            return pending_tasks
        else:
            self.logger.warning("Ray cluster is not running")
            return []

    @property
    def pending_actors(self) -> list:
        """Get pending actors that need resources."""
        if ray.is_initialized():
            pending_actors = list_actors(
                address=self.cluster_manager.head_node_address,
                filters=[("state", "=", "PENDING_CREATION")],
            )
            if pending_actors:
                self.logger.debug(f"Found {len(pending_actors)} pending actor(s)")
            return pending_actors
        else:
            self.logger.warning("Ray cluster is not running")
            return []

    @property
    def n_worker_jobs(self) -> int:
        """Get number of SLURM workers in the cluster (configuring, pending or running)."""
        return len(self.cluster_manager.slurm_actor.get_jobs())

    def _collect_metrics(self) -> tuple:
        """Collect resource utilization metrics from worker nodes using cluster status."""

        node_metrics = self.node_metrics.copy()

        # Get current cluster status
        current_time = time.time()
        cluster_status = self.cluster_manager.get_status()
        active_nodes = cluster_status["worker_nodes"]["Alive"]
        dead_nodes = cluster_status["worker_nodes"]["Dead"]

        active_worker_ids = {node["WorkerID"] for node in active_nodes}
        dead_worker_ids = {node["WorkerID"] for node in dead_nodes}

        # Update metrics for active nodes
        for node in active_nodes:
            worker_id = node["WorkerID"]

            # Initialize metrics for new nodes
            if worker_id not in node_metrics:
                self.logger.debug(f"Adding new worker '{worker_id}' to metrics")
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

            node_metrics[worker_id]["timestamps"].append(current_time)
            node_metrics[worker_id]["gpu_util"].append(node["GPU Utilization"])
            node_metrics[worker_id]["cpu_util"].append(node["CPU Utilization"])
            node_metrics[worker_id]["memory_util"].append(node["Memory Utilization"])

            # Update active status based on both GPU and CPU utilization
            if (
                node["GPU Utilization"] >= self.autoscale_config["gpu_idle_threshold"]
                or node["CPU Utilization"]
                >= self.autoscale_config["cpu_idle_threshold"]
            ):
                node_metrics[worker_id]["last_active_time"] = current_time

            # Limit history size
            max_history = 100
            if len(node_metrics[worker_id]["timestamps"]) > max_history:
                for key in ["timestamps", "gpu_util", "cpu_util", "memory_util"]:
                    node_metrics[worker_id][key] = node_metrics[worker_id][key][
                        -max_history:
                    ]

        # Remove metrics for nodes that are no longer in cluster
        node_metrics_keys = list(node_metrics.keys())
        for worker_id in node_metrics_keys:
            if worker_id not in active_worker_ids:
                node_metrics.pop(worker_id, None)

        for worker_id in node_metrics.keys():
            self.logger.debug(
                f"Worker '{worker_id}' has {len(node_metrics[worker_id]['timestamps'])} metric(s)"
            )

        assert len(active_worker_ids) == len(
            node_metrics
        ), "Mismatch between active workers and metrics"

        return active_worker_ids, dead_worker_ids, node_metrics

    def _scale_up(
        self, num_gpus: int, num_cpus: int, mem_per_cpu: int, time_limit: str
    ) -> None:
        """Scale up the cluster by adding a new worker node."""
        # TODO: Check if this is blocking
        _ = self.cluster_manager.add_worker(
            num_gpus=num_gpus,
            num_cpus=num_cpus,
            mem_per_cpu=mem_per_cpu,
            time_limit=time_limit,
        )
        self.last_scale_up_time = time.time()

    def _make_scaling_decisions(self, node_metrics: dict) -> None:
        """Make scaling decisions based on resource utilization and pending tasks."""

        current_time = time.time()
        pending_tasks = self.pending_tasks
        pending_actors = self.pending_actors
        num_pending_tasks = len(pending_tasks) + len(pending_actors)
        n_worker_jobs = self.n_worker_jobs

        # If at least one task needs resources, check scale up, otherwise check scale down
        if num_pending_tasks > 0:

            # SCALE UP LOGIC
            can_scale_up = (
                # Cooldown period has passed
                (current_time - self.last_scale_up_time)
                > self.autoscale_config["scale_up_cooldown"]
                # Not at max worker limit
                and n_worker_jobs < self.autoscale_config["max_workers"]
            )
            if not can_scale_up:
                return

            # Check if any pending task needs more resources than default
            num_gpus = self.autoscale_config["default_num_gpus"]
            num_cpus = self.autoscale_config["default_num_cpus"]

            # TODO: try to get resources of task / actor
            # for task in pending_tasks:
            #     task_req = task.get("required_resources", {}) or {}
            #     if task_req.get("GPU", 0) > num_gpus:
            #         num_gpus = max(num_gpus, int(task_req.get("GPU", 1)))
            #     if task_req.get("CPU", 0) > num_cpus:
            #         num_cpus = max(num_cpus, int(task_req.get("CPU", 4)))

            self.logger.info(
                f"Scaling up with {num_gpus} GPU(s) and {num_cpus} CPU(s) due to {num_pending_tasks} pending task(s)"
            )
            self._scale_up(
                num_gpus=num_gpus,
                num_cpus=num_cpus,
                mem_per_cpu=self.autoscale_config["default_mem_per_cpu"],
                time_limit=self.autoscale_config["default_time_limit"],
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
                    f"Scaling down worker '{longest_idle_worker}' due to inactivity for {longest_idle_time:.1f}s"
                )
                self.cluster_manager.remove_worker(longest_idle_worker)
                self.last_scale_down_time = current_time

    def _cleanup_dead_nodes(self, worker_ids: set) -> None:
        if worker_ids:
            self.logger.info(
                f"Cleaning up {len(worker_ids)} dead worker(s) from cluster: {worker_ids}"
            )
            for worker_id in worker_ids:
                self.cluster_manager.remove_worker(worker_id)

    async def _monitoring_loop(self, max_consecutive_errors: int = 3) -> None:
        """Main monitoring loop for the autoscaler.

        Args:
            max_consecutive_errors: Maximum number of consecutive errors before stopping
        """
        consecutive_errors = 0

        while self.is_running:
            await asyncio.sleep(1)
            if (
                time.time() - self.last_time_collected_metrics
                < self.autoscale_config["metrics_interval"]
            ):
                continue

            try:
                # Check if Ray is still initialized
                if not ray.is_initialized():
                    self.logger.error("Ray is not initialized, stopping autoscaler")
                    break

                # Collect and analyze metrics
                self.logger.debug("Starting autoscaler monitoring iteration")
                _, dead_worker_ids, node_metrics = self._collect_metrics()
                self._make_scaling_decisions(node_metrics)
                self._cleanup_dead_nodes(dead_worker_ids)

                # Update node metrics
                self.node_metrics = node_metrics
                consecutive_errors = 0  # Reset error counter on success

            except asyncio.CancelledError:
                self.logger.info("Autoscaler monitoring loop cancelled")
                break

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                consecutive_errors += 1

                if consecutive_errors >= max_consecutive_errors:
                    self.logger.error(
                        f"Stopping monitoring loop after {consecutive_errors} consecutive errors"
                    )
                    raise e
                
            # Update last metrics collection time
            self.last_time_collected_metrics = time.time()


        # Ensure autoscaler stops if we break from the loop
        if self.is_running:
            asyncio.create_task(self.stop())

    async def start(self):
        """Start the autoscaler monitoring task."""
        try:
            if self.is_running:
                self.logger.info("Autoscaler is already running")

            if not ray.is_initialized():
                raise RuntimeError("Ray cluster is not running")

            self.logger.info(
                f"Starting Ray autoscaler with config {self.autoscale_config}"
            )
            self.is_running = True

            # Create monitoring task
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())

        except Exception as e:
            self.logger.error(f"Failed to start autoscaler: {e}")
            self.is_running = False
            self.monitoring_task = None
            raise e

    async def stop(self):
        """Stop the autoscaler monitoring task."""
        if not self.is_running:
            return

        self.logger.info("Stopping Ray autoscaler")
        self.is_running = False

        try:
            # Clear statistics
            self.node_metrics.clear()

            # Cancel monitoring task with timeout
            if self.monitoring_task:
                self.monitoring_task.cancel()
                try:
                    await asyncio.wait_for(self.monitoring_task, timeout=5.0)
                except asyncio.TimeoutError:
                    self.logger.warning("Timeout waiting for monitoring task to cancel")
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    self.logger.error(f"Error while stopping monitoring task: {e}")
                finally:
                    self.monitoring_task = None

        except Exception as e:
            self.logger.error(f"Error during autoscaler shutdown: {e}")
            # Ensure monitoring_task is cleared even if there's an error
            self.monitoring_task = None
            raise e

    async def shutdown_cluster(self, grace_period: int = 30) -> None:
        """Shut down the Ray cluster.

        Args:
            grace_period: Time in seconds to wait for tasks to finish before shutting down
        """
        await self.stop()
        self.cluster_manager.shutdown_cluster(grace_period=grace_period)

    async def notify(self, delay_s: int = 3) -> None:
        """
        Notify the autoscaler of a change in cluster state.
        This method is called when the cluster state changes, such as when a new task is submitted or a node is added/removed.
        It can be used to trigger scaling decisions or other actions based on the current state of the cluster.
        """
        self.logger.info("Notifying autoscaler of cluster state change")
        self.last_time_collected_metrics = time.time() - self.autoscale_config["metrics_interval"] + delay_s

    async def get_status(self, history_len: int = 5) -> dict:
        """Get the current status of the autoscaler.

        Args:
            history_len: Number of recent data points to use for averaging. Must be > 0.

        Returns:
            Dict containing cluster status, node metrics and scaling information
        """
        try:
            current_time = time.time()
            history_len = max(1, min(history_len, 100))  # Ensure valid history length

            # Collect metrics
            active_worker_ids, _, node_metrics = self._collect_metrics()
            pending_workers = max(self.n_worker_jobs - len(active_worker_ids), 0)

            # Initialize metrics summary
            recent_metrics = {
                "worker_count": len(active_worker_ids) + pending_workers,
                "active_workers": 0,
                "idle_workers": 0,
                "pending_workers": pending_workers,
                "pending_tasks": len(self.pending_tasks) + len(self.pending_actors),
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
                    recent_memory_util = float(
                        np.mean(metrics["memory_util"][recent_slice])
                    )

                    total_metrics["cpu"] += recent_cpu_util
                    total_metrics["gpu"] += recent_gpu_util
                    total_metrics["memory"] += recent_memory_util

                    # Determine if node is idle based on current utilization
                    cpu_is_idle = (
                        metrics["cpu_util"][-1]
                        < self.autoscale_config["cpu_idle_threshold"]
                    )
                    gpu_is_idle = (
                        metrics["gpu_util"][-1]
                        < self.autoscale_config["gpu_idle_threshold"]
                    )

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
                        "in_grace_period": node_age
                        < self.autoscale_config["node_grace_period"],
                        "idle_time_seconds": int(idle_time),
                        "cpu_is_idle": cpu_is_idle,
                        "gpu_is_idle": gpu_is_idle,
                        "average_cpu_utilization": round(recent_cpu_util, 3),
                        "average_gpu_utilization": round(recent_gpu_util, 3),
                        "average_memory_utilization": round(recent_memory_util, 3),
                    }

                # Calculate cluster-wide averages safely
                recent_metrics["average_cpu"] = round(
                    total_metrics["cpu"] / len(node_metrics), 3
                )
                recent_metrics["average_gpu"] = round(
                    total_metrics["gpu"] / len(node_metrics), 3
                )
                recent_metrics["average_memory"] = round(
                    total_metrics["memory"] / len(node_metrics), 3
                )

            # Calculate cooldown times
            scale_up_cooldown = max(
                0,
                self.autoscale_config["scale_up_cooldown"]
                - (current_time - self.last_scale_up_time),
            )
            scale_down_cooldown = max(
                0,
                self.autoscale_config["scale_down_cooldown"]
                - (current_time - self.last_scale_down_time),
            )

            self.logger.info(
                f"Autoscaler status: {recent_metrics['worker_count']} worker(s) "
                f"({recent_metrics['active_workers']} active, {recent_metrics['idle_workers']} idle, "
                f"{recent_metrics['pending_workers']} pending), {recent_metrics['pending_tasks']} pending task(s), "
                f"{scale_up_cooldown:.1f}s until possible scale up, "
                f"{scale_down_cooldown:.1f}s until possible scale down"
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
        except Exception as e:
            self.logger.error(f"Error getting autoscaler status {e}")
            raise e


if __name__ == "__main__":
    """Test the RayAutoscaler class with Ray Serve deployment"""
    import asyncio
    import os
    import time
    from pathlib import Path

    print("\n===== Testing Ray Autoscaler class =====\n")

    @ray.remote(num_cpus=1, num_gpus=1)
    def test_remote():
        import time

        import torch

        time.sleep(3)
        return "Successfully run a task on the worker node!"

    async def test_autoscaler():
        try:
            cluster_manager = RayClusterManager(
                temp_dir=str(Path(__file__).parent.parent / "ray_sessions"),
                data_dir=str(Path(__file__).parent.parent / "data"),
                container_image=str(Path(__file__).parent.parent / "bioengine_worker_0.1.2.sif"),
            )
            cluster_manager.start_cluster(force_clean_up=True)

            # Create and start the autoscaler with shorter thresholds for quicker testing
            autoscaler = RayAutoscaler(
                cluster_manager=cluster_manager,
                # Use shorter times for faster testing
                default_time_limit="00:10:00",
                max_workers=3,
                metrics_interval_seconds=3,
                scale_down_threshold_seconds=15,  # 15 seconds idle before scale down
                scale_up_cooldown_seconds=10,  # 10 seconds between scale up
                scale_down_cooldown_seconds=10,  # 10 seconds between scale down
                node_grace_period_seconds=10,
            )
            autoscaler.cluster_manager.logger.setLevel(logging.DEBUG)
            autoscaler.logger.setLevel(logging.DEBUG)

            # Start autoscaler
            await autoscaler.start()

            # Test autoscaler status
            await autoscaler.get_status()

            # Submit some test tasks
            obj_refs = [test_remote.remote() for _ in range(5)]

            # Observe autoscaler status for a while
            for _ in range(20):
                await asyncio.sleep(3)
                await autoscaler.get_status()

            results = await asyncio.gather(*obj_refs)
            print(results)

        except Exception as e:
            print(f"An error occurred {e}")
            raise e
        finally:
            await autoscaler.shutdown_cluster()

    asyncio.run(test_autoscaler())
