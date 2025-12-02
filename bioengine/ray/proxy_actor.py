import re
from dataclasses import asdict
from typing import Dict, List, Optional, Union

import ray
from ray._private.state import GlobalState
from ray._raylet import GcsClientOptions
from ray.util.state import StateApiClient, get_log
from ray.util.state.common import (
    DEFAULT_LIMIT,
    DEFAULT_RPC_TIMEOUT,
    ListApiOptions,
    StateResource,
)
from ray.util.state.exception import RayStateApiException


@ray.remote(
    num_cpus=0,
    resources={"node:__internal_head__": 0.001},
    max_restarts=-1,  # Allow unlimited restarts
)
class BioEngineProxyActor:
    """
    Ray actor for monitoring cluster resources and deployment state.

    This actor provides a centralized way to query the Ray cluster for resource
    availability, node information, and deployment status. It acts as a "cluster
    dashboard" that other components can use to make informed decisions about
    resource allocation and scaling.

    Key Features:
    • Real-time cluster resource monitoring (CPU, GPU, memory)
    • Node-by-node resource breakdown with IP addresses
    • Pending resource tracking (jobs, actors, tasks)
    • Deployment replica status monitoring
    • SLURM job integration for HPC environments

    The actor runs on the head node with minimal resource requirements and
    automatically restarts if it fails.
    """

    def __init__(
        self, exclude_head_node: bool = False, check_pending_resources: bool = False
    ):
        """
        Initialize the proxy actor with Ray GCS connection and monitoring options.

        Sets up connections to Ray's Global Control Service (GCS) for querying
        cluster state and configures monitoring behavior.

        Args:
            exclude_head_node: Skip head node in resource calculations (useful for worker-only metrics)
            check_pending_resources: Include pending jobs/actors/tasks in cluster state reports
        """
        # Get the GCS address using ray._private.worker
        gcs_address = ray._private.worker.global_worker.gcs_client.address

        # Create GCS client options
        gcs_options = GcsClientOptions.create(
            gcs_address,
            None,
            allow_cluster_id_nil=True,
            fetch_cluster_id_if_nil=False,
        )

        # Initialize global state for cluster resources
        self.global_state = GlobalState()
        self.global_state._initialize_global_state(gcs_options)

        # Force initialization of the global state accessor
        self.global_state._check_connected()

        # Initialize the state API client for querying states
        self.state_api_client = StateApiClient(gcs_address)
        self.exclude_head_node = exclude_head_node
        self.check_pending_resources = check_pending_resources

        print("ClusterState initialized with GCS address:", gcs_address)

    def _get_pending_jobs(self) -> int:
        """Get list of jobs waiting to start in the cluster."""
        pending_jobs = self.state_api_client.list(
            resource=StateResource.JOBS,
            options=ListApiOptions(
                limit=DEFAULT_LIMIT,
                timeout=DEFAULT_RPC_TIMEOUT,
                filters=[("status", "=", "PENDING")],
                detail=True,
                explain=False,
            ),
            raise_on_missing_output=True,
        )
        return [asdict(job) for job in pending_jobs]

    def _get_pending_actors(self) -> int:
        """Get list of actors waiting to be created in the cluster."""
        pending_actors = self.state_api_client.list(
            resource=StateResource.ACTORS,
            options=ListApiOptions(
                limit=DEFAULT_LIMIT,
                timeout=DEFAULT_RPC_TIMEOUT,
                filters=[("state", "=", "PENDING_CREATION")],
                detail=True,
                explain=False,
            ),
            raise_on_missing_output=True,
        )
        return [asdict(actor) for actor in pending_actors]

    def _get_pending_tasks(self) -> int:
        """Get list of tasks waiting for node assignment in the cluster."""
        pending_tasks = self.state_api_client.list(
            resource=StateResource.TASKS,
            options=ListApiOptions(
                limit=DEFAULT_LIMIT,
                timeout=DEFAULT_RPC_TIMEOUT,
                filters=[
                    ("state", "=", "PENDING_NODE_ASSIGNMENT"),
                    # Avoid duplicates with `_get_pending_actors`
                    ("type", "!=", "ACTOR_CREATION_TASK"),
                ],
                detail=True,
                explain=False,
            ),
            raise_on_missing_output=True,
        )
        return [asdict(task) for task in pending_tasks]

    def _get_node_ip(self, resources: Dict[str, float]) -> Optional[str]:
        """Extract the IP address from a node's resource dictionary."""
        ip_pattern = r"^node:(\d+\.\d+\.\d+\.\d+)$"
        for resource_name in resources:
            if resource_name.startswith("node:"):
                # Use regex to extract IP address from node resource name
                # (there might be multiple resources starting with "node:")
                match = re.match(ip_pattern, resource_name)
                if match:
                    return match.group(1)

    def _get_accelerator_type(self, resources: Dict[str, float]) -> str:
        """Extract the GPU/accelerator type from a node's resource dictionary."""
        for resource_name in resources:
            if resource_name.startswith("accelerator_type:"):
                return resource_name.split(":")[-1]

    def _get_slurm_job_id(self, resources: Dict[str, float]) -> str:
        """Extract the SLURM job ID from a node's resource dictionary."""
        for resource_name in resources:
            if resource_name.startswith("slurm_job_id:"):
                return resource_name.split(":")[-1]

    def get_cluster_state(self) -> Dict[str, Union[float, int, str]]:
        """
        Get comprehensive cluster resource information including per-node breakdown.

        Returns a detailed snapshot of cluster resources showing both total capacity
        and currently available resources across all alive nodes. Useful for
        capacity planning and resource allocation decisions.

        Returns:
            Dictionary with 'cluster' totals and 'nodes' breakdown containing:
            - CPU/GPU cores and availability
            - Memory and object store memory (in bytes)
            - Node IP addresses and accelerator types
            - SLURM job IDs (if applicable)
            - Pending resources (if enabled)
        """
        total_resources_per_node = self.global_state.total_resources_per_node()
        available_resources_per_node = self.global_state.available_resources_per_node()

        cluster_state = {
            "cluster": {
                "total_cpu": 0,
                "available_cpu": 0,
                "total_gpu": 0,
                "available_gpu": 0,
                "total_memory": 0,
                "available_memory": 0,
                "total_object_store_memory": 0,
                "available_object_store_memory": 0,
            },
            "nodes": {},
        }

        # Process each node's resources
        for node_id, total_resources in total_resources_per_node.items():
            if (
                self.exclude_head_node
                and "node:__internal_head__" in total_resources.keys()
            ):
                continue

            available_resources = available_resources_per_node.get(node_id, {})

            # Add per-node resources to the cluster state
            cluster_state["nodes"][node_id] = {
                "node_ip": self._get_node_ip(total_resources),
                "total_cpu": total_resources.get("CPU", 0),
                "available_cpu": available_resources.get("CPU", 0),
                "total_gpu": total_resources.get("GPU", 0),
                "available_gpu": available_resources.get("GPU", 0),
                "total_memory": total_resources.get("memory", 0),  # in bytes
                "available_memory": available_resources.get("memory", 0),  # in bytes
                "total_object_store_memory": total_resources.get(
                    "object_store_memory", 0  # in bytes
                ),
                "available_object_store_memory": available_resources.get(
                    "object_store_memory", 0  # in bytes
                ),
                "accelerator_type": self._get_accelerator_type(total_resources),
                "slurm_job_id": self._get_slurm_job_id(total_resources),
            }

            # Accumulate cluster resources
            for resource_name in cluster_state["cluster"].keys():
                node_value = cluster_state["nodes"][node_id][resource_name]
                cluster_state["cluster"][resource_name] += node_value

        if self.check_pending_resources:
            # TODO: Don't count task/actor/job in runtime creation
            # Check if there are any runtime environment creations in progress
            cluster_state["cluster"]["pending_resources"] = {
                "actors": self._get_pending_actors(),
                "jobs": self._get_pending_jobs(),
                "tasks": self._get_pending_tasks(),
            }
            cluster_state["cluster"]["pending_resources"]["total"] = sum(
                len(pending_resources)
                for pending_resources in cluster_state["cluster"][
                    "pending_resources"
                ].values()
            )

        return cluster_state

    def get_deployment_replica(self, app_name: str, deployment_name: str) -> List[str]:
        """
        Get the list of active replica IDs for a specific Ray Serve deployment.

        Useful for monitoring deployment health and scaling status.

        Args:
            app_name: Name of the Ray Serve application
            deployment_name: Name of the specific deployment within the app

        Returns:
            List of replica ID strings for replicas currently in ALIVE state
        """
        class_name = f"ServeReplica:{app_name}:{deployment_name}"
        replica_actors = self.state_api_client.list(
            resource=StateResource.ACTORS,
            options=ListApiOptions(
                limit=DEFAULT_LIMIT,
                timeout=DEFAULT_RPC_TIMEOUT,
                filters=[("class_name", "=", class_name), ("state", "=", "ALIVE")],
                detail=False,
                explain=False,
            ),
            raise_on_missing_output=True,
        )
        replica = {
            actor.name.split("#")[-1]: actor.actor_id for actor in replica_actors
        }
        return replica

    def get_deployment_logs(
        self,
        app_name: str,
        deployment_name: str,
        tail: int = 1000,
    ) -> str:
        """
        Get logs for a specific Ray Serve deployment.

        Retrieves the stdout or stderr logs from a deployment replica actor.
        This is useful for debugging deployment issues and monitoring
        application behavior.

        Args:
            app_name: Name of the Ray Serve application
            deployment_name: Name of the specific deployment within the app
            tail: Number of lines to retrieve from the end of the log.
                  Use -1 to get the entire log. Default is -1.

        Returns:
            String containing the log content, or an error message if the
            replica is not found.
        """
        replica = self.get_deployment_replica(app_name, deployment_name)
        if not replica:
            return f"No active replicas found for deployment '{deployment_name}' in app '{app_name}'."

        deployment_logs = {}
        for replica_id, actor_id in replica.items():
            deployment_logs[replica_id] = {"stdout": [], "stderr": []}

            try:
                stdout = next(
                    get_log(
                        actor_id=actor_id,
                        tail=tail,
                        timeout=DEFAULT_RPC_TIMEOUT,
                        suffix="out",
                    )
                )
                deployment_logs[replica_id]["stdout"] = stdout.splitlines()
            except RayStateApiException as e:
                deployment_logs[replica_id]["stdout"] = [
                    f"Error retrieving stdout logs: {str(e)}"
                ]

            try:
                stderr = next(
                    get_log(
                        actor_id=actor_id,
                        tail=tail,
                        timeout=DEFAULT_RPC_TIMEOUT,
                        suffix="err",
                    )
                )
                deployment_logs[replica_id]["stderr"] = stderr.splitlines()
            except RayStateApiException as e:
                deployment_logs[replica_id]["stderr"] = [
                    f"Error retrieving stderr logs: {str(e)}"
                ]

        return deployment_logs
