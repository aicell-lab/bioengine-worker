import re
from typing import Dict, Optional, Union

import ray
from ray._private.state import GlobalState
from ray._raylet import GcsClientOptions
from ray.util.state import StateApiClient
from ray.util.state.common import (
    DEFAULT_LIMIT,
    DEFAULT_RPC_TIMEOUT,
    ListApiOptions,
    StateResource,
)


@ray.remote(num_cpus=0, resources={"node:__internal_head__": 0.001})
class ClusterState:
    def __init__(
        self, exclude_head_node: bool = False, check_pending_resources: bool = False
    ):
        # Get the GCS address using ray._private.worker
        gcs_address = ray._private.worker.global_worker.gcs_client.address
        print(f"Using GCS address: {gcs_address}")

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

    def _get_pending_actors(self) -> int:
        """Get the number of pending actors in the cluster."""
        return self.state_api_client.list(
            resource=StateResource.ACTORS,
            options=ListApiOptions(
                limit=DEFAULT_LIMIT,
                timeout=DEFAULT_RPC_TIMEOUT,
                filters=[("state", "=", "PENDING_CREATION")],
                detail=False,
                explain=False,
            ),
            raise_on_missing_output=True,
        )

    def _get_pending_tasks(self) -> int:
        """Get the number of pending tasks in the cluster."""
        return self.state_api_client.list(
            resource=StateResource.TASKS,
            options=ListApiOptions(
                limit=DEFAULT_LIMIT,
                timeout=DEFAULT_RPC_TIMEOUT,
                filters=[("state", "=", "PENDING_NODE_ASSIGNMENT")],
                detail=False,
                explain=False,
            ),
            raise_on_missing_output=True,
        )

    def _get_pending_jobs(self) -> int:
        """Get the number of pending jobs in the cluster."""
        return self.state_api_client.list(
            resource=StateResource.JOBS,
            options=ListApiOptions(
                limit=DEFAULT_LIMIT,
                timeout=DEFAULT_RPC_TIMEOUT,
                filters=[("status", "=", "PENDING")],
                detail=False,
                explain=False,
            ),
            raise_on_missing_output=True,
        )

    def _get_node_ip(self, resources: Dict[str, float]) -> Optional[str]:
        """Get the node IP address from the resources dictionary."""
        ip_pattern = r"^node:(\d+\.\d+\.\d+\.\d+)$"
        for resource_name in resources:
            if resource_name.startswith("node:"):
                # Use regex to extract IP address from node resource name
                # (there might be multiple resources starting with "node:")
                match = re.match(ip_pattern, resource_name)
                if match:
                    return match.group(1)

    def _get_accelerator_type(self, resources: Dict[str, float]) -> str:
        """Get the type of accelerator used in the cluster."""
        for resource_name in resources:
            if resource_name.startswith("accelerator_type:"):
                return resource_name.split(":")[-1]

    def _get_slurm_job_id(self, resources: Dict[str, float]) -> str:
        """Get the SLURM job ID from the resources dictionary."""
        for resource_name in resources:
            if resource_name.startswith("slurm_job_id:"):
                return resource_name.split(":")[-1]

    def get_state(self) -> Dict[str, Union[float, int, str]]:
        """
        Get comprehensive cluster state including total and available resources per node with state 'ALIVE'.

        Returns:
            Dict[str, Union[float, int, str]]: A dictionary containing the cluster state
            with total and available resources per node, excluding the head node if specified.
            The dictionary includes CPU, GPU, memory, object store memory, and accelerator type.
            If `check_pending_resources` is True, it also includes the count of pending resources.
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
                "total_memory": total_resources.get("memory", 0),
                "available_memory": available_resources.get("memory", 0),
                "total_object_store_memory": total_resources.get(
                    "object_store_memory", 0
                ),
                "available_object_store_memory": available_resources.get(
                    "object_store_memory", 0
                ),
                "accelerator_type": self._get_accelerator_type(total_resources),
                "slurm_job_id": self._get_slurm_job_id(total_resources),
            }

            # Accumulate cluster resources
            for resource_name in cluster_state["cluster"].keys():
                node_value = cluster_state["nodes"][node_id][resource_name]
                cluster_state["cluster"][resource_name] += node_value

        if self.check_pending_resources:
            # TODO: Don't count task/actor/job runtime creation
            # Check if there are any runtime environment creations in progress
            cluster_state["cluster"]["n_pending_resources"] = (
                len(self._get_pending_actors())
                + len(self._get_pending_tasks())
                + len(self._get_pending_jobs())
            )

        return cluster_state
