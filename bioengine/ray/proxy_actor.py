import re
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple
import logging

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

logger = logging.getLogger("ray")
date_format = "%Y-%m-%d %H:%M:%S %Z"


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
        self.gcs_address = ray._private.worker.global_worker.gcs_client.address

        # Create GCS client options
        gcs_options = GcsClientOptions.create(
            self.gcs_address,
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
        self.state_api_client = StateApiClient(self.gcs_address)
        self.exclude_head_node = exclude_head_node
        self.check_pending_resources = check_pending_resources

        # Initialize per application actor ID tracking
        # Structure: {app_id: {deployment_name: {replica_id: (actor_id, timestamp, timezone)}}}
        self.application_replicas: Dict[
            str, Dict[str, Dict[str, Tuple[str, float, str]]]
        ] = {}

        logger.info(f"ClusterState initialized with GCS address: {self.gcs_address}")

    def _get_pending_jobs(self) -> List[Dict[str, Any]]:
        """Get list of jobs waiting to start in the cluster.

        Returns:
            List of dictionaries containing job information for pending jobs.
        """
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

    def _get_pending_actors(self) -> List[Dict[str, Any]]:
        """Get list of actors waiting to be created in the cluster.

        Returns:
            List of dictionaries containing actor information for pending actors.
        """
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

    def _get_pending_tasks(self) -> List[Dict[str, Any]]:
        """Get list of tasks waiting for node assignment in the cluster.

        Returns:
            List of dictionaries containing task information for pending tasks.
            Excludes actor creation tasks to avoid duplicates with _get_pending_actors.
        """
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
        """Extract the IP address from a node's resource dictionary.

        Args:
            resources: Node resource dictionary with resource names as keys.

        Returns:
            IP address string if found (e.g., "192.168.1.1"), None otherwise.
        """
        ip_pattern = r"^node:(\d+\.\d+\.\d+\.\d+)$"
        for resource_name in resources:
            if resource_name.startswith("node:"):
                # Use regex to extract IP address from node resource name
                # (there might be multiple resources starting with "node:")
                match = re.match(ip_pattern, resource_name)
                if match:
                    return match.group(1)

    def _get_accelerator_type(self, resources: Dict[str, float]) -> Optional[str]:
        """Extract the GPU/accelerator type from a node's resource dictionary.

        Args:
            resources: Node resource dictionary with resource names as keys.

        Returns:
            Accelerator type string if found (e.g., "NVIDIA-A100"), None otherwise.
        """
        for resource_name in resources:
            if resource_name.startswith("accelerator_type:"):
                return resource_name.split(":")[-1]

    def _get_slurm_job_id(self, resources: Dict[str, float]) -> Optional[str]:
        """Extract the SLURM job ID from a node's resource dictionary.

        Args:
            resources: Node resource dictionary with resource names as keys.

        Returns:
            SLURM job ID string if found, None otherwise.
        """
        for resource_name in resources:
            if resource_name.startswith("slurm_job_id:"):
                return resource_name.split(":")[-1]

    def get_cluster_state(self) -> Dict[str, Any]:
        """
        Get comprehensive cluster resource information including per-node breakdown.

        Returns a detailed snapshot of cluster resources showing both total capacity
        and currently available resources across all alive nodes. Useful for
        capacity planning and resource allocation decisions.

        Returns:
            Dictionary with the following structure:
            {
                "cluster": {
                    "total_cpu": float,
                    "available_cpu": float,
                    "total_gpu": float,
                    "available_gpu": float,
                    "total_memory": float,  # in bytes
                    "available_memory": float,  # in bytes
                    "total_object_store_memory": float,  # in bytes
                    "available_object_store_memory": float,  # in bytes
                    "pending_resources": {  # if check_pending_resources=True
                        "actors": List[Dict],
                        "jobs": List[Dict],
                        "tasks": List[Dict],
                        "total": int
                    }
                },
                "nodes": {
                    node_id: {
                        "node_ip": Optional[str],
                        "total_cpu": float,
                        "available_cpu": float,
                        "total_gpu": float,
                        "available_gpu": float,
                        "total_memory": float,
                        "available_memory": float,
                        "total_object_store_memory": float,
                        "available_object_store_memory": float,
                        "accelerator_type": Optional[str],
                        "slurm_job_id": Optional[str]
                    }
                }
            }
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

    def get_deployment_replicas(
        self, application_id: str, deployment_name: str
    ) -> Dict[str, str]:
        """
        Get the list of active replica IDs for a specific Ray Serve deployment.

        Queries the Ray cluster for all ALIVE replicas of the specified deployment
        and returns their replica IDs mapped to actor IDs. Useful for monitoring
        deployment health and scaling status.

        Args:
            application_id: Name of the Ray Serve application.
            deployment_name: Name of the specific deployment within the app.

        Returns:
            Mapping of replica ID strings to their actor IDs for replicas
            currently in ALIVE state. Example: {"replica_1": "actor_abc123"}.
        """
        class_name = f"ServeReplica:{application_id}:{deployment_name}"
        deployment_actors = self.state_api_client.list(
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
        replica_info = {
            actor.name.split("#")[-1]: actor.actor_id for actor in deployment_actors
        }
        return replica_info

    def register_serve_replica(
        self,
        application_id: str,
        deployment_name: str,
        replica_id: str,
        timezone: str,
    ) -> None:
        """
        Register a Ray Serve replica for tracking and log retrieval.

        Registers a replica's actor ID, timestamp, and timezone for later log retrieval,
        especially useful for accessing logs from replicas that have died.
        Validates that the replica exists in the Ray cluster before registration.

        Args:
            application_id: Name of the Ray Serve application.
            deployment_name: Name of the deployment within the application.
            replica_id: Unique identifier for the replica (e.g., "replica_1").
            timezone: Timezone string for the replica (e.g., "Europe/Stockholm").

        Raises:
            ValueError: If the replica is already registered, not found in the
                       cluster, or if multiple actors with the same name exist.
        """
        registration_time = time.time()
        self.application_replicas.setdefault(application_id, {})
        self.application_replicas[application_id].setdefault(deployment_name, {})

        # Check if the replica is already registered
        if replica_id in self.application_replicas[application_id][deployment_name]:
            actor_id, registration_time, _ = self.application_replicas[application_id][
                deployment_name
            ][replica_id]
            formatted_time = time.strftime(
                date_format, time.localtime(registration_time)
            )
            raise ValueError(
                f"Replica '{replica_id}' already registered for application "
                f"'{application_id}', deployment '{deployment_name}' at {formatted_time}."
            )

        # Query Ray cluster for the actor ID using the naming convention
        actor_name = f"SERVE_REPLICA::{application_id}#{deployment_name}#{replica_id}"
        actor = self.state_api_client.list(
            resource=StateResource.ACTORS,
            options=ListApiOptions(
                limit=DEFAULT_LIMIT,
                timeout=DEFAULT_RPC_TIMEOUT,
                filters=[("name", "=", actor_name)],
                detail=False,
                explain=False,
            ),
            raise_on_missing_output=True,
        )
        if len(actor) == 0:
            raise ValueError(
                f"Actor with name '{actor_name}' not found in Ray cluster."
            )
        if len(actor) > 1:
            raise ValueError(
                f"Multiple actors with name '{actor_name}' found in Ray cluster."
            )
        actor_id = actor[0].actor_id

        # Register the replica with timestamp, actor ID, and timezone
        self.application_replicas[application_id][deployment_name][replica_id] = (
            actor_id,
            registration_time,
            timezone,
        )
        logger.info(
            f"Registered replica '{replica_id}' for application '{application_id}', "
            f"deployment '{deployment_name}' with actor ID '{actor_id}' (replica timezone: {timezone})."
        )

    def clear_application_replicas(self, application_id: str) -> None:
        """
        Clear all registered replicas for a specific application.

        Removes all replica registration data for the given application,
        typically called when an application is deleted or redeployed.

        Args:
            application_id: Name of the Ray Serve application to clear.
        """
        self.application_replicas.pop(application_id, None)
        logger.info(
            f"Cleared all registered replicas for application '{application_id}'."
        )

    def get_actor_logs(
        self,
        actor_id: str,
        tail: int = 100,
        timezone: Optional[str] = None,
        creation_timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Get logs for a specific Ray actor by its actor ID.

        Retrieves both stdout and stderr logs from the specified actor.
        This is useful for debugging actor issues and monitoring behavior.
        Error messages are returned in place of logs if retrieval fails.

        Args:
            actor_id: Ray actor ID to retrieve logs from.
            tail: Number of lines to retrieve from the end of each log.
                  Use -1 to get the entire log. Default is 100.
            timezone: Optional timezone string for the replica.
            creation_timestamp: Optional Unix timestamp when the replica was created.

        Returns:
            Dictionary with log information. Always includes "stdout" and "stderr" keys.
            If timezone is provided, also includes "timezone" key.
            If creation_timestamp is provided, also includes "creation_timestamp" key.
            Example:
            {
                "creation_timestamp": 1234567890.123,  # if provided
                "timezone": "Europe/Stockholm",  # if provided
                "stdout": ["line1", "line2", ...],
                "stderr": ["error1", ...],
            }
        """
        logs = {}

        # Add optional metadata
        if creation_timestamp is not None:
            logs["creation_timestamp"] = creation_timestamp
        if timezone is not None:
            logs["timezone"] = timezone

        try:
            stdout = next(
                get_log(
                    address=self.gcs_address,
                    actor_id=actor_id,
                    tail=tail,
                    timeout=DEFAULT_RPC_TIMEOUT,
                    suffix="out",
                )
            )
            logs["stdout"] = stdout.splitlines()
        except RayStateApiException as e:
            logs["stdout"] = [f"Error retrieving stdout logs: {str(e)}"]

        try:
            stderr = next(
                get_log(
                    address=self.gcs_address,
                    actor_id=actor_id,
                    tail=tail,
                    timeout=DEFAULT_RPC_TIMEOUT,
                    suffix="err",
                )
            )
            logs["stderr"] = stderr.splitlines()
        except RayStateApiException as e:
            logs["stderr"] = [f"Error retrieving stderr logs: {str(e)}"]

        return logs

    def get_deployment_logs(
        self,
        application_id: str,
        deployment_name: str,
        n_previous_replica: int = 0,
        tail: int = 30,
    ) -> Dict[str, Dict[str, List[str]]]:
        """
        Get logs for all replicas of a Ray Serve deployment.

        Retrieves stdout and stderr logs from all active replicas and optionally
        from recently terminated replicas. This provides a comprehensive view of
        deployment behavior across all instances.

        Args:
            application_id: Name of the Ray Serve application.
            deployment_name: Name of the deployment within the application.
            n_previous_replica: Number of most recent dead replicas to include.
                               Set to 0 to only get logs from active replicas.
                               Set to -1 to get logs from all dead replicas.
                               Default is 0.
            tail: Number of lines to retrieve from the end of each log.
                  Use -1 to get the entire log. Default is 30.

        Returns:
            Dictionary mapping replica IDs to their log dictionaries:
            {
                "replica_1": {
                    "stdout": ["line1", "line2", ...],
                    "stderr": ["error1", ...]
                },
                "replica_2": {...}
            }
            Error messages are included in stderr if log retrieval fails.
        """
        logs = {}

        # Get active replicas from Ray Serve
        active_replicas = self.get_deployment_replicas(
            application_id=application_id, deployment_name=deployment_name
        )

        # Logs of active replicas
        for replica_id, actor_id in active_replicas.items():
            try:
                # Get replica metadata for timezone and creation timestamp
                replica_info = self.application_replicas.get(application_id, {}).get(
                    deployment_name, {}
                ).get(replica_id)
                replica_tz = None
                replica_creation = None
                if replica_info:
                    # Unpack: (actor_id, timestamp, timezone)
                    _, replica_creation, replica_tz = replica_info

                actor_logs = self.get_actor_logs(
                    actor_id=actor_id,
                    tail=tail,
                    timezone=replica_tz,
                    creation_timestamp=replica_creation,
                )
                logs[replica_id] = actor_logs
            except Exception as e:
                logs[replica_id] = {
                    "stdout": [],
                    "stderr": [f"Error retrieving logs: {str(e)}"],
                }

        if n_previous_replica == 0:
            return logs

        # Logs of previously registered replicas
        registered_replicas = self.application_replicas.get(application_id, {}).get(
            deployment_name, {}
        )
        dead_replica_ids = [
            replica_id
            for replica_id in registered_replicas.keys()
            if replica_id not in active_replicas
        ]
        # Add logs of the n most recent dead replicas (or all if n_previous_replica is -1)
        dead_replicas_to_fetch = (
            dead_replica_ids[::-1]
            if n_previous_replica == -1
            else dead_replica_ids[::-1][:n_previous_replica]
        )
        for replica_id in dead_replicas_to_fetch:
            actor_id, replica_creation, replica_tz = registered_replicas[replica_id]
            try:
                actor_logs = self.get_actor_logs(
                    actor_id=actor_id,
                    tail=tail,
                    timezone=replica_tz,
                    creation_timestamp=replica_creation,
                )
                logs[replica_id] = actor_logs
            except Exception as e:
                logs[replica_id] = {
                    "stdout": [],
                    "stderr": [f"Error retrieving logs: {str(e)}"],
                }

        return logs
