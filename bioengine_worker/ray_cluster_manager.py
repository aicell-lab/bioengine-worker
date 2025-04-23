import logging
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import ray
from ray import serve

from bioengine_worker.slurm_actor import SlurmActor
from bioengine_worker.utils.format_time import format_time
from bioengine_worker.utils.logger import create_logger, logging_format


# TODO: make async
class RayClusterManager:
    """Manages Ray cluster lifecycle and worker nodes on an HPC system.

    Handles starting/stopping a Ray cluster, submitting worker jobs via SLURM,
    and monitoring cluster state. Supports dynamic port allocation and
    container-based worker deployment.
    """

    def __init__(
        self,
        # Ray cluster configuration parameters
        head_node_ip: str = None,
        head_node_port: int = 6379,
        node_manager_port: int = 6700,
        object_manager_port: int = 6701,
        redis_shard_port: int = 6702,
        serve_port: int = 8100,
        dashboard_port: int = 8269,
        ray_client_server_port: int = 10001,
        redis_password: str = None,
        ray_temp_dir: str = "/tmp/ray",
        data_dir: str = None,
        head_num_cpus: int = 0,
        head_num_gpus: int = 0,
        # Job configuration parameters
        image_path: str = "./apptainer_images/bioengine-worker_0.1.6.sif",
        slurm_logs_dir: str = None,
        further_slurm_args: List[str] = None,
        # Logger
        logger: Optional[logging.Logger] = None,
        _debug: bool = False,
    ):
        """Initialize cluster manager with networking and resource configurations.

        Ray ports configuration: https://docs.ray.io/en/latest/ray-core/configure.html#ports-configurations
        SLURM networking caveats: https://github.com/ray-project/ray/blob/1000ae9671967994f7bfdf7b1e1399223ad4fc61/doc/source/cluster/vms/user-guides/community/slurm.rst#id22

        Args:
            head_node_ip: IP address for head node. Uses first system IP if None.
            head_node_port: Port for Ray head node and GCS server.
            node_manager_port: Base port for Ray node manager services.
            object_manager_port: Port for object manager service.
            redis_shard_port: Port for Redis sharding.
            serve_port: Port for Ray Serve.
            dashboard_port: Port for the dashboard.
            ray_client_server_port: Base port for Ray client services.
            redis_password: Password for Redis server.
            ray_temp_dir: Temporary directory for Ray. Default is '/tmp/ray'.
            data_dir: Data directory mounted to the container. Default is None.
            head_num_cpus: Number of CPUs for head node if starting a local cluster (slurm_available=False).
            head_num_gpus: Number of GPUs for head node if starting a local cluster (slurm_available=False).
            image_path: Worker container image path.
            slurm_logs_dir: Directory for SLURM logs.
            further_slurm_args: Additional arguments for SLURM job script.
            logger: Custom logger instance. Creates default logger if None.
            _debug: Enable debug logging level.
        """
        # Set up logging
        self.logger = logger or create_logger(
            name="RayClusterManager",
            level=logging.DEBUG if _debug else logging.INFO,
        )

        # Find and store Ray executable path
        self._ray_exec_path = self._find_ray_executable()

        # Check number of CPUs and GPUs
        self.slurm_available = self._check_slurm_available()
        if self.slurm_available:
            if head_num_cpus:
                self.logger.warning(
                    "Ignoring head_num_cpus setting since SLURM is available - will be set to 0"
                )
                head_num_cpus = 0
            if head_num_gpus:
                self.logger.warning(
                    "Ignoring head_num_gpus setting since SLURM is available - will be set to 0"
                )
                head_num_gpus = 0
        elif head_num_cpus <= 0 and head_num_gpus <= 0:
            raise ValueError(
                "When SLURM is not available, either head_num_cpus or head_num_gpus must be greater than 0"
            )

        self.ray_cluster_config = {
            "head_node_ip": head_node_ip or self._find_internal_ip(),
            "head_node_port": head_node_port,  # GCS server port
            "node_manager_port": node_manager_port,
            "object_manager_port": object_manager_port,
            "redis_shard_port": redis_shard_port,
            "serve_port": serve_port,
            "dashboard_port": dashboard_port,
            "ray_client_server_port": ray_client_server_port,
            "redis_password": redis_password or os.urandom(16).hex(),
            "ray_temp_dir": ray_temp_dir,
            "data_dir": data_dir,
            "head_num_cpus": head_num_cpus,
            "head_num_gpus": head_num_gpus,
        }
        self.ray_start_time = None

        if self.slurm_available:
            # Set job configuration from parameters
            image_path = Path(image_path).resolve()

            self.job_config = {
                "image_path": str(image_path),
                "further_slurm_args": further_slurm_args or [],
            }

            # Set up SLURM actor
            self.slurm_actor = SlurmActor(
                job_name="ray_worker", logs_dir=slurm_logs_dir, _debug=_debug
            )

    @property
    def head_node_address(self) -> str:
        """Get the address of the Ray head node.

        Returns:
            str with the address of the head node
        """
        if ray.is_initialized():
            head_node_ip = self.ray_cluster_config["head_node_ip"]
            head_node_port = self.ray_cluster_config["head_node_port"]
            return f"{head_node_ip}:{head_node_port}"
        else:
            self.logger.error(
                "Can not get head node address - Ray cluster is not running"
            )
            raise RuntimeError("Ray cluster is not running")

    def _find_ray_executable(self) -> str:
        """Find the Ray executable path.

        Returns:
            str with the path to the Ray executable
        """
        ray_path = Path(sys.executable).parent / "ray"
        if not ray_path.exists():
            raise FileNotFoundError("Ray executable not found")
        self.logger.debug(f"Ray executable found at: {ray_path}")
        return str(ray_path)

    def _find_internal_ip(self) -> str:
        """Find the internal IP address of the system.

        Returns:
            str with the internal IP address
        """
        result = subprocess.run(["hostname", "-I"], capture_output=True, text=True)
        return result.stdout.strip().split()[0]  # Take the first IP

    def _check_slurm_available(self) -> bool:
        """Check if SLURM is available on the system.

        Returns:
            True if SLURM is available, False otherwise
        """
        try:
            subprocess.run(["sinfo"], capture_output=True, text=True, check=True)
            self.logger.info("SLURM is available")
            return True
        except FileNotFoundError:
            self.logger.info("SLURM is not available")
            return False

    def _find_ports(self, port: int, step: int = 1) -> int:
        """Find next available port starting from given port number.

        Args:
            port: Starting port number to check
            step: Increment between port numbers to check

        Returns:
            First available port number
        """
        available = False
        out_port = port
        while not available:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(("localhost", out_port)) != 0:
                    available = True
                else:
                    out_port += step
        if out_port != port:
            self.logger.warning(
                f"Port {port} is not available. Using {out_port} instead."
            )
        return out_port

    def _set_cluster_ports(self):
        # Update ports to available ones
        self.ray_cluster_config["head_node_port"] = self._find_ports(
            self.ray_cluster_config["head_node_port"], step=1
        )
        self.ray_cluster_config["node_manager_port"] = self._find_ports(
            self.ray_cluster_config["node_manager_port"], step=100
        )
        self.ray_cluster_config["object_manager_port"] = self._find_ports(
            self.ray_cluster_config["object_manager_port"], step=100
        )
        self.ray_cluster_config["redis_shard_port"] = self._find_ports(
            self.ray_cluster_config["redis_shard_port"], step=100
        )
        self.ray_cluster_config["serve_port"] = self._find_ports(
            self.ray_cluster_config["serve_port"], step=1
        )
        self.ray_cluster_config["dashboard_port"] = self._find_ports(
            self.ray_cluster_config["dashboard_port"], step=1
        )
        self.ray_cluster_config["ray_client_server_port"] = self._find_ports(
            self.ray_cluster_config["ray_client_server_port"], step=10000
        )
        self.ray_cluster_config["min_worker_port"] = (
            self.ray_cluster_config["ray_client_server_port"] + 1
        )
        self.ray_cluster_config["max_worker_port"] = (
            self.ray_cluster_config["ray_client_server_port"] + 9998
        )

    def _node_resource_to_worker_id(self, node_resource: Dict) -> Optional[str]:
        # Extract worker ID from resources
        worker_id = None
        for resource in node_resource.keys():
            if resource.startswith("node:__internal_worker"):
                worker_id = resource.split("_")[-3]
                break
        return worker_id

    def _get_worker_status(self, worker_id: str) -> str:
        # Check if worker node exists in cluster
        for node in ray.nodes():
            node_worker_id = self._node_resource_to_worker_id(node["Resources"])
            if node_worker_id == worker_id:
                node_id = node["NodeID"]
                node_ip = node["NodeManagerAddress"]

                if node["Alive"]:
                    self.logger.debug(
                        f"Ray worker '{worker_id}' on machine '{node_ip}' with node ID '{node_id}' is currently running"
                    )
                    return "alive"
                else:
                    self.logger.debug(
                        f"Ray worker '{worker_id}' on machine '{node_ip}' with node ID '{node_id}' is already stopped"
                    )
                    return "dead"

        # If not found in either alive or dead nodes
        self.logger.debug(f"Ray worker '{worker_id}' not found in cluster")
        return "not_found"

    def start_cluster(self, force_clean_up: bool = False) -> str:
        """Start Ray cluster head node with configured ports and resources.

        Args:
            force_clean_up: Force cleanup of previous Ray cluster

        Returns:
            str with the address of the head node
        """
        if force_clean_up:
            self.logger.info("Forcing Ray cleanup...")
            self.shutdown_cluster()
        elif ray.is_initialized():
            self.logger.info("Ray cluster is already initialized")
            return self.head_node_address
        try:
            self.logger.info("Starting Ray cluster...")

            # Check and set cluster ports
            self._set_cluster_ports()

            # Start ray as the head node with the specified parameters
            result = subprocess.run(
                [
                    self._ray_exec_path,
                    "start",
                    "--head",
                    f"--num-cpus={self.ray_cluster_config['head_num_cpus']}",
                    f"--num-gpus={self.ray_cluster_config['head_num_gpus']}",
                    f"--node-ip-address={self.ray_cluster_config['head_node_ip']}",
                    f"--port={self.ray_cluster_config['head_node_port']}",
                    f"--node-manager-port={self.ray_cluster_config['node_manager_port']}",
                    f"--object-manager-port={self.ray_cluster_config['object_manager_port']}",
                    f"--redis-shard-ports={self.ray_cluster_config['redis_shard_port']}",
                    f"--ray-client-server-port={self.ray_cluster_config['ray_client_server_port']}",
                    f"--min-worker-port={self.ray_cluster_config['min_worker_port']}",
                    f"--max-worker-port={self.ray_cluster_config['max_worker_port']}",
                    "--include-dashboard=True",
                    f"--dashboard-port={self.ray_cluster_config['dashboard_port']}",
                    f"--redis-password={self.ray_cluster_config['redis_password']}",
                    f"--temp-dir={self.ray_cluster_config['ray_temp_dir']}",
                ],
                capture_output=True,
                check=True,
            )
            self.logger.debug(
                f"Ray start command output:\n----------\n{result.stdout.decode()}"
            )

            # Verify the cluster is running on the correct IP and port
            head_node_ip = self.ray_cluster_config["head_node_ip"]
            head_node_port = self.ray_cluster_config["head_node_port"]
            address = f"{head_node_ip}:{head_node_port}"
            ray.init(address=address, logging_format=logging_format)
            serve.start(
                http_options={
                    "host": "0.0.0.0",
                    "port": self.ray_cluster_config["serve_port"],
                },
                # logging_config=logging_format,  # TODO: Fix logging
            )
            self.ray_start_time = time.time()
            formatted_time = format_time(self.ray_start_time)
            start_time = formatted_time["start_time"]
            self.logger.info(
                f"Ray cluster and Ray Serve started successfully on '{address}' - Start time: {start_time}"
            )

            # Change '<ray_temp_dir>/session_latest' symlink to use relative path instead of absolute (container) path
            # (needed to start ray in container)
            symlink_path = (
                Path(self.ray_cluster_config["ray_temp_dir"]) / "session_latest"
            )
            if symlink_path.is_symlink():
                # Get the target of the symlink
                symlink_target = symlink_path.readlink()
            else:
                self.logger.error(f"Symlink '{symlink_path}' does not exist")
                raise FileNotFoundError(f"Symlink '{symlink_path}' does not exist")

            relative_symlink_target = Path(symlink_target.name)
            self.logger.debug(
                f"Changing symlink target from '{symlink_target}' to '{relative_symlink_target}'"
            )
            symlink_path.unlink()
            symlink_path.symlink_to(relative_symlink_target)

            return address

        except subprocess.CalledProcessError as e:
            self.logger.error(
                f"Ray start command failed with error code {e.returncode}:\n{e.stderr.decode()}"
            )
            if ray.is_initialized():
                self.shutdown_cluster()
            raise e
        except Exception as e:
            self.logger.error(f"Error initializing Ray: {e}")
            if ray.is_initialized():
                self.shutdown_cluster()
            raise e

    def get_status(self) -> Dict:
        """Get current cluster state including head node and worker information.

        Returns:
            Dict with head node address and worker node IDs
        """
        output = {"head_address": None, "worker_nodes": {"Alive": [], "Dead": []}}
        if not ray.is_initialized():
            self.logger.info("Ray cluster is not running")
            return output
        try:
            # Get the head node address
            output["head_address"] = self.head_node_address

            # Get cluster start time and uptime
            formatted_time = format_time(self.ray_start_time)
            output["start_time"] = formatted_time["start_time"]
            output["uptime"] = formatted_time["duration_since"]

            # Get the available resources per node
            available_resources_per_node = ray.state.available_resources_per_node()

            # Get the list of running jobs
            if self.slurm_available:
                running_jobs = self.slurm_actor.get_jobs().keys()

            # Get the status of all worker nodes
            for node in ray.nodes():
                # Skip the head node
                if (
                    "node:__internal_head__" in node["Resources"].keys()
                    and self.slurm_available
                ):
                    continue

                if not node["Resources"]:
                    self.logger.debug(f"Encountered dead worker node without worker ID")
                    assert not node["Alive"]
                    # Dead nodes have empty resources -> no worker ID
                    worker_id = None
                else:
                    # Extract worker ID from resources
                    worker_id = self._node_resource_to_worker_id(node["Resources"])

                    # Skip nodes if job is not running anymore
                    if self.slurm_available and worker_id not in running_jobs:
                        self.logger.debug(
                            f"Skipping worker node '{node['NodeID']}' with already cancelled job ID '{worker_id}'"
                        )
                        continue

                # Extract available resources
                available_resources = available_resources_per_node.get(
                    node["NodeID"], {}
                )

                # Extract node resources and status
                total_gpu = node["Resources"].get("GPU", 0)
                available_gpu = available_resources.get("GPU", 0)
                if total_gpu > 0:
                    gpu_util = (total_gpu - available_gpu) / total_gpu
                else:
                    gpu_util = 0.0

                total_cpu = node["Resources"].get("CPU", 0)
                available_cpu = available_resources.get("CPU", 0)
                if total_cpu > 0:
                    cpu_util = (total_cpu - available_cpu) / total_cpu
                else:
                    cpu_util = 0.0

                total_memory = node["Resources"].get("memory", 0)
                available_memory = available_resources.get("memory", 0)
                if total_memory > 0:
                    memory_util = (total_memory - available_memory) / total_memory
                else:
                    memory_util = 0.0

                node_dict = {
                    "WorkerID": worker_id,
                    "NodeID": node["NodeID"],
                    "NodeIP": node["NodeManagerAddress"],
                    "Total GPU": total_gpu,
                    "Available GPU": available_gpu,
                    "GPU Utilization": gpu_util,
                    "Total CPU": total_cpu,
                    "Available CPU": available_cpu,
                    "CPU Utilization": cpu_util,
                    "Total Memory": total_memory,
                    "Available Memory": available_memory,
                    "Memory Utilization": memory_util,
                }

                status = "Alive" if node["Alive"] else "Dead"
                output["worker_nodes"][status].append(node_dict)

        except Exception as e:
            # Log only unexpected errors
            if not str(e).startswith("Could not find any running Ray instance"):
                self.logger.error(f"Error checking ray cluster: {e}")
            raise e
        finally:
            return output

    def shutdown_cluster(self, grace_period: int = 30) -> None:
        """Stop Ray cluster and all worker nodes.

        Args:
            grace_period: Seconds to wait for graceful shutdown
        """
        try:
            # Disconnect from Ray cluster if it was initialized
            if ray.is_initialized():

                # Stop Ray Serve if it was initialized
                try:
                    serve.context._get_global_client()
                    self.logger.info("Shutting down Ray Serve...")
                    serve.shutdown()
                except serve.exceptions.RayServeException:
                    pass

                self.logger.info("Disconnecting from Ray cluster...")
                ray.shutdown()

            # Shutdown the Ray cluster head node
            self.logger.info("Shutting down Ray cluster...")
            result = subprocess.run(
                [self._ray_exec_path, "stop", f"--grace-period={grace_period}"],
                capture_output=True,
                text=True,
                check=True,
            )

            output = result.stdout
            if re.search(r"Stopped all \d+ Ray processes\.", output):
                self.logger.info("All Ray processes stopped successfully")
            elif "Did not find any active Ray processes." in output:
                self.logger.info("No active Ray processes found")
            else:
                message = re.search(
                    r"Stopped only (\d+) out of (\d+) Ray processes within the grace period (\d+) seconds\.",
                    output,
                )
                if message:
                    self.logger.warning(
                        f"Some Ray processes could not be stopped: {message.group(0)}"
                    )
                else:
                    self.logger.warning(
                        f"Unknown message during Ray shutdown:\n----------\n{output}"
                    )

            # Clean up any remaining worker jobs
            time.sleep(5)  # Wait for Ray to fully shut down
            if self.slurm_available:
                self.slurm_actor.cancel_jobs(grace_period=grace_period)

            self.logger.info("Ray cluster shut down complete")

            self.ray_start_time = None

        except Exception as e:
            self.logger.error(f"Error shutting down Ray cluster: {e}")
            raise e

    def add_worker(
        self,
        num_gpus: int = 1,
        num_cpus: int = 8,
        mem_per_cpu: int = 16,
        time_limit: str = "6:00:00",
    ) -> None:
        """Submit SLURM job for new Ray worker node.

        Creates temporary batch script to launch containerized Ray worker
        with configured resources.

        Args:
            num_gpus: GPUs allocated per worker
            num_cpus: CPUs allocated per worker
            mem_per_cpu: Memory (GB) allocated per CPU
            time_limit: SLURM job time limit (HH:MM:SS)
        """
        try:
            # Check if SLURM is available
            if not self.slurm_available:
                raise RuntimeError(
                    "SLURM is not available. Adding a worker is not possible"
                )

            # Check if Ray cluster is running
            if not ray.is_initialized():
                raise RuntimeError("Ray cluster is not running")

            # Define the Ray worker command that will run inside the container
            ray_worker_cmd = (
                "ray start "
                f"--address={self.head_node_address} "
                f"--num-cpus={num_cpus} "
                f"--num-gpus={num_gpus} "
                "--resources='{\\\"node:__internal_worker_${SLURM_JOB_ID}__\\\": 1}' "
                "--block"
            )
            # ray start --address='10.81.254.11:6379' --num-cpus=8 --num-gpus=1 --resources='{"node:__internal_worker_${SLURM_JOB_ID}__": 1}' --block

            # Define the apptainer command with the Ray worker command
            if self.ray_cluster_config["data_dir"]:
                data_dir = self.ray_cluster_config["data_dir"]
                self.logger.info(
                    f"Binding outside data directory '{data_dir}' to container directory '/data'"
                )
                bind_dir_flag = f"--bind {data_dir}:/data "
            else:
                bind_dir_flag = ""

            # TODO: check if --env works with double quotes in single quotes (without backslashes)
            apptainer_cmd = (
                "apptainer run "
                "--nv "
                "--cleanenv "
                "--env=\"SLURM_JOB_ID\"=\"$SLURM_JOB_ID\" "
                "--pwd /app "
                f"{bind_dir_flag}"
                f"{self.job_config['image_path']} "
                f"{ray_worker_cmd}"
            )

            # Create sbatch script using SlurmActor
            sbatch_script = self.slurm_actor.create_sbatch_script(
                command=apptainer_cmd,
                gpus=num_gpus,
                cpus_per_task=num_cpus,
                mem_per_cpu=mem_per_cpu,
                time=time_limit,
                further_slurm_args=self.job_config.get("further_slurm_args"),
            )

            # Submit the job
            job_id = self.slurm_actor.submit_job(sbatch_script, delete_script=True)

            if job_id:
                self.logger.info(
                    f"Worker job submitted successfully. Worker & Job ID: {job_id}, Resources: {num_gpus} GPU(s), "
                    f"{num_cpus} CPU(s), {mem_per_cpu}G mem/CPU, {time_limit} time limit"
                )

                return job_id  # equivalent to worker ID
            else:
                raise RuntimeError("Failed to submit worker job")

        except Exception as e:
            self.logger.error(f"Error adding worker: {e}")
            raise e

    def remove_worker(self, worker_id: str, grace_period: int = 30) -> None:
        """Shut down specific worker node from Ray cluster.

        Args:
            worker_id: Ray worker ID to shut down (e.g. 'w1', 'w2')

        Returns:
            True if worker was successfully shut down
        """
        try:
            # Check if SLURM is available
            if not self.slurm_available:
                raise RuntimeError(
                    "SLURM is not available. Removing a worker is not possible"
                )

            # Check if Ray cluster is running
            if not ray.is_initialized():
                raise RuntimeError("Ray cluster is not running")

            # Check worker status
            worker_status = self._get_worker_status(worker_id)
            if worker_status == "not_found":
                raise ValueError(f"Worker ID '{worker_id}' not found in cluster status")

            self.logger.info(
                f"Removing worker '{worker_id}' (status='{worker_status}') from cluster status..."
            )

            if worker_status == "alive":
                self.logger.info(f"Stopping all processes on worker '{worker_id}'...")

                @ray.remote(resources={f"node:__internal_worker_{worker_id}__": 0.01})
                def stop_worker():
                    """Stops the worker node by executing 'ray stop'."""
                    import subprocess

                    subprocess.Popen(
                        ["ray", "stop"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )

                # This should only effect a single worker node if the worker runs in a container
                obj_ref = stop_worker.remote()

                try:
                    ray.get(obj_ref, timeout=15)
                except ray.exceptions.GetTimeoutError:
                    self.logger.error(
                        f"Failed to send shutdown command to worker '{worker_id}' within 15 seconds"
                    )
                    raise e

                # Wait for worker node to disappear from cluster status
                start_time = time.time()
                while time.time() - start_time < grace_period:
                    time.sleep(3)
                    status = self._get_worker_status(worker_id)
                    if status != "alive":
                        break

            cancelled_jobs = self.slurm_actor.cancel_jobs([worker_id])
            if cancelled_jobs == [worker_id]:
                self.logger.info(f"Successfully removed worker '{worker_id}'")
            else:
                raise RuntimeError(f"Failed to cancel job '{worker_id}'")

        except Exception as e:
            self.logger.error(f"Error shutting down worker {worker_id}: {e}")
            raise e


if __name__ == "__main__":
    from pathlib import Path

    # TODO: move this to tests
    print("\n===== Testing Ray cluster manager =====\n")

    # ray_manager = RayClusterManager(
    #     head_num_cpus=4,
    #     head_num_gpus=0,
    #     ray_temp_dir=f"/tmp/ray/{os.environ['USER']}",
    # )
    # ray_manager.start_cluster(force_clean_up=True)
    # ray_manager.get_status()
    # ray_manager.shutdown_cluster()

    # Test the class
    ray_manager = RayClusterManager(
        ray_temp_dir=f"/tmp/ray/{os.environ['USER']}",
        data_dir=str(Path(__file__).parent.parent / "data"),
        image_path=str(
            Path(__file__).parent.parent / "apptainer_images/bioengine-worker_0.1.6.sif"
        ),
        # further_slurm_args=["-C 'thin'"]
        _debug=True,
    )

    # Start Ray cluster
    ray_manager.start_cluster(force_clean_up=True)

    # Test submitting a worker job
    print("Adding a worker...")
    worker_id = ray_manager.add_worker(time_limit="00:30:00")

    # Wait for worker to start
    status = ""
    while status != "RUNNING":
        # Wait for worker node to appear in cluster status
        print("Waiting for job to start...")
        time.sleep(3)
        jobs = ray_manager.slurm_actor.get_jobs()
        if worker_id not in jobs:
            raise RuntimeError(
                f"Job died before worker node appeared in cluster status"
            )
        status = jobs[worker_id]["state"]

    while status != "alive":
        print("Waiting for worker node to start...")
        time.sleep(3)
        status = ray_manager._get_worker_status(worker_id)

    # Test cluster status
    cluster_status = ray_manager.get_status()
    print("\n=== Cluster status ===\n", cluster_status, end="\n\n")

    # Test running a remote function on worker node
    @ray.remote(
        num_cpus=1,
        num_gpus=1,
        # runtime_env={"pip": ["hypha-rpc"]},
    )
    def test_remote():
        import os
        import time

        # Check if runtime environment is set up correctly
        from hypha_rpc.sync import connect_to_server

        # Check Ray's temporary directory
        assert os.path.exists("/tmp/ray"), "Temporary directory does not exist"

        # Check data directory
        num_files = len(os.listdir("/data"))
        assert num_files, "Data directory is empty"

        time.sleep(1)

        return f"Successfully run a task in runtime environment on the worker node! (Number of files in data directory: {num_files})"

    obj_ref = test_remote.remote()
    print(ray.get(obj_ref))

    # Test closing a worker node
    ray_manager.remove_worker(worker_id)

    # Test shutting down Ray cluster
    ray_manager.shutdown_cluster()
