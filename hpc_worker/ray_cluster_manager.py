import logging
import os
import re
import shutil
import socket
import subprocess
import tempfile
import time
from typing import Dict, List, Optional, Union

import ray


class RayClusterManager:
    """Manages Ray cluster lifecycle and worker nodes on an HPC system.

    Handles starting/stopping a Ray cluster, submitting worker jobs via SLURM,
    and monitoring cluster state. Supports dynamic port allocation and
    container-based worker deployment.
    """

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        # Ray cluster configuration parameters
        head_node_ip: str = None,
        head_node_port: int = 6379,
        node_manager_port: int = 6700,
        object_manager_port: int = 6701,
        redis_shard_port: int = 6702,
        ray_client_server_port: int = 10001,
        redis_password: str = None,
        # Job configuration parameters
        container_image: str = "bioengine_worker_0.1.0.sif",
        further_slurm_args: List[str] = None,
    ):
        """Initialize cluster manager with networking and resource configurations.

        Args:
            logger: Custom logger instance. Creates default logger if None.
            head_node_ip: IP address for head node. Uses first system IP if None.
            head_node_port: Port for Ray head node and GCS server.
            node_manager_port: Base port for Ray node manager services.
            object_manager_port: Port for object manager service.
            redis_shard_port: Port for Redis sharding.
            ray_client_server_port: Base port for Ray client services.
            container_image: Worker container image path.
            further_slurm_args: Additional arguments for SLURM job script.
        """
        # Set up logging
        self.logger = logger or logging.getLogger("RayClusterManager")
        if not logger:
            self.logger.setLevel(logging.INFO)
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "\033[36m%(asctime)s\033[0m - \033[32m%(name)s\033[0m - \033[1;33m%(levelname)s\033[0m - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # Set ray port configuration
        # Ports configuration: https://docs.ray.io/en/latest/ray-core/configure.html#ports-configurations
        # SLURM networking caveats: https://github.com/ray-project/ray/blob/1000ae9671967994f7bfdf7b1e1399223ad4fc61/doc/source/cluster/vms/user-guides/community/slurm.rst#id22
        if not head_node_ip:
            result = subprocess.run(["hostname", "-I"], capture_output=True, text=True)
            internal_ip = result.stdout.strip().split()[0]  # Take the first IP
        self.ray_cluster_config = {
            "head_node_ip": head_node_ip or internal_ip,
            "head_node_port": self._find_ports(
                head_node_port, step=1
            ),  # GCS server port
            "node_manager_port": self._find_ports(node_manager_port, step=100),
            "object_manager_port": self._find_ports(object_manager_port, step=100),
            "redis_shard_port": self._find_ports(redis_shard_port, step=100),
            "ray_client_server_port": self._find_ports(
                ray_client_server_port, step=10000
            ),
            "redis_password": redis_password or os.urandom(16).hex(),
        }
        self.ray_cluster_config["min_worker_port"] = (
            self.ray_cluster_config["ray_client_server_port"] + 1
        )
        self.ray_cluster_config["max_worker_port"] = (
            self.ray_cluster_config["ray_client_server_port"] + 9998
        )

        # Set job configuration from parameters
        self.job_config = {
            "container_image": container_image,
            "further_slurm_args": further_slurm_args or [],
        }
        # TODO: change worker_id to a worker job list to keep record which worker has which slurm job id
        self.worker_id = 0

        # Base directory for logs and scripts
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.logs_dir = os.path.join(self.base_dir, "logs")
        os.makedirs(self.logs_dir, exist_ok=True)

    @property
    def _ray_executable(self) -> str:
        """Get Ray executable path from PATH or Python environment."""
        # Check ray binary as it appears in PATH
        ray_path = shutil.which("ray")
        if ray_path:
            return ray_path
        else:
            # Check ray binary in python environment
            ray_path = os.path.join(ray.__file__.split("/lib/python")[0], "bin/ray")
            if os.path.exists(ray_path):
                return ray_path
            else:
                raise FileNotFoundError("Ray executable not found")

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

    def start_cluster(
        self, force: bool = False, context: Optional[Dict] = None
    ) -> Dict:
        """Start Ray cluster head node with configured ports and resources.

        Args:
            force: Force restart if cluster is running
            context: RPC context for remote calls

        Returns:
            Dict with success status and result message
        """
        # Check if Ray is already running
        if ray.is_initialized():
            if force:
                self.logger.info(
                    "Ray cluster is already initialized. Forcing restart..."
                )
                self.shutdown_cluster()
            else:
                self.logger.info("Ray cluster is already initialized")
                return {"success": True, "message": "Ray cluster is already running"}
        try:
            # Start ray as the head node with the specified parameters
            subprocess.run(
                [
                    self._ray_executable,
                    "start",
                    "--head",
                    f"--node-ip-address={self.ray_cluster_config['head_node_ip']}",
                    f"--port={self.ray_cluster_config['head_node_port']}",
                    f"--node-manager-port={self.ray_cluster_config['node_manager_port']}",
                    f"--object-manager-port={self.ray_cluster_config['object_manager_port']}",
                    f"--redis-shard-ports={self.ray_cluster_config['redis_shard_port']}",
                    f"--ray-client-server-port={self.ray_cluster_config['ray_client_server_port']}",
                    f"--min-worker-port={self.ray_cluster_config['min_worker_port']}",
                    f"--max-worker-port={self.ray_cluster_config['max_worker_port']}",
                    "--num-cpus=0",
                    "--num-gpus=0",
                    "--include-dashboard=True",
                    f"--redis-password={self.ray_cluster_config['redis_password']}",
                ],
                capture_output=True,
                check=True,
            )

            # Verify the cluster is running on the correct IP and port
            head_node_ip = self.ray_cluster_config["head_node_ip"]
            head_node_port = self.ray_cluster_config["head_node_port"]
            ray.init(
                address=f"{head_node_ip}:{head_node_port}",
                logging_format="\033[36m%(asctime)s\033[0m - \033[32m%(name)s\033[0m - \033[1;33m%(levelname)s\033[0m - %(message)s",
                configure_logging=True
            )

            return {
                "success": True,
                "message": f"Ray cluster started successfully on {head_node_ip}:{head_node_port}",
            }

        except Exception as e:
            self.logger.error(f"Error initializing Ray: {str(e)}")
            if ray.is_initialized():
                self.shutdown_cluster()
            return {"success": False, "message": f"Error initializing Ray: {str(e)}"}

    def cluster_status(self, context: Optional[Dict] = None) -> Dict:
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
            head_node_ip = self.ray_cluster_config["head_node_ip"]
            head_node_port = self.ray_cluster_config["head_node_port"]
            output["head_address"] = f"{head_node_ip}:{head_node_port}"

            # Get the available resources per node
            available_resources_per_node = ray.state.available_resources_per_node()

            # Get the status of all worker nodes
            for node in ray.nodes():
                # Skip the head node
                if "node:__internal_head__" in node["Resources"].keys():
                    continue

                # Extract worker ID from resources
                worker_id = None
                for resource in node["Resources"].keys():
                    if resource.startswith("node:__internal_worker"):
                        worker_id = int(resource.split("_")[-3])
                        break

                # Extract available resources
                available_resources = available_resources_per_node.get(
                    node["NodeID"], {}
                )

                # Extract node resources and status
                node_dict = {
                    "WorkerID": worker_id,
                    "NodeID": node["NodeID"],
                    "NodeIP": node["NodeManagerAddress"],
                    "Total GPU": node["Resources"].get("GPU", 0),
                    "Available GPU": available_resources.get("GPU", 0),
                    "Total CPU": node["Resources"].get("CPU", 0),
                    "Available CPU": available_resources.get("CPU", 0),
                    "Total Memory": node["Resources"].get("memory", 0),
                    "Available Memory": available_resources.get("memory", 0),
                }
                status = "Alive" if node["Alive"] else "Dead"
                output["worker_nodes"][status].append(node_dict)

        except Exception as e:
            # Log only unexpected errors
            if not str(e).startswith("Could not find any running Ray instance"):
                self.logger.error(f"Error checking ray cluster: {str(e)}")
        finally:
            return output

    def shutdown_cluster(
        self, grace_period: int = 30, context: Optional[Dict] = None
    ) -> Dict:
        """Stop Ray cluster and all worker nodes.

        Args:
            grace_period: Seconds to wait for graceful shutdown
            context: RPC context for remote calls

        Returns:
            Dict with success status and shutdown results
        """
        if not ray.is_initialized():
            self.logger.info("Ray cluster is not running")
            return {"success": True, "message": "Ray cluster is not running"}
        try:
            self.logger.info("Shutting down Ray cluster...")
            # Disconnect from the Ray cluster
            ray.shutdown()

            # Shutdown the Ray cluster
            result = subprocess.run(
                [self._ray_executable, "stop", f"--grace-period={grace_period}"],
                capture_output=True,
                text=True,
                check=True,
            )

            output = result.stdout
            if re.search(r"Stopped all \d+ Ray processes\.", output):
                self.logger.info("All Ray processes stopped successfully")
            elif "Did not find any active Ray processes." in output:
                self.logger.info("No active Ray processes found")
            elif re.search(r"Stopped only \d+ out of \d+ Ray processes", output):
                self.logger.warning("Some Ray processes could not be stopped")
            else:
                self.logger.warning(
                    f"Unknown message during Ray shutdown:\n----------\n{output}"
                )

            # Clean up any remaining worker jobs
            time.sleep(5)  # Wait for Ray to fully shut down
            result = self.cancel_worker_jobs()
            if not result["success"]:
                return {
                    "success": False,
                    "message": f"Error during shutdown: {result['message']}",
                }

            return {"success": True, "message": f"Successfully shut down Ray cluster."}

        except Exception as e:
            self.logger.error(f"Error shutting down Ray cluster: {str(e)}")
            return {"success": False, "message": f"Error during shutdown: {str(e)}"}

    def submit_worker_job(
        self,
        num_gpus: int = 1,
        num_cpus: int = 8,
        mem_per_cpu: int = 8,
        time_limit: str = "4:00:00",
        context: Optional[Dict] = None,
    ) -> Dict:
        """Submit SLURM job for new Ray worker node.

        Creates temporary batch script to launch containerized Ray worker
        with configured resources.

        Args:
            num_gpus: GPUs allocated per worker
            num_cpus: CPUs allocated per worker
            mem_per_cpu: Memory (GB) allocated per CPU
            time_limit: SLURM job time limit (HH:MM:SS)
            context: RPC context for remote calls

        Returns:
            Dict with job ID and resource allocation details
        """
        # First check if Ray cluster is running
        if not ray.is_initialized():
            self.logger.info("Ray cluster is not running")
            return {
                "success": False,
                "message": "Cannot start worker: Ray head node is not running",
            }
        try:
            head_node_ip = self.ray_cluster_config["head_node_ip"]
            head_node_port = self.ray_cluster_config["head_node_port"]
            address = f"{head_node_ip}:{head_node_port}"

            # Create job configuration from parameters
            job_config = self.job_config.copy()
            job_config["num_gpus"] = num_gpus
            job_config["num_cpus"] = num_cpus
            job_config["mem_per_cpu"] = mem_per_cpu
            job_config["time_limit"] = time_limit
            job_config["further_slurm_args"] = "\n".join(
                [f"#SBATCH {arg}" for arg in job_config["further_slurm_args"] if arg]
            )

            # Define the Ray worker command that will run inside the container
            self.worker_id += 1
            ray_worker_cmd = (
                f"ray start "
                f"--address={address} "
                f"--num-cpus={job_config['num_cpus']} "
                f"--num-gpus={job_config['num_gpus']} "
                "--resources='{\\\"node:__internal_worker_"
                + str(self.worker_id)
                + "__\\\": 1}' "
                f"--block"
            )

            # Define the apptainer command with the Ray worker command
            apptainer_cmd = (
                f"apptainer run "
                f"--writable-tmpfs "
                f"--contain "
                f"--nv "
                f"{job_config['container_image']} "
                f"{ray_worker_cmd}"
            )

            # Create a temporary batch script
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".sh", delete=False
            ) as batch_file:
                batch_script = f"""
                #!/bin/bash
                #SBATCH --job-name=ray_worker
                #SBATCH --ntasks=1
                #SBATCH --nodes=1
                #SBATCH --gpus={job_config["num_gpus"]}
                #SBATCH --cpus-per-task={job_config["num_cpus"]}
                #SBATCH --mem-per-cpu={job_config["mem_per_cpu"]}G
                #SBATCH --time={job_config["time_limit"]}
                #SBATCH --output={self.logs_dir}/%x_%j.out
                #SBATCH --error={self.logs_dir}/%x_%j.err
                {job_config["further_slurm_args"]}

                # Print some diagnostic information
                echo "Starting Ray worker node"
                echo "Host: $(hostname)"
                echo "Date: $(date)"
                echo "Connecting to head node: {address}"
                echo "GPUs: {job_config["num_gpus"]}, CPUs: {job_config["num_cpus"]}"
                echo "GPU info: $(nvidia-smi -L)"
                echo "Container: {job_config["container_image"]}"
                echo "Apptainer command: {apptainer_cmd}"
                echo ""
                echo "========================================"
                echo ""

                # Run the apptainer container with Ray worker
                {apptainer_cmd}

                # Print completion status
                echo "Ray worker job completed with status $?" 
                """
                for line in batch_script.split("\n"):
                    # Remove leading whitespace and write non-empty lines
                    line = line.strip()
                    if line:
                        batch_file.write(line + "\n")
                temp_script_path = batch_file.name

            # Submit the job with sbatch
            result = subprocess.run(
                ["sbatch", temp_script_path], capture_output=True, text=True, check=True
            )

            # Clean up the temporary file
            try:
                os.remove(temp_script_path)
            except Exception:
                self.logger.warning(
                    f"Failed to remove temporary script: {temp_script_path}"
                )

            # Parse job ID from Slurm output (usually "Submitted batch job 12345")
            job_id = None
            if result.stdout and "Submitted batch job" in result.stdout:
                job_id = result.stdout.strip().split()[-1]

            self.logger.info(
                f"Worker job submitted successfully. Job ID: {job_id}, Resources: {job_config['num_gpus']} GPU(s), "
                f"{job_config['num_cpus']} CPU(s), {job_config['mem_per_cpu']}G mem/CPU, {job_config['time_limit']} time limit"
            )

            return {
                "success": True,
                "job_id": job_id,
                "resources": {
                    "gpus": job_config["num_gpus"],
                    "cpus": job_config["num_cpus"],
                    "mem_per_cpu": f"{job_config['mem_per_cpu']}G",
                    "time_limit": job_config["time_limit"],
                },
            }

        except Exception as e:
            self.logger.error(
                f"Error submitting worker job: {e.stderr if hasattr(e, 'stderr') else str(e)}"
            )
            return {
                "success": False,
                "message": f"Error submitting worker job: {e.stderr if hasattr(e, 'stderr') else str(e)}",
            }

    def get_worker_jobs(self, context: Optional[Dict] = None) -> Dict:
        """Query SLURM for status of all Ray worker jobs.

        Args:
            context: RPC context for remote calls

        Returns:
            Dict with list of worker job details
        """
        try:
            # Run squeue command to get all jobs for the current user
            # Format: JobID, Name, State, Time, TimeLimit
            result = subprocess.run(
                ["squeue", "-u", os.environ["USER"], "-o", "%i %j %T %M %L"],
                capture_output=True,
                text=True,
                check=True,
            )

            # Parse the output
            lines = result.stdout.strip().split("\n")

            # Skip the header line if present
            if len(lines) > 0 and not lines[0][0].isdigit():
                lines = lines[1:]

            jobs = []
            for line in lines:
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 5:
                        job_id, job_name, state, runtime, time_limit = (
                            parts[0],
                            parts[1],
                            parts[2],
                            parts[3],
                            parts[4],
                        )

                        # Only include Ray worker jobs
                        if job_name == "ray_worker":
                            jobs.append(
                                {
                                    "job_id": job_id,
                                    "name": job_name,
                                    "state": state,
                                    "runtime": runtime,
                                    "time_limit": time_limit,
                                }
                            )

            return {"success": True, "ray_worker_jobs": jobs}

        except Exception as e:
            self.logger.error(f"Error getting job status: {str(e)}")
            return {
                "success": False,
                "message": f"Error getting job status: {str(e)}",
                "ray_worker_jobs": [],
            }

    def cancel_worker_jobs(
        self, job_ids: Optional[List[str]] = None, context: Optional[Dict] = None
    ) -> Dict:
        """Cancel running Ray worker jobs via SLURM.

        Args:
            job_ids: Specific jobs to cancel. Cancels all if None.
            context: RPC context for remote calls

        Returns:
            Dict with cancellation results
        """
        try:
            # First get all ray worker jobs
            jobs_result = self.get_worker_jobs()

            if not jobs_result["success"]:
                self.logger.error("Failed to retrieve job list for cleanup")
                return {
                    "success": False,
                    "message": "Failed to retrieve job list for cleanup",
                }

            ray_jobs = jobs_result.get("ray_worker_jobs", [])

            if not ray_jobs:
                self.logger.info("No Ray worker jobs to clean up")
                return {
                    "success": True,
                    "message": "No Ray worker jobs to clean up",
                    "cancelled_jobs": 0,
                }

            # Filter jobs if specific job_ids were provided
            if job_ids:
                job_ids = [str(job_id) for job_id in job_ids]  # Convert to strings
                ray_jobs = [job for job in ray_jobs if job["job_id"] in job_ids]
                if not ray_jobs:
                    return {
                        "success": True,
                        "message": "No matching Ray worker jobs found to cancel",
                        "cancelled_jobs": 0,
                    }

            # Get list of job IDs to cancel
            jobs_to_cancel = [job["job_id"] for job in ray_jobs]
            job_id_str = " ".join(jobs_to_cancel)

            # Cancel the jobs
            self.logger.info(
                f"Cancelling {len(jobs_to_cancel)} Ray worker job(s): {job_id_str}"
            )

            if jobs_to_cancel:
                subprocess.run(
                    ["scancel", *jobs_to_cancel], capture_output=True, check=True
                )

            return {
                "success": True,
                "message": f"Successfully cancelled {len(jobs_to_cancel)} Ray worker jobs",
                "cancelled_jobs": len(jobs_to_cancel),
                "job_ids": jobs_to_cancel,
            }

        except Exception as e:
            self.logger.error(f"Error cancelling worker jobs: {str(e)}")
            return {
                "success": False,
                "message": f"Error cancelling worker jobs: {str(e)}",
            }

    def shutdown_worker_node(
        self, worker_id: Union[int, str], context: Optional[Dict] = None
    ) -> Dict:
        """Shut down specific worker node from Ray cluster.

        Args:
            worker_id: Ray worker ID to shut down
            context: RPC context for remote calls

        Returns:
            Dict with node removal status
        """
        if not ray.is_initialized():
            self.logger.info("Ray cluster is not running")
            return {"success": False, "message": "Ray cluster is not running"}

        if isinstance(worker_id, str):
            worker_id = int(worker_id)
        elif not isinstance(worker_id, int):
            self.logger.error(f"Invalid worker ID: {worker_id}")
            return {
                "success": False,
                "message": f"Invalid worker ID: {worker_id} (dtype={type(worker_id)})",
            }
        try:
            # Check if worker node is alive
            cluster_status = self.cluster_status()
            alive_workers = cluster_status["worker_nodes"]["Alive"]
            target_node = next(
                (node for node in alive_workers if node["WorkerID"] == worker_id), None
            )
            if not target_node:
                worker_is_dead = next(
                    (
                        True
                        for node in cluster_status["worker_nodes"]["Dead"]
                        if node["WorkerID"] == worker_id
                    ),
                    False,
                )
                if worker_is_dead:
                    self.logger.warning(f"Node {worker_id} is already stopped")
                    return {
                        "success": False,
                        "message": f"Node {worker_id} is already stopped",
                    }
                else:
                    self.logger.warning(f"Node {worker_id} not found in cluster")
                    return {
                        "success": False,
                        "message": f"Node {worker_id} not found in cluster",
                    }

            # Define remote functions to get SLURM job ID
            @ray.remote(resources={f"node:__internal_worker_{worker_id}__": 0.001})
            def get_slurm_job_id():
                import os

                return os.getenv("SLURM_JOB_ID")

            # TODO: Stop all tasks on the worker node before shutting down

            # Shut down the worker node
            node_id = target_node["NodeID"]
            node_ip = target_node["NodeIP"]
            job_id = ray.get(get_slurm_job_id.remote())
            self.logger.info(
                f"Closing worker {worker_id} (NodeID={node_id} | IP={node_ip} | SLURM_JOB_ID={job_id})"
            )

            # Node was successfully shut down, now stop the worker process
            result = self.cancel_worker_jobs([job_id])
            if not result["success"]:
                return {
                    "success": False,
                    "message": f"Error closing node: {result['message']}",
                }

            self.logger.info(f"Successfully shut down worker {worker_id}")
            return {
                "success": True,
                "message": f"Successfully shut down worker {worker_id}",
            }

        except Exception as e:
            self.logger.error(f"Error closing worker {worker_id}: {str(e)}")
            return {
                "success": False,
                "message": f"Error closing worker {worker_id}: {str(e)}",
            }


if __name__ == "__main__":
    # TODO: move this to tests
    print("===== Testing Ray cluster class =====", end="\n\n")

    # Test the class
    ray_manager = RayClusterManager(
        # further_slurm_args=["-C 'thin'"]
    )

    # Start Ray cluster
    print("Starting Ray cluster...")
    start_result = ray_manager.start_cluster(force=True)
    print(start_result, end="\n\n")

    # Test getting job status
    print("Getting worker job status...")
    jobs_result = ray_manager.get_worker_jobs()
    print(jobs_result, end="\n\n")

    if len(jobs_result.get("ray_worker_jobs", [])) > 0:
        # Cancel all previous worker jobs
        print("Cancelling all worker jobs...")
        cancel_result = ray_manager.cancel_worker_jobs()
        print(cancel_result, end="\n\n")

    # Test submitting a worker job
    print("Submitting a worker job...")
    submit_result = ray_manager.submit_worker_job()
    print(submit_result, end="\n\n")

    # Wait for job to start
    print("Waiting for submitted worker job to start...")
    started = False
    while not started:
        time.sleep(3)
        cluster_status = ray_manager.get_worker_jobs()
        print(cluster_status, end="\n\n")
        started = any(
            job["state"] == "RUNNING"
            for job in cluster_status.get("ray_worker_jobs", [])
        )

    # Test getting node IDs
    worker_nodes = []
    while not worker_nodes:
        # Wait for worker node to appear in cluster status
        print("Waiting for worker node to appear in cluster status...")
        time.sleep(3)
        cluster_status = ray_manager.cluster_status()
        worker_nodes = cluster_status["worker_nodes"]["Alive"]
    print(cluster_status, end="\n\n")

    # Test running a remote function on worker node
    @ray.remote(num_cpus=1, num_gpus=1)
    def test_remote():
        import time

        time.sleep(3)
        return "Successfully run a task on the worker node!"

    obj_ref = test_remote.remote()
    print(ray.get(obj_ref))

    # Test closing a worker node
    print("Shutting down a worker node...")
    worker_id = worker_nodes[0]["WorkerID"]
    shutdown_result = ray_manager.shutdown_worker_node(worker_id)
    print(shutdown_result, end="\n\n")

    # Test shutting down Ray cluster
    print("Shutting down Ray cluster...")
    shutdown_result = ray_manager.shutdown_cluster()
    print(shutdown_result, end="\n\n")
