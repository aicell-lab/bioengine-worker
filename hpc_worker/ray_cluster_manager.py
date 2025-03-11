import logging
import os
import ray
import subprocess
import tempfile
from typing import Dict, List, Optional
import shutil


class RayClusterManager:
    """
    Manages Ray cluster operations including starting/stopping the cluster
    and submitting/monitoring worker jobs.
    """
    
    def __init__(
        self, 
        logger: Optional[logging.Logger] = None,
        # Job configuration parameters
        num_gpus: int = 1,
        num_cpus: int = 4,
        mem_per_cpu: int = 8,
        time_limit: str = "4:00:00",
        container_image: str = "chiron_worker_0.1.0.sif",
    ):
        """Initialize the Ray cluster manager
        
        Args:
            logger: Optional logger instance. If None, creates a new logger.
            num_gpus: Number of GPUs per worker (default: 1)
            num_cpus: Number of CPUs per worker (default: 4) 
            mem_per_cpu: Memory per CPU in GB (default: 8)
            time_limit: Time limit in HH:MM:SS format (default: 4:00:00)
            container_image: Container image for workers (default: chiron_worker_0.1.0.sif)
        """
        # Set up logging
        self.logger = logger or logging.getLogger("ray_cluster")
        if not logger:
            self.logger.setLevel(logging.INFO)
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # Set job configuration from parameters
        self.job_config = {
            "num_gpus": num_gpus,
            "num_cpus": num_cpus,
            "mem_per_cpu": mem_per_cpu,
            "time_limit": time_limit,
            "container_image": container_image,
        }
        
        # Base directory for logs and scripts
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.logs_dir = os.path.join(self.base_dir, "logs")
        os.makedirs(self.logs_dir, exist_ok=True)

    @property
    def ray_executable(self) -> str:
        """Get the path to the Ray executable
        
        Returns:
            Path to the Ray executable
        """
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
        
    def check_cluster(self) -> Dict:
        """Check the status of the Ray cluster
        
        Returns:
            Dict containing head node status, worker count and worker node IDs
        """
        output = {
            "head_running": False,
            "head_address": None,
            "worker_count": 0,
            "worker_node_ids": []
        }
        try:
            was_connected = ray.is_initialized()
            if not was_connected:
                try:
                    ray.init(address="auto")
                except ConnectionError:
                    # Expected state - no Ray cluster running
                    return output

            # Set the head node as running
            output["head_running"] = True

            # Get the head node address
            runtime_context = ray.get_runtime_context()
            output["head_address"] = runtime_context.gcs_address
            head_ip = output["head_address"].split(":")[0]
            
            # Get worker nodes and their IDs
            worker_node_ids = [
                node["NodeID"] for node in ray.nodes() 
                if node["Alive"]  # Exclude dead nodes
                and f"node:{head_ip}" not in node["Resources"]  # Exclude head node
                and node["Resources"].get("CPU", 0) == self.job_config["num_cpus"]  # Match worker CPU count
                and node["Resources"].get("GPU", 0) == self.job_config["num_gpus"]  # Match worker GPU count
            ]
            output["worker_count"] = len(worker_node_ids)
            output["worker_node_ids"] = worker_node_ids

            # Disconnect only if we connected in this function
            if not was_connected:
                ray.shutdown()

            return output
        except Exception as e:
            # Log only unexpected errors
            if not str(e).startswith("Could not find any running Ray instance"):
                self.logger.error(f"Error checking ray cluster: {str(e)}")

            # Make sure to disconnect even if there was an error
            if ray.is_initialized():
                ray.shutdown()
                
            return output
    
    def start_cluster(self, force: bool = False, context: Optional[Dict] = None) -> Dict:
        """Start a Ray cluster head node
        
        Args:
            context: RPC context
            
        Returns:
            Dict containing operation status and result message
        """
        try:
            # Check if Ray is already running
            ray_status = self.check_cluster()
            if ray_status["head_running"]:
                if force:
                    self.logger.info("Ray cluster is already running. Forcing restart...")
                    self.shutdown_cluster()
                else:    
                    self.logger.info("Ray cluster is already running")
                    return {"success": True, "message": "Ray cluster is already running"}

            # Start ray as the head node with the specified parameters
            subprocess.run(
                [
                    self.ray_executable,
                    "start",
                    "--head",
                    "--num-cpus=0",
                    "--num-gpus=0",
                    "--include-dashboard=False",
                ],
                check=True
            )

            # Verify the cluster is running on the correct IP and port
            ray_status = self.check_cluster()
            if not ray_status["head_running"]:
                self.logger.error("Ray cluster failed to start")
                return {"success": False, "message": "Ray cluster failed to start"}

            host_ip, port = ray_status["head_address"].split(":")
            return {
                "success": True,
                "message": f"Ray cluster started successfully on {host_ip}:{port}",
            }

        except Exception as e:
            self.logger.error(f"Error initializing Ray: {str(e)}")
            return {"success": False, "message": f"Error initializing Ray: {str(e)}"}
    
    def shutdown_cluster(self, context: Optional[Dict] = None) -> Dict:
        """Shutdown the Ray cluster and all its nodes
        
        Args:
            context: RPC context
            
        Returns:
            Dict containing operation status and result message
        """
        try:
            # Shutdown the Ray cluster
            subprocess.run(
                [self.ray_executable, "stop"], check=True
            )

            # Verify shutdown was successful
            ray_status = self.check_cluster()
            if not ray_status["head_running"]:
                self.logger.info(f"Successfully shut down Ray cluster.")
                return {
                    "success": True,
                    "message": f"Successfully shut down Ray cluster."
                }
            else:
                self.logger.warning("Ray cluster is still running after shutdown attempt.")
                return {
                    "success": False,
                    "message": "Failed to shut down Ray cluster completely",
                }

        except Exception as e:
            self.logger.error(f"Error shutting down Ray cluster: {str(e)}")
            return {"success": False, "message": f"Error during shutdown: {str(e)}"}
    
    def submit_worker_job(
        self,
        context: Optional[Dict] = None,
    ) -> Dict:
        """Submit a Slurm job to start a Ray worker
        
        Args:
            context: RPC context
            
        Returns:
            Dict with job submission status
        """
        try:
            # First check if Ray cluster is running
            ray_status = self.check_cluster()
            if not ray_status["head_running"]:
                return {
                    "success": False,
                    "message": "Cannot start worker: Ray head node is not running",
                }

            # Define the Ray worker command that will run inside the container
            ray_worker_cmd = f"ray start --address={ray_status['head_address']} --num-cpus={self.job_config['num_cpus']} --num-gpus={self.job_config['num_gpus']} --block"

            # Define the apptainer command with the Ray worker command
            apptainer_cmd = f"apptainer run --writable-tmpfs --contain --nv {self.job_config['container_image']} {ray_worker_cmd}"

            # Create a temporary batch script
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".sh", delete=False
            ) as batch_file:
                batch_script = f"""
                #!/bin/bash
                #SBATCH --job-name=ray_worker
                #SBATCH --ntasks=1
                #SBATCH --nodes=1
                #SBATCH --gpus={self.job_config["num_gpus"]}
                #SBATCH --cpus-per-task={self.job_config["num_cpus"]}
                #SBATCH --mem-per-cpu={self.job_config["mem_per_cpu"]}G
                #SBATCH --time={self.job_config["time_limit"]}
                #SBATCH --output={self.logs_dir}/%x_%j.out
                #SBATCH --error={self.logs_dir}/%x_%j.err

                # Print some diagnostic information
                echo "Starting Ray worker node"
                echo "Host: $(hostname)"
                echo "Date: $(date)"
                echo "Connecting to head node: {ray_status['head_address']}"
                echo "GPUs: {self.job_config["num_gpus"]}, CPUs: {self.job_config["num_cpus"]}"
                echo "GPU info: $(nvidia-smi -L)"

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
                self.logger.warning(f"Failed to remove temporary script: {temp_script_path}")

            # Parse job ID from Slurm output (usually "Submitted batch job 12345")
            job_id = None
            if result.stdout and "Submitted batch job" in result.stdout:
                job_id = result.stdout.strip().split()[-1]

            self.logger.info(
                f"Worker job submitted successfully. Job ID: {job_id}, Resources: {self.job_config['num_gpus']} GPU(s), "
                f"{self.job_config['num_cpus']} CPU(s), {self.job_config['mem_per_cpu']}G mem/CPU, {self.job_config['time_limit']} time limit"
            )
            
            return {
                "success": True,
                "message": f"Worker job submitted successfully with job ID {job_id}",
                "job_id": job_id,
                "head_node": f"{ray_status['head_address']}",
                "resources": {
                    "gpus": self.job_config['num_gpus'],
                    "cpus": self.job_config['num_cpus'],
                    "mem_per_cpu": f"{self.job_config['mem_per_cpu']}G",
                    "time_limit": self.job_config['time_limit'],
                    "container": self.job_config['container_image'],
                },
            }

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error submitting Slurm job: {e.stderr}")
            return {"success": False, "message": f"Error submitting Slurm job: {e.stderr}"}
        except Exception as e:
            self.logger.error(f"Unexpected error submitting worker job: {str(e)}")
            return {"success": False, "message": f"Unexpected error: {str(e)}"}
    
    def get_worker_jobs(self, context: Optional[Dict] = None) -> Dict:
        """Get all Ray worker jobs for the current user
        
        Args:
            context: RPC context
            
        Returns:
            Dict with Ray worker job information
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

            return {"success": True, "ray_worker_jobs": jobs, "worker_count": len(jobs)}

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error getting job status: {e.stderr}")
            return {
                "success": False,
                "message": f"Error getting job status: {e.stderr}",
                "ray_worker_jobs": [],
            }
        except Exception as e:
            self.logger.error(f"Unexpected error getting job status: {str(e)}")
            return {
                "success": False,
                "message": f"Unexpected error: {str(e)}",
                "ray_worker_jobs": [],
            }
    
    def cancel_worker_jobs(
        self, job_ids: Optional[List[str]] = None, context: Optional[Dict] = None
    ) -> Dict:
        """Cancel Ray worker jobs for the current user
        
        Args:
            job_ids: Optional list of specific job IDs to cancel. If None or empty, 
                    cancels all Ray worker jobs
            context: RPC context
            
        Returns:
            Dict with cancellation result information
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
            self.logger.info(f"Cancelling {len(jobs_to_cancel)} Ray worker jobs: {job_id_str}")

            if jobs_to_cancel:
                subprocess.run(
                    ["scancel", *jobs_to_cancel], check=True
                )

            return {
                "success": True,
                "message": f"Successfully cancelled {len(jobs_to_cancel)} Ray worker jobs",
                "cancelled_jobs": len(jobs_to_cancel),
                "job_ids": jobs_to_cancel,
            }

        except Exception as e:
            self.logger.error(f"Unexpected error cancelling jobs: {str(e)}")
            return {"success": False, "message": f"Unexpected error: {str(e)}"}
    
    def close_worker_node(self, node_id: str, context: Optional[Dict] = None) -> Dict:
        """Close a specific Ray worker node
        
        Args:
            node_id: ID of the Ray node to close
            context: RPC context
            
        Returns:
            Dict containing operation status and result message
        """
        try:
            # Connect to the Ray cluster
            ray.init(address="auto")

            # Get all nodes
            nodes = ray.nodes()
            target_node = None
            
            # Find the target node
            for node in nodes:
                if node["NodeID"] == node_id:
                    target_node = node
                    break
                    
            if not target_node:
                return {
                    "success": False,
                    "message": f"Node {node_id} not found in cluster"
                }
            
            # Check if the target node is the head node
            runtime_context = ray.get_runtime_context()
            if target_node["NodeManagerAddress"] == runtime_context.gcs_address:
                self.logger.warning("Cannot close head node")
                return {
                    "success": False,
                    "message": "Cannot close head node"
                }
                
            # Remove the node from the cluster
            try:
                @ray.remote
                def stop_worker():
                    """Stops the worker node by executing 'ray stop'."""
                    import subprocess
                    subprocess.run(["ray", "stop"], check=True)

                # ray.autoscaler._private.commands.kill_node(node_id)
                worker_ip = target_node["NodeManagerAddress"]
                print(f"Node IP: {worker_ip}, Resources: {target_node['Resources']}")
                obj_ref = stop_worker.options(resources={f"node:{worker_ip}": 0.01}).remote()
                ray.get(obj_ref)

                self.logger.info(f"Successfully removed node {node_id} from cluster")
                return {
                    "success": True,
                    "message": f"Successfully removed node {node_id}",
                    "node_id": node_id
                }
            except Exception as e:
                self.logger.error(f"Error removing node {node_id}: {str(e)}")
                return {
                    "success": False,
                    "message": f"Error removing node: {str(e)}"
                }
                
        except Exception as e:
            self.logger.error(f"Unexpected error closing worker node: {str(e)}")
            return {"success": False, "message": f"Unexpected error: {str(e)}"}

if __name__ == "__main__":
    from time import sleep

    # Test the class
    ray_manager = RayClusterManager()

    print("===== Testing Ray cluster class =====", end="\n\n")

    # Make sure Ray cluster is not running
    ray_manager.shutdown_cluster()

    # Check Ray cluster status
    print("Checking Ray cluster status...")
    status = ray_manager.check_cluster()
    print(status, end="\n\n")

    # Test starting Ray cluster
    print("Starting Ray cluster...")
    start_result = ray_manager.start_cluster(force=True)
    print(start_result, end="\n\n")

    # Test getting job status
    print("Getting worker job status...")
    status_result = ray_manager.get_worker_jobs()
    print(status_result, end="\n\n")

    # Test submitting a worker job
    print("Submitting a worker job...")
    submit_result = ray_manager.submit_worker_job()
    print(submit_result, end="\n\n")

    # Test getting job status
    print("Waiting for submitted worker job to start...")
    started = False
    while not started:
        status_result = ray_manager.get_worker_jobs()
        print(status_result, end="\n\n")
        started = any(job['state'] == 'RUNNING' for job in status_result.get('ray_worker_jobs', []))
        sleep(1)

    # Test getting node IDs
    print("Getting worker node IDs...")
    node_ids = []
    while not node_ids:
        print("Waiting for worker nodes...")
        status = ray_manager.check_cluster()
        # node_ids = status.get("worker_node_ids", [])
        sleep(1)
    print(status, end="\n\n")

    # # Test closing a worker node
    print("Closing a worker node...")
    close_result = ray_manager.close_worker_node(node_ids[0])
    print(close_result, end="\n\n")

    # Test cancelling all jobs
    print("Cancelling all worker jobs...")
    cancel_result = ray_manager.cancel_worker_jobs()
    print(cancel_result, end="\n\n")

    # Test shutting down Ray cluster
    print("Shutting down Ray cluster...")
    shutdown_result = ray_manager.shutdown_cluster()
    print(shutdown_result, end="\n\n")
