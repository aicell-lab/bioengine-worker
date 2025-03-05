import logging
import os
import ray
import subprocess
import tempfile
import time
from typing import Dict, Optional


def check_ray_cluster() -> Dict:
    """Check ray cluster status using Ray's Python API"""
    try:
        was_connected = ray.is_initialized()
        if not was_connected:
            try:
                ray.init(address="auto")
            except ConnectionError:
                # This is an expected state - no Ray cluster running
                return {"head_running": False, "worker_count": 0}

        # Parse output from ray.nodes() which gives more detailed info
        nodes = ray.nodes()
        is_head_running = any(node["Alive"] for node in nodes)
        worker_count = sum(
            1 for node in nodes if node["Alive"] and not node.get("IsSyncPoint", False)
        )

        # Disconnect only if we connected in this function
        if not was_connected:
            ray.shutdown()

        return {
            "head_running": is_head_running,
            "worker_count": max(0, worker_count - 1),  # Subtract 1 to exclude head node
        }
    except Exception as e:
        # Log only unexpected errors
        if not str(e).startswith("Could not find any running Ray instance"):
            logger = logging.getLogger(__name__)
            logger.error(f"Error checking ray cluster: {str(e)}")

        # Make sure to disconnect even if there was an error
        if ray.is_initialized():
            ray.shutdown()
        return {"head_running": False, "worker_count": 0}


def start_ray_cluster(
    logger: Optional[logging.Logger] = None, context: dict = None
) -> Dict:
    """Start a Ray cluster head node"""
    if logger is None:
        logger = logging.getLogger(__name__)

    try:
        # Check if Ray is already running
        ray_status = check_ray_cluster()
        if ray_status["head_running"]:
            return {"success": True, "message": "Ray cluster is already running"}

        # Start ray as the head node with the specified parameters using correct arguments
        ray.init(
            address="local",  # Force creating a new Ray instance
            num_cpus=0,
            num_gpus=0,
            include_dashboard=False,
        )

        # Wait a moment for initialization
        time.sleep(2)

        # Verify the cluster is running
        runtime_context = ray.get_runtime_context()
        host_ip, port = runtime_context.gcs_address.split(":")

        return {
            "success": True,
            "message": f"Ray cluster started successfully on {host_ip}:{port}",
        }

    except Exception as e:
        logger.error(f"Error initializing Ray: {str(e)}")
        return {"success": False, "message": f"Error initializing Ray: {str(e)}"}


def shutdown_ray_cluster(
    logger: Optional[logging.Logger] = None, context: dict = None
) -> Dict:
    """Shutdown the Ray cluster and all its nodes"""
    if logger is None:
        logger = logging.getLogger(__name__)

    try:
        # Check if Ray is running
        ray_status = check_ray_cluster()
        if not ray_status["head_running"]:
            return {"success": True, "message": "Ray cluster is not running"}

        # Connect to the cluster if not already connected
        if not ray.is_initialized():
            ray.init(address="auto")

        # Get node info before shutdown for reporting
        nodes = ray.nodes()
        node_count = sum(1 for node in nodes if node["Alive"])

        # Perform the shutdown
        ray.shutdown()

        # Wait a moment for shutdown to complete
        time.sleep(2)

        # Verify shutdown was successful
        ray_status = check_ray_cluster()
        if not ray_status["head_running"]:
            logger.info(f"Successfully shut down Ray cluster with {node_count} node(s)")
            return {
                "success": True,
                "message": f"Successfully shut down Ray cluster with {node_count} node(s)",
            }
        else:
            logger.warning("Ray cluster is still running after shutdown attempt.")
            return {
                "success": False,
                "message": "Failed to shut down Ray cluster completely",
            }

    except Exception as e:
        logger.error(f"Error shutting down Ray cluster: {str(e)}")
        return {"success": False, "message": f"Error during shutdown: {str(e)}"}


def submit_ray_worker_job(
    num_gpus: int = 1,
    num_cpus: int = 4,
    mem_per_cpu: int = 8,
    time_limit: str = "1:00:00",
    container_image: str = "chiron_worker_0.1.0.sif",
    logger: Optional[logging.Logger] = None,
    context: dict = None,
) -> Dict:
    """Submit a Slurm job to start a Ray worker directly with an apptainer container

    Args:
        num_gpus: Number of GPUs to allocate to the worker
        num_cpus: Number of CPUs to allocate to the worker
        mem_per_cpu: Memory per CPU in GB
        time_limit: Time limit for the job in HH:MM:SS format
        container_image: Apptainer/Singularity container image to run
        logger: Logger instance
        context: RPC context

    Returns:
        Dict with job submission status
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    try:
        # First check if Ray cluster is running
        ray_status = check_ray_cluster()
        if not ray_status["head_running"]:
            return {
                "success": False,
                "message": "Cannot start worker: Ray head node is not running",
            }

        # Get Ray GCS address if not already connected
        was_connected = ray.is_initialized()
        if not was_connected:
            ray.init(address="auto")

        # Get the head node IP and port
        runtime_context = ray.get_runtime_context()
        head_ip, ray_port = runtime_context.gcs_address.split(":")

        # Clean up Ray connection if we initiated it
        if not was_connected:
            ray.shutdown()

        # Create logs directory if it doesn't exist
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        logs_dir = os.path.join(base_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)

        # Define the Ray worker command that will run inside the container
        ray_worker_cmd = f"ray start --address={head_ip}:{ray_port} --num-cpus={num_cpus} --num-gpus={num_gpus} --block"

        # Define the apptainer command with the Ray worker command
        apptainer_cmd = f"apptainer run --contain --writable-tmpfs --nv {container_image} {ray_worker_cmd}"

        # Create a temporary batch script
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".sh", delete=False
        ) as batch_file:
            batch_script = f"""
            #!/bin/bash
            #SBATCH --job-name=ray_worker
            #SBATCH --ntasks=1
            #SBATCH --nodes=1
            #SBATCH --gpus={num_gpus}
            #SBATCH --cpus-per-task={num_cpus}
            #SBATCH --mem-per-cpu={mem_per_cpu}G
            #SBATCH --time={time_limit}
            #SBATCH --output={logs_dir}/%x_%j.out
            #SBATCH --error={logs_dir}/%x_%j.err

            # Print some diagnostic information
            echo "Starting Ray worker node"
            echo "Host: $(hostname)"
            echo "Date: $(date)"
            echo "Connecting to head node: {head_ip}:{ray_port}"
            echo "GPUs: {num_gpus}, CPUs: {num_cpus}"
            echo "GPU info: $(nvidia-smi -L)"

            # Run the apptainer container with Ray worker
            {apptainer_cmd}

            # Print completion status
            echo "Ray worker job completed with status $?" 
            """
            for line in batch_script.split("\n"):
                # Remove leading whitespace and line breaks
                line = line.strip()
                if line:
                    batch_file.write(line + "\n")
            temp_script_path = batch_file.name

        # Create logs directory if it doesn't exist
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        logs_dir = os.path.join(base_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)

        # Submit the job with sbatch
        result = subprocess.run(
            ["sbatch", temp_script_path], capture_output=True, text=True, check=True
        )

        # Clean up the temporary file
        try:
            os.remove(temp_script_path)
        except:
            logger.warning(f"Failed to remove temporary script: {temp_script_path}")

        # Parse job ID from Slurm output (usually "Submitted batch job 12345")
        job_id = None
        if result.stdout and "Submitted batch job" in result.stdout:
            job_id = result.stdout.strip().split()[-1]

        logger.info(
            f"Worker job submitted successfully. Job ID: {job_id}, Resources: {num_gpus} GPU(s), "
            f"{num_cpus} CPU(s), {mem_per_cpu}G mem/CPU, {time_limit} time limit"
        )
        logger.info(f"Worker command: {apptainer_cmd}")

        return {
            "success": True,
            "message": f"Worker job submitted successfully with job ID {job_id}",
            "job_id": job_id,
            "head_node": f"{head_ip}:{ray_port}",
            "resources": {
                "gpus": num_gpus,
                "cpus": num_cpus,
                "mem_per_cpu": f"{mem_per_cpu}G",
                "time_limit": time_limit,
                "container": container_image,
            },
        }

    except subprocess.CalledProcessError as e:
        logger.error(f"Error submitting Slurm job: {e.stderr}")
        return {"success": False, "message": f"Error submitting Slurm job: {e.stderr}"}
    except Exception as e:
        logger.error(f"Unexpected error submitting worker job: {str(e)}")
        return {"success": False, "message": f"Unexpected error: {str(e)}"}


def get_ray_worker_jobs(
    logger: Optional[logging.Logger] = None, context: dict = None
) -> Dict:
    """Get all Ray worker jobs for the current user

    Returns a dictionary with job information including ID, name, state,
    runtime, and time limit for all Ray worker jobs owned by the current user.

    Args:
        logger: Logger instance
        context: RPC context

    Returns:
        Dict with Ray worker job information
    """
    if logger is None:
        logger = logging.getLogger(__name__)

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
        logger.error(f"Error getting job status: {e.stderr}")
        return {
            "success": False,
            "message": f"Error getting job status: {e.stderr}",
            "ray_worker_jobs": [],
        }
    except Exception as e:
        logger.error(f"Unexpected error getting job status: {str(e)}")
        return {
            "success": False,
            "message": f"Unexpected error: {str(e)}",
            "ray_worker_jobs": [],
        }


def cancel_ray_worker_jobs(
    job_ids: Optional[list] = None,
    logger: Optional[logging.Logger] = None, 
    context: dict = None
) -> Dict:
    """Cancel Ray worker jobs for the current user

    Args:
        job_ids: Optional list of specific job IDs to cancel. If None or empty, cancels all Ray worker jobs
        logger: Logger instance
        context: RPC context

    Returns:
        Dict with cancellation result information
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    try:
        # First get all ray worker jobs
        jobs_result = get_ray_worker_jobs(logger=logger)

        if not jobs_result["success"]:
            logger.error("Failed to retrieve job list for cleanup")
            return {
                "success": False,
                "message": "Failed to retrieve job list for cleanup",
            }

        ray_jobs = jobs_result.get("ray_worker_jobs", [])

        if not ray_jobs:
            logger.info("No Ray worker jobs to clean up")
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
        logger.info(f"Cancelling {len(jobs_to_cancel)} Ray worker jobs: {job_id_str}")

        if jobs_to_cancel:
            result = subprocess.run(
                ["scancel", *jobs_to_cancel], capture_output=True, text=True
            )

            if result.returncode != 0:
                logger.error(f"Error cancelling jobs: {result.stderr}")
                return {
                    "success": False,
                    "message": f"Error cancelling jobs: {result.stderr}",
                }

        return {
            "success": True,
            "message": f"Successfully cancelled {len(jobs_to_cancel)} Ray worker jobs",
            "cancelled_jobs": len(jobs_to_cancel),
            "job_ids": jobs_to_cancel,
        }

    except Exception as e:
        logger.error(f"Unexpected error cancelling jobs: {str(e)}")
        return {"success": False, "message": f"Unexpected error: {str(e)}"}


if __name__ == "__main__":
    # Test the functions
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    print("===== Testing Ray cluster functions =====", end="\n\n")

    # Check Ray cluster status
    logger.info("Checking Ray cluster status")
    status = check_ray_cluster()
    print(status, end="\n\n")

    # Test starting Ray cluster
    start_result = start_ray_cluster(logger=logger)
    print(start_result, end="\n\n")

    # Test getting job status
    logger.info("Checking ray worker job status")
    status_result = get_ray_worker_jobs(logger=logger)
    print(status_result, end="\n\n")

    # Test submitting a worker job
    submit_result = submit_ray_worker_job(logger=logger)
    print(submit_result, end="\n\n")

    submit_result = submit_ray_worker_job(logger=logger)
    print(submit_result, end="\n\n")

    # Test cancelling all jobs
    cancel_result = cancel_ray_worker_jobs(logger=logger)
    print(cancel_result, end="\n\n")

    # Test shutting down Ray cluster
    shutdown_result = shutdown_ray_cluster(logger=logger)
    print(shutdown_result, end="\n\n")
