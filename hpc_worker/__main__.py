import argparse
import asyncio
import logging
import os
import sys
import signal
import atexit

from hpc_worker.hpc_worker import HpcWorker

# Configure logger
logger = logging.getLogger("hpc_worker")
logger.setLevel(logging.INFO)
logger.propagate = False
console_handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Global worker instance
worker = None

# Define cleanup and signal handlers
def cleanup_handler():
    """Clean up resources when the program exits"""
    global worker
    if worker:
        logger.info("Running cleanup tasks...")
        try:
            # Run the cleanup method in a new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            cleanup_result = loop.run_until_complete(worker.cleanup())
            loop.close()
            
            logger.info("Cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

def signal_handler(sig, frame):
    """Handle termination signals"""
    signal_name = signal.Signals(sig).name
    logger.info(f"Received signal {signal_name}, shutting down...")
    cleanup_handler()
    sys.exit(0)

async def main(args):
    """Main function to initialize and register HPC worker"""
    global worker
    
    # Create HPC worker instance
    worker = HpcWorker(
        config_path=args.config_file,
        server_url=args.server_url,
        num_gpu=args.num_gpu,
        dataset_paths=args.dataset_paths,
        trusted_models=args.trusted_models,
        logger=logger
    )
    
    # Register worker with Hypha
    registration_result = await worker.register()
    
    if not registration_result.get("success", False):
        logger.error(f"Failed to register worker: {registration_result.get('message', 'Unknown error')}")
        return False
    
    return True

if __name__ == "__main__":
    description = """
Register HPC worker to Chiron platform

This script registers a HPC worker service with the Hypha server, enabling:
- Remote monitoring of worker status
- Ray cluster management (start/stop)
- Submitting worker jobs to the HPC system
- Managing trusted models and datasets
- Auto-scaling of resources based on workload

Container execution:
    Run in image chiron_worker_0.1.0.sif

    Pull the image with:
    `apptainer pull chiron_worker_0.1.0.sif docker://ghcr.io/aicell-lab/chiron-worker:0.1.0`

    Run with:
    `apptainer run --contain --nv chiron_worker_0.1.0.sif python -m hpc_worker [options]`
"""

    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Config file options
    config_group = parser.add_argument_group("Configuration Options")
    config_group.add_argument(
        '--config_file',
        help='Path to YAML configuration file (if provided, other options except server_url are ignored)',
    )
    config_group.add_argument(
        '--config_dir',
        help='Directory to save configuration file (only used when config_file is not provided)',
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")
    )
    
    # Server options
    parser.add_argument(
        "--server_url",
        default="https://hypha.aicell.io",
        help="URL of the Hypha server",
    )
    
    # Worker options - required if config_file is not provided
    worker_group = parser.add_argument_group("Worker Options (required if config_file not provided)")
    worker_group.add_argument(
        "--num_gpu",
        type=int,
        help="Number of available GPUs",
    )
    worker_group.add_argument(
        '--dataset_paths',
        nargs='+',
        help='Paths to dataset directories'
    )
    worker_group.add_argument(
        '--trusted_models',
        nargs='+',
        help='List of trusted docker images'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.config_file and (args.num_gpu is None or args.dataset_paths is None or args.trusted_models is None):
        parser.error(
            "Either --config_file must be provided OR all of --num_gpu, --dataset_paths, and --trusted_models"
        )
        
    # Register cleanup handlers
    atexit.register(cleanup_handler)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start the main application
    loop = asyncio.get_event_loop()
    try:
        if loop.run_until_complete(main(args)):
            # If registration successful, keep the loop running
            loop.run_forever()
        else:
            logger.error("Failed to initialize HPC worker, exiting.")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Unhandled exception in main loop: {str(e)}")
        sys.exit(1)
    finally:
        loop.close()