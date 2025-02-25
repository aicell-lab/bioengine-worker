import argparse
import asyncio

from hpc_worker.register_worker import register_hpc_worker

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Register HPC worker service."
    )
    
    # Add config file option
    parser.add_argument(
        '--config_file',
        help='Path to YAML configuration file',
    )
    
    # Make other arguments optional
    parser.add_argument(
        "--server_url",
        default="https://hypha.aicell.io",
        help="URL of the Hypha server",
    )
    parser.add_argument(
        "--num_gpu",
        default=3,
        type=int,
        help="Number of available GPUs",
    )
    parser.add_argument(
        '--dataset_paths',
        nargs='+',
        help='Paths to dataset directories'
    )
    parser.add_argument(
        '--trusted_models',
        nargs='+',
        default=['ghcr.io/aicell-lab/tabula:0.1.1'],
        help='List of trusted docker images'
    )
    
    args = parser.parse_args()
    
    # Validate that either config_file or required args are provided
    if not args.config_file and not args.dataset_paths:
        parser.error("Either --config_file or --dataset_paths must be provided")

    loop = asyncio.get_event_loop()
    loop.create_task(register_hpc_worker(args=args))
    loop.run_forever()