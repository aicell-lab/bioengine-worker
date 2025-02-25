import logging
import os
import time
from functools import partial

from dotenv import find_dotenv, load_dotenv
from hypha_rpc import connect_to_server, login

from hpc_worker.config_manager import create_worker_config
from hpc_worker.worker_status import worker_status

# Load environment variables
ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)

logger = logging.getLogger(__name__)
logger.setLevel("INFO")
# Disable propagation to avoid duplication of logs
logger.propagate = False
# Create a new console handler
console_handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Set the base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


async def register_hpc_worker(args):
    config_path = args.config_file or os.path.join(os.path.dirname(__file__), 'worker_config.yaml')
    
    if not args.config_file:
        create_worker_config(
            dataset_paths=args.dataset_paths,
            max_gpus=args.num_gpu,
            trusted_models=args.trusted_models,
            config_path=config_path
        )

    registered_at = int(time.time())
    
    # Create worker_status with bound arguments
    status_func = partial(worker_status, config_path=config_path, registered_at=registered_at)

    hypha_token = os.environ.get("HYPHA_TOKEN") or await login({"server_url": args.server_url})
    server = await connect_to_server({"server_url": args.server_url, "token": hypha_token})

    # Register status service
    service_info = await server.register_service({
        "name": "HPC Worker",
        "id": "hpc-worker",
        "config": {"visibility": "public"},
        "get_status": status_func
    })
    logger.info(f"Service registered with ID: {service_info.id}")
    sid = service_info.id.split("/")[1]
    service_url = f"{args.server_url}/{server.config.workspace}/services/{sid}"
    logger.info(f"Test the HPC worker service here: {service_url}/get_status")

    await server.serve()
