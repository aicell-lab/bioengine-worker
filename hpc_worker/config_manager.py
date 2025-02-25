import socket
from typing import List

import yaml

def create_worker_config(dataset_paths: List[str], max_gpus: int, trusted_models: List[str], config_path: str) -> None:
    """Create minimal worker configuration"""
    config = {
        'machine_name': socket.gethostname(),
        'max_gpus': max_gpus,
        'dataset_paths': dataset_paths,
        'trusted_models': trusted_models,
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
