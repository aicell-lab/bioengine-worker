from datetime import timedelta
from pathlib import Path
import util

class Config:
    class Shell:
        script_directory_name = "scripts"
        script_directory_path = util.get_dir_path(relative_path=script_directory_name)
        worker_script_name = "worker.sh"
        worker_script_path = util.get_script_path(script_dir=script_directory_name,script_filename=worker_script_name)

    class Scaling:
        MAX_NODES = 94
        CHECK_INTERVAL = timedelta(seconds=10)
        ZOMBIE_TIMEOUT = timedelta(seconds=60, minutes=0)
    
    class Logging:
        LEVEL = "DEBUG" 
        FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
        INITIAL_LOG_FORMAT = "Log file created on {timestamp}.\n"
        INITIAL_LOG_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
        LOGS_DIR = Path.home() / "logs"
        AUTOSCALER_LOGS_FILENAME = "autoscaler.out"

    class Workspace:
        server_url = "https://hypha.aicell.io"
        workspace_name = "hpc-ray-cluster"
        service_id = "ray"
        client_id = "berzelius"
        service_name = "Ray"
        TOKEN_VAR_NAME = "hypha_token_env_name"

    class Head:
        port=6379
        ip=util.get_head_LAN_IP()
        address=f"ray://{ip}:{port}"
        num_cpus=0
        num_gpus=0


