import os

def get_path_n_levels_up(file_path: str, n: int) -> str:
    base_dir = os.path.abspath(file_path)
    for _ in range(n):
        base_dir = os.path.dirname(base_dir)
    return base_dir

class Config:
    MAX_NODES = 94
    LOGS_DIR = os.path.expanduser("~/logs")
    AUTOSCALER_LOGS_FILENAME = "autoscaler.out"

    WORKER_BATCH_FILENAME = "worker.sh"
    SCRIPTS_DIR = "scripts"
    CONFIG_FILE_PATH = __file__
    WORKER_SCRIPT_PATH = os.path.join(
        get_path_n_levels_up(CONFIG_FILE_PATH, 2),
        SCRIPTS_DIR,
        WORKER_BATCH_FILENAME
    )

    AUTOSCALER_CHECK_INTERVAL = 5
    
    class Logging:
        LEVEL = "DEBUG" 
        FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
        INITIAL_LOG_FORMAT = "Log file created on {timestamp}.\n"
        INITIAL_LOG_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
        START_MSG = "Starting autoscaler..."


