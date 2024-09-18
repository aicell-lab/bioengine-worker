import os
import logging

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


def _setup_log_files():
        os.makedirs(Config.LOGS_DIR, exist_ok=True)
        log_file = os.path.join(Config.LOGS_DIR, Config.AUTOSCALER_LOGS_FILENAME)
        if not os.path.exists(log_file):
            with open(log_file, 'w') as f:
                pass
        return log_file

def _clear_loggers():
    # Clear existing handlers to avoid duplicates
    for handler in logging.root.handlers[:]:
        handler.close()
        logging.root.removeHandler(handler)

def _setup_logging():
        _clear_loggers()
        log_file = _setup_log_files()
        logging.basicConfig(
        level=Config.Logging.LEVEL,
        format=Config.Logging.FORMAT,
        datefmt=Config.Logging.DATE_FORMAT,
        handlers=[
            logging.FileHandler(log_file, mode='a'),  # Ensure append mode
            logging.StreamHandler()
            ]
        )

def shutdown_singletons():
    _clear_loggers()

def init_singletons():
     _setup_logging()

