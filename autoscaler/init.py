import os
import logging
from config import Config
import ray
from scaling.zombie import ZombieTerminator
import hypha.token_init
import terminal

## Initialize and close global resources

def _setup_log_files():
        os.makedirs(Config.Logging.LOGS_DIR, exist_ok=True)
        log_file = os.path.join(Config.Logging.LOGS_DIR, Config.Logging.AUTOSCALER_LOGS_FILENAME)
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
        logging.getLogger("urllib3").setLevel(logging.WARNING)


def _start_head_node() -> bool:
    result = False
    try:
        ray.init(
            address=Config.Head.address,
            num_cpus=Config.Head.num_cpus,
            num_gpus=Config.Head.num_gpus,
        )
        result = True
    except ray.exceptions.RayConnectionError as e:
        logging.error(f"Failed to connect to Ray cluster: {e}")
    except TimeoutError as e:
        logging.error(f"Connection attempt timed out: {e}")
    except Exception as e:
        logging.error(f"An error occurred during Ray initialization: {e}")
    return result

def shutdown():
    _clear_loggers()
    ZombieTerminator.terminate_all_jobs()

def _stop_existing_ray_head():
     terminal.run_command(["ray", "stop"])

def setup() -> bool:
    _setup_logging()
    _stop_existing_ray_head()

    if not hypha.token_init.set_token():
        return False

    if not _start_head_node():
        logging.error("Unable to start ray server.")
        return False
    
    return True

