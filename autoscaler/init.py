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
        logging.getLogger("websockets.client").setLevel(logging.WARNING)


def _connect_to_ray_head() -> bool:
    result = False
    logging.info(f"connecting to ray head... address: {Config.Head.address}")
    try:
        ray.init(address="auto")
        result = True
    except TimeoutError as e:
        logging.error(f"Connection attempt timed out: {e}")
    except Exception as e:
        logging.error(f"An error occurred during Ray initialization: {e}")
    return result

def shutdown():
    _clear_loggers()
    ZombieTerminator.terminate_all_jobs()

def setup() -> bool:
    _setup_logging()

    if not hypha.token_init.set_token():
        return False

    if not _connect_to_ray_head():
        logging.error("Unable to connect to ray server.")
        return False
    
    return True

