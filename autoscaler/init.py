import os
import logging
from config import Config
import terminal
import sys
import ray

## Initialize and close global resources such as singletons.

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

def _connect_to_head_node() -> bool:
    result = False
    try:
        ray.init(address="auto")
        result = True
    except ray.exceptions.RayConnectionError as e:
        print(f"Failed to connect to Ray cluster: {e}", file=sys.stderr)
    except TimeoutError as e:
        print(f"Connection attempt timed out: {e}", file=sys.stderr)
    except Exception as e:
        print(f"An error occurred during Ray initialization: {e}", file=sys.stderr)
    return result

def shutdown():
    _clear_loggers()

def setup() -> bool:
     result = True
     _setup_logging()
     if not _connect_to_head_node():
          print(f"No head node detected. Launch a Ray head node before running this autoscaler.", file=sys.stderr)
          result = False
     return result

