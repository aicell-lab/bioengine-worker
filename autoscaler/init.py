import os
import logging
from config import Config
import terminal

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

def _check_ray_status() -> bool:
    result = terminal.run_command(args=['ray', 'status'])
    return result and "ERROR" not in result

def shutdown():
    _clear_loggers()

def setup() -> bool:
     import sys
     _setup_logging()
     if not _check_ray_status():
          print(f"No head node detected. Launch a Ray head node before running this autoscaler.", file=sys.stderr)
          return False
     return True
