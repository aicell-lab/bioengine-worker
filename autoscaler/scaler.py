from config import Config
from status import Status
import terminal
import os
import logging
import time 

class Scaler:
    def __init__(self):
        self.setup_logging()
        self.status = Status()

    def setup_log_files(self):
        os.makedirs(Config.LOGS_DIR, exist_ok=True)
        log_file = os.path.join(Config.LOGS_DIR, Config.AUTOSCALER_LOGS_FILENAME)
        if not os.path.exists(log_file):
            with open(log_file, 'w') as f:
                pass
        return log_file

    def setup_logging(self):
        log_file = self.setup_log_files()
        # Clear existing handlers to avoid duplicates
        for handler in logging.root.handlers[:]:
            handler.close() 
            logging.root.removeHandler(handler)
        logging.basicConfig(
            level=Config.Logging.LEVEL,
            format=Config.Logging.FORMAT,
            datefmt=Config.Logging.DATE_FORMAT,
            handlers=[
                logging.FileHandler(log_file, mode='a'),  # Ensure append mode
                logging.StreamHandler()
            ]
        )

    def handle_workers(self):
        if not self.status.is_worker_queue_full() and self.status.need_more_workers():
            terminal.launch_worker_node()

    def loop_step(self):
        self.status.update()
        logging.info(f"Status: {self.status}")
        self.handle_workers()
        time.sleep(Config.AUTOSCALER_CHECK_INTERVAL)

    def on_shutdown(self):
        for handler in logging.root.handlers[:]:
            handler.close()
            logging.root.removeHandler(handler)

    def run(self):
        logging.info(Config.Logging.START_MSG)
        try:
            while True:
                self.loop_step()
        finally:
            self.on_shutdown()
