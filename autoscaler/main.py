from config import Config
import os
import logging
from datetime import datetime

class Autoscaler:
    def __init__(self):
        self.setup_log_files()
        self.init_logger()

    def setup_log_files(self):
        os.makedirs(Config.LOGS_DIR, exist_ok=True)
        log_file = os.path.join(Config.LOGS_DIR, Config.AUTOSCALER_LOGS_FILENAME)
        if not os.path.exists(log_file):
            with open(log_file, 'w') as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"Log file created on {timestamp}.\n")

    def init_logger(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)        
        file_handler = logging.FileHandler(os.path.join(Config.LOGS_DIR, Config.AUTOSCALER_LOGS_FILENAME))
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.info("Autoscaler initialized.")

    def run(self):
        self.logger.info("Starting autoscaler...")
        

if __name__ == "__main__":
    autoscaler = Autoscaler()
    autoscaler.run()

