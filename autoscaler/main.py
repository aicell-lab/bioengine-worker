from config import Config
import os
import logging
from datetime import datetime

class Autoscaler:
    def __init__(self):
        self.setup_logging()

    def setup_log_files(self):
        os.makedirs(Config.LOGS_DIR, exist_ok=True)
        log_file = os.path.join(Config.LOGS_DIR, Config.AUTOSCALER_LOGS_FILENAME)
        if not os.path.exists(log_file):
            with open(log_file, 'w') as f:
                timestamp = datetime.now().strftime(Config.Logging.INITIAL_LOG_TIMESTAMP_FORMAT)
                initial_log_entry = Config.Logging.INITIAL_LOG_FORMAT.format(timestamp=timestamp)
                f.write(initial_log_entry)
        return log_file

    def setup_logging(self):
        logging.basicConfig(
            level=Config.Logging.LEVEL,
            format=Config.Logging.FORMAT,
            datefmt=Config.Logging.DATE_FORMAT,
            handlers=[
                logging.FileHandler(self.setup_log_files()),
                logging.StreamHandler() 
            ]
        )

    def run(self):
        self.logger.info(Config.Logging.START_MSG)
        

if __name__ == "__main__":
    autoscaler = Autoscaler()
    autoscaler.run()

