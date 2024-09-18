from config import Config
import status
import terminal

import os
import logging
import time 

class Autoscaler:
    def __init__(self):
        self.setup_logging()

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

    def is_worker_queue_full(self) -> bool:
        if status.get_num_slurm_jobs() >= Config.MAX_NODES:
            return True
        if status.get_num_slurm_pending_jobs() > 0:
            return True # Avoid spamming slurm jobs
        return False

    def handle_workers(self):
        if self.is_worker_queue_full():
            return
        if status.get_num_ray_jobs() > status.get_num_slurm_jobs():
            terminal.launch_worker_node()

    def log_slurm_status(self):
        logging.info(f"Pending SLURM jobs: {status.get_num_slurm_pending_jobs()}")
        logging.info(f"Running SLURM jobs: {status.get_num_slurm_running_jobs()}")
        logging.info(f"Total SLURM jobs: {status.get_num_slurm_jobs()}")

    def log_ray_status(self):
        logging.info(f"Pending Ray jobs: {status.get_num_ray_pending_jobs()}")
        logging.info(f"Running Ray jobs: {status.get_num_ray_running_jobs()}")
        logging.info(f"Total Ray jobs: {status.get_num_ray_jobs()}")

    def log_status(self):
        self.log_slurm_status()
        self.log_ray_status()

    def loop_step(self):
        self.handle_workers()
        self.log_status()
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
        

if __name__ == "__main__":
    autoscaler = Autoscaler()
    autoscaler.run()

