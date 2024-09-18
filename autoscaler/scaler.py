from config import Config
from status import Status
import terminal
import logging
import time

class Scaler:
    def __init__(self):
        self.status = Status()

    def handle_workers(self):
        if not self.status.is_worker_queue_full() and self.status.need_more_workers():
            terminal.launch_worker_node()

    def loop_step(self):
        self.status.update()
        logging.info(f"Status: {self.status}")
        self.handle_workers()
        time.sleep(Config.AUTOSCALER_CHECK_INTERVAL)

    def run(self):
        logging.info(Config.Logging.START_MSG)
        while True:
            self.loop_step()
