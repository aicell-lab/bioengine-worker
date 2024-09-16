import os

class Config:
    MAX_NODES = 94
    LOGS_DIR = os.path.expanduser("~/logs")
    AUTOSCALER_LOGS_FILENAME = "autoscaler.out"
    
    class Logging:
        LEVEL = "DEBUG" 
        FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
        INITIAL_LOG_FORMAT = "Log file created on {timestamp}.\n"
        INITIAL_LOG_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
        START_MSG = "Starting autoscaler..."