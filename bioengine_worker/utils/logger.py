import logging

logging_format = "\033[36m%(asctime)s\033[0m - \033[32m%(name)s\033[0m - \033[1;33m%(levelname)s\033[0m - %(message)s"
date_format = "%Y-%m-%d %H:%M:%S"


def create_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(fmt=logging_format, datefmt=date_format)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger
