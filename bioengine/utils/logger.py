import logging
from pathlib import Path
from typing import Optional, Union

stream_logging_format = "\033[36m%(asctime)s\033[0m - \033[32m%(name)s\033[0m - \033[1;33m%(levelname)s\033[0m - %(message)s"
file_logging_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
date_format = "%Y-%m-%d %H:%M:%S %Z"


def create_logger(
    name: str, level: int = logging.INFO, log_file: Optional[Union[str, Path]] = None
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Create console handler with a custom format
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(fmt=stream_logging_format, datefmt=date_format)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        log_dir = Path(log_file).resolve().parent
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(fmt=file_logging_format, datefmt=date_format)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
