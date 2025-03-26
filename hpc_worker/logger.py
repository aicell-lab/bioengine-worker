import logging

def create_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "\033[36m%(asctime)s\033[0m - \033[32m%(name)s\033[0m - \033[1;33m%(levelname)s\033[0m - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger
