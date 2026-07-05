import logging
import os


LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
DEFAULT_LEVEL = "INFO"


def configure_logging(service_name: str) -> logging.Logger:
    level_name = os.environ.get("RUNFLOW_LOG_LEVEL", DEFAULT_LEVEL).upper()
    levels = logging.getLevelNamesMapping()
    if level_name not in levels:
        raise ValueError(f"Unknown RUNFLOW_LOG_LEVEL: {level_name}")

    level = levels[level_name]
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=level, format=LOG_FORMAT)
    else:
        root.setLevel(level)

    logger = logging.getLogger(service_name)
    logger.setLevel(level)
    logger.info("logging configured")
    return logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
