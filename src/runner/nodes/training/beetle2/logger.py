import logging

from shared.logging_setup import configure_logging as configure_shared_logging


LOGGER_NAME = "runner.nodes.training.beetle2"
logger = logging.getLogger(LOGGER_NAME)


def configure_logging() -> None:
    configure_shared_logging(LOGGER_NAME)
