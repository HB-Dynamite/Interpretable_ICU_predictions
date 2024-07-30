import logging
import sys
import datetime
import os
from utils.config import LOG_DIR


def setup_logger(name, logfile, formatter, stream_handler=True, level=logging.DEBUG):
    """Function to create loggers."""

    file_handler = logging.FileHandler(logfile, mode="a")
    stdout_handler = logging.StreamHandler()

    file_handler.setFormatter(formatter)
    stdout_handler.setFormatter(formatter)

    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(level)
        logger.addHandler(file_handler)
        if stream_handler:
            logger.addHandler(stdout_handler)

    return logger


def update_logger(logger, level=logging.DEBUG, file_mode="a", use_new_file=True):
    """
    Update the logger's level and file handler mode.
    """
    logger.setLevel(level)

    log_directory = LOG_DIR

    new_logfile = None

    if use_new_file:
        current_datetime = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        new_logfile_name = f"logfile_{current_datetime}.log"
        new_logfile = os.path.join(log_directory, new_logfile_name)

    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.close()
            logger.removeHandler(handler)

            if new_logfile is not None:
                handler_file = new_logfile
            else:
                handler_file = handler.baseFilename

            new_file_handler = logging.FileHandler(handler_file, mode=file_mode)
            new_file_handler.setFormatter(handler.formatter)
            logger.addHandler(new_file_handler)


formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s:%(lineno)d -> %(message)s\n"
)
path = LOG_DIR / "logfile.log"
log = setup_logger("logger", path, formatter)
