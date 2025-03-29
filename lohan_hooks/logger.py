import logging
import sys

log_levels = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


def create_logger(name=None, level=logging.INFO):
    if name is None:
        raise ValueError("name for logger cannot be None")

    formatter = logging.Formatter(
        "[%(asctime)s] "
        "[%(levelname)s] "
        "[%(filename)s:%(lineno)d:%(funcName)s] %(message)s"
    )

    logger_ = logging.getLogger(name)
    logger_.setLevel(level)
    logger_.propagate = False
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger_.addHandler(ch)
    return logger_
