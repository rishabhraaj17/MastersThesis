import logging

default_format = "%(asctime)s.%(msecs)03d | %(levelname)-6s |  %(name)-10.30s | %(message)s"
default_date_format = "%Y-%m-%d %H:%M:%S"


def initialize_logging(lvl=logging.INFO, fmt=default_format, date_fmt=default_date_format):
    """
    Initialize logging globally. Modules have logging implemented, but do not initialize it.
    """
    logging.basicConfig(level=lvl, format=fmt, datefmt=date_fmt)


def get_logger(name):
    logger = logging.getLogger(name)
    return logger
