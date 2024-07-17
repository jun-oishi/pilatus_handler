#! /usr/bin/python3

from logging import getLogger as __getLogger
from logging import StreamHandler as __StreamHandler
from logging import Formatter as __Formatter
from logging import config as logging_config
from logging import DEBUG, INFO, WARNING, ERROR, CRITICAL


__FORMATTER = __Formatter(
    "%(name)s [%(levelname)s]: %(message)s from %(filename)s:%(lineno)d"
)

__DEFAULT_LEVEL = DEBUG


def getLogger(name: str, level=__DEFAULT_LEVEL):
    logger = __getLogger(name=name)
    logger.setLevel(level)
    handler = __StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(__FORMATTER)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
