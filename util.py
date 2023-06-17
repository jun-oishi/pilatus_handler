#!/usr/bin/python
# -*- coding: utf-8 -*-

import json
from logging import getLogger as __getLogger
from logging import StreamHandler as __StreamHandler
from logging import Formatter as __Formatter
from logging import config as logging_config
from logging import DEBUG, INFO, WARNING, ERROR, CRITICAL

# __LOG_CONFIG_FILE = "config/log_config.json"

# with open(__LOG_CONFIG_FILE, "r") as f:
#     logging_config.dictConfig(json.load(f))

__FORMATTER = __Formatter(
    "%(name)s [%(levelname)s]: %(message)s from %(filename)s:%(lineno)d"
)


def getLogger(name, level=WARNING):
    logger = __getLogger(name)
    logger.setLevel(level)
    handler = __StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(__FORMATTER)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
