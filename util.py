#!/usr/bin/python
# -*- coding: utf-8 -*-

from logging import getLogger as __getLogger
from logging import StreamHandler as __StreamHandler
from logging import Formatter as __Formatter
from logging import config as logging_config
from logging import DEBUG, INFO, WARNING, ERROR, CRITICAL
import os

__version__ = "0.0.1"

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


def listFiles(dir: str, *, ext="") -> list[str]:
    """指定ディレクオリ直下のファイル名をソートして返す
    'common_001.ext'のようなファイル名を想定してハイフン後の数字でソートする
    extが指定されている場合はその拡張子のみを対象とする
    返すのはファイル名のみでdirは含まない
    """
    all = os.listdir(dir)
    files = list(filter(lambda x: x.endswith(ext), all))
    getNum = lambda s: int(s.split(".")[0].split("_")[-1])
    return sorted(files, key=getNum)
