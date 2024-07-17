#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from typing import TypeAlias, Union
from SpectraSpark.util.io import listFiles, CsvFile, loadCsv, write_json

ArrayLike: TypeAlias = Union[list[float], tuple[float, ...], np.ndarray]

__all__ = [
    "ArrayLike",
    "listFiles",
    "CsvFile",
    "loadCsv",
    "write_json",
]
