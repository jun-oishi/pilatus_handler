#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from typing import TypeAlias, Union
from numbers import Number

from SpectraSpark.util.io import listFiles, loadtxt, savetxt, write_json, read_json
from SpectraSpark.util.basic_calculation import *

ArrayLike: TypeAlias = Union[list[float], tuple[float, ...], np.ndarray]

def is_numeric(x) -> bool:
    if isinstance(x, Number) and np.isfinite(x):
        return True
    else:
        return False

__all__ = [
    "ArrayLike",
    "listFiles",
    "loadtxt",
    "savetxt",
    "write_json",
    "read_json",
]
