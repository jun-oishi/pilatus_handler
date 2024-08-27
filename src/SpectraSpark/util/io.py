#! /usr/bin/python3

import os
import json
import numpy as np
import json

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

def _format_for_json(data, special_float_to=None):
    """nan, infをjsonで書き込めるように変換する"""
    for key in data:
        if isinstance(data[key], dict):
            data[key] = _format_for_json(data[key], special_float_to)
        elif isinstance(data[key], float) and not np.isfinite(data[key]):
            data[key] = special_float_to
        elif hasattr(data[key], "__len__") and isinstance(data[key][0], float):
            data[key] = [special_float_to if not np.isfinite(x) else x for x in data[key]]
    return data

def write_json(path: str, data: dict, indent=2):
    """jsonファイルに書き込む"""
    data = _format_for_json(data, special_float_to=None)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent)