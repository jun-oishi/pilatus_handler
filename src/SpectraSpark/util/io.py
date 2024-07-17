#! /usr/bin/python3

import os
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


class CsvFile:
    def __init__(self, data: np.ndarray, header: list[str]):
        self.data: np.ndarray = data
        self.header: list[str] = header


def loadCsv(
    path: str,
    *,
    n_columns: int = 0,
    usecols: tuple[int, ...] = (),
    dtype=float,
    comment: str | int = "#",
    delimiter: str | None = None,
) -> CsvFile:
    """csvファイルを読み込む

    Parameters
    ----------
    path : str
        読み込むファイルのパス
    n_columns : int
        列数(全列読み出しの場合)
    usecols : tuple[int]
        使う列の番号(選択する場合)
    dtype : DTypelike, optional
        データ型, by default np.float64
    comment : str|int, optional
        strならその文字から始まる行を自動でヘッダとみなしintなら指定の行数をヘッダとみなす, by default "#"
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} is not found")

    if n_columns == 0 and usecols is None:
        raise ValueError("n_columns or usecols must be specified")
    elif n_columns != 0 and usecols is not None:
        usecols = tuple(range(n_columns))
    n_columns = max(n_columns, max(usecols) + 1)

    skiprows: int = 0
    header = ""
    if isinstance(comment, str) and comment != "":
        c = 0
        with open(path, "r") as f:
            while True:
                line = f.readline()
                if line.startswith(comment):
                    header = line
                    c += 1
                else:
                    break
        skiprows = c
    elif comment == "":
        skiprows = 0
    else:
        skiprows = int(comment)

    if delimiter is None:
        candidates = (",", "\t", " ")
        for d in candidates:
            data = np.loadtxt(
                path,
                delimiter=d,
                skiprows=skiprows,
                usecols=usecols,
                max_rows=1,
                dtype=dtype,
            )
            if len(data) >= n_columns:
                delimiter = d
                break
        if delimiter is None:
            raise ValueError("delimiter is not found")

    try:
        data = np.loadtxt(
            path, delimiter=delimiter, skiprows=skiprows, usecols=usecols, dtype=dtype
        )
    except:
        data = np.loadtxt(
            path,
            delimiter=delimiter,
            skiprows=skiprows,
            usecols=usecols,
            dtype=dtype,
            encoding="cp932",
        )

    return CsvFile(data, header.split(delimiter))

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