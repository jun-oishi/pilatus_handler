#!/usr/bin/python3

import numpy as np
from numba import jit

# @jit(nopython=True, cache=True)
def convert(raw, dtype=np.uint8, *, min_val=np.nan, max_val=np.nan, zero_shift=True):
    """dtype型の配列に変換する
    min,maxが指定されていればそれに合わせて正規化する
    指定されていなければ自動で最大を取得しzero_shiftがあれば最小が0になるようにシフトして正規化する
    """
    if not dtype in (np.uint8, np.uint16, np.uint32):
        raise ValueError(f"dtype must be np.uint8, np.uint16, or np.uint32, but {dtype} is given.")
    if np.isnan(min_val):
        if zero_shift:
            min_val = raw.min()
        else:
            min_val = 0
    if np.isnan(max_val):
        max_val = raw.max()

    if dtype == np.uint8:
        width = 2**8 - 1
    elif dtype == np.uint16:
        width = 2**16 - 1
    elif dtype == np.uint32:
        width = 2**32 - 1

    return ((raw - min_val) / (max_val - min_val) * width).astype(dtype)

@jit(nopython=True, cache=True)
def ev2nm(ev):
    """エネルギー[eV]を波長[nm]に変換する"""
    return 1240 / ev

@jit(nopython=True, cache=True)
def nm2ev(nm):
    """波長[nm]をエネルギー[eV]に変換する"""
    return 1240 / nm

@jit(nopython=True, cache=True)
def q2theta(q, wavelength, unit="degree"):
    """散乱ベクトルq[nm^-1]を散乱角θに変換する"""
    theta = np.arcsin(q * wavelength / (4 * np.pi))
    return np.rad2deg(theta) if unit == "degree" else theta

@jit(nopython=True, cache=True)
def theta2q(theta, wavelength, unit="degree"):
    """散乱角θを散乱ベクトルq[nm^-1]に変換する"""
    theta = np.deg2rad(theta) if unit == "degree" else theta
    return 4 * np.pi * np.sin(theta) / wavelength
