#! /usr/bin/env python3

import numpy as np
from numba import jit
from larch import Group, xafs
from .io import label_path

def merge(groups):
    """muを持つGroupのリストを結合する"""
    energy = groups[0].energy
    mu = np.empty((len(groups), len(energy)), dtype=float)
    for i, group in enumerate(groups):
        if len(energy) != len(group.energy):
            raise ValueError("energy length mismatch")
        elif not np.allclose(energy, group.energy):
            raise ValueError("energy mismatch")
        elif not hasattr(group, "mu") or group.mu.size != len(energy):
            raise ValueError("mu mismatch")
        mu[i] = group.mu
    mu_mean = np.nanmean(mu, axis=0)
    merged = Group(energy=energy, mu=mu_mean)
    return merged

def feffit(params, dataset):
    """feffitを実行して最適パラメタをセットしたパスをfeffitの戻り値に格納して返す"""
    trans = dataset.transform
    pathlist = dataset.pathlist[:]
    out = xafs.feffit(params, dataset)

    # 最適化されたパラメタを変数に展開
    param_names = out.paramgroup.__dir__()  # type: ignore
    for name in param_names:
        definition = f'{name} = {getattr(out.paramgroup, name).value}' # type: ignore
        exec(definition)

    # パスのパラメタを最適化されたパラメタに置き換えてフーリエ変換
    ft_param = {
        'kmin': trans.kmin,
        'kmax': trans.kmax,
        'kweight': trans.kweight,
        'dk': trans.dk,
        'dk2': trans.dk2,
        'window': trans.window
    }
    is_numeric = lambda x: isinstance(x, (int, float))
    ret = []
    for path in pathlist:
        label_path(path)
        reff = path.reff
        for param in ('s02', 'deltar', 'sigma2', 'e0'):
            val = getattr(path, param)
            if not is_numeric(val):
                setattr(path, param, eval(val))
        xafs.xftf(k=path.k, chi=path.chi, group=path, **ft_param)
        ret.append(path)

    setattr(out, 'pathlist', ret)
    return out

# TODO: 原子ペアと距離からfeff入力を生成してfeffを実行してpathを取得する関数を作る
