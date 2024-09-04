#! /usr/bin/env python3

import os
from larch import xafs
from copy import deepcopy

from .io import label_path, pair2feffinp, cif2feffinp, run_feff

def _copy_path(src):
    dst = deepcopy(src)
    if hasattr(src, "k") and hasattr(src, "chi"):
        dst.k, dst.chi = src.k.copy(), src.chi.copy()
    if hasattr(src, "r"):
        dst.r = src.r.copy()
        dst.chir_mag, dst.chir_pha = src.chir_mag.copy(), src.chir_pha.copy()
        dst.chir_re, dst.chir_im = src.chir_re.copy(), src.chir_im.copy()
    return dst

def feffit(params, dataset):
    """feffitを実行して最適パラメタをセットしたパスをfeffitの戻り値に格納して返す"""
    trans = dataset.transform
    pathlist = dataset.pathlist[:]
    out = xafs.feffit(params, dataset)

    # 最適化されたパラメタを変数に展開
    param_names = out.paramgroup.__dir__()  # type: ignore
    params_val = {name: getattr(out.paramgroup, name).value for name in param_names} # type: ignore

    # パスのパラメタを最適化されたパラメタに置き換えてフーリエ変換
    ft_param = {
        'kmin': trans.kmin,
        'kmax': trans.kmax,
        'kweight': trans.kweight,
        'dk': trans.dk,
        'dk2': trans.dk2,
        'window': trans.window
    }
    def is_numeric(x): return isinstance(x, (int, float))
    ret = []
    for base in pathlist:
        path = _copy_path(base)
        path.label = label_path(path)
        for param in ('s02', 'deltar', 'sigma2', 'e0'):
            val = getattr(base, param)
            if not is_numeric(val):
                params_val['reff'] = path.reff
                setattr(path, param, eval(val, params_val))
        xafs.xftf(k=path.k, chi=path.chi, group=path, **ft_param)
        ret.append(path)

    setattr(out, 'pathlist', ret)
    return out

def pair_feff(abs, scat, r, *, degen=1.0, folder='./feff', title='', **kwargs):
    """原子ペアと距離からfeff入力を生成してfeffを実行してpathを取得する
    see also: pair2feffinp, run_feff
    """
    if not os.path.isdir(folder):
        if os.path.exists(folder):
            raise ValueError(f"{folder} is not a directory")
        os.makedirs(folder)
    outdir = pair2feffinp(abs, scat, r, folder=folder, title=title, **kwargs)
    path = run_feff(outdir)[0]
    path.degen = degen
    label_path(path)
    return path

def cif_feff(cif, abs, radius, *, folder='./feff', abs_site=-1, **kwargs):
    """CIFファイルと吸収原子の情報からfeff入力を生成してfeffを実行してpathを取得する
    see also: cif2feffinp, run_feff
    """
    outdir = cif2feffinp(cif, abs, radius, folder=folder, abs_site=abs_site, **kwargs)
    return run_feff(outdir)