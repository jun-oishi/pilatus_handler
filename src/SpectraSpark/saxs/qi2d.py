import numpy as np
import warnings
import re
import os
from numba import jit
from matplotlib import pyplot as plt
import tqdm
import cv2
from typing import Tuple, Callable

from ..util import listFiles, write_json, read_json, savetxt, is_numeric
from ..util.basic_calculation import r2q
from ..constants import DETECTER_PX_SIZES

class Saxs2dParams:
    CALIBRATION_TYPES = ('geometry', 'linear_regression', 'none')

    def __init__(self, *,
                 center_x: float=np.nan,
                 center_y: float=np.nan,
                 calibration_type:str='none',
                 px_size: float=np.nan,
                 camera_length: float=np.nan,
                 wave_length: float=np.nan,
                 slope: float=np.nan,
                 intercept: float=np.nan,
                 flip: str='',
                 mask_src:str=''):
        self.center_x = float(center_x) # [px]
        self.center_y = float(center_y) # [px]
        self.calibration_type = calibration_type
        self.px_size = float(px_size)   # [mm]
        self.camera_length = float(camera_length) # [mm]
        self.wave_length = float(wave_length)     # [AA]
        self.slope = float(slope)  # [nm^-1/px]
        self.intercept = float(intercept) # [nm^-1]
        self.flip = flip
        self.mask_src = mask_src

    @classmethod
    def load(cls, src: str):
        _params = read_json(src)
        params = {}
        for k, v in _params.items():
            if v is None or v == 'none':
                continue
            elif v == 'nan':
                params[k] = np.nan
            else:
                params[k] = v
        return cls(**params)

    def save(self, dst: str, overwrite=False):
        if not overwrite and os.path.exists(dst):
            raise FileExistsError(f"{dst} is already exists.")
        if self.calibration_type not in self.CALIBRATION_TYPES:
            raise ValueError(f"Invalid calibration type: {self.calibration_type}")
        write_json(dst, self.__dict__)

@jit(nopython=True, cache=True)
def _radial_average(img:np.ndarray, center_x:float, center_y:float, threshold:int=2)->Tuple[np.ndarray, np.ndarray]:
    """画像の中心を中心にして、動径平均を計算する

    Parameters
    ----------
    img : np.ndarray
        散乱強度の2次元配列
    center_x : int
        ビームセンターのx座標
    center_y : int
        ビームセンターのy座標
    threshold : float
        この値より小さい画素は無視する

    Returns
    -------
    r : np.ndarray
        動径[px]の配列
    i : np.ndarray
        動径方向の平均散乱強度の配列
    """
    width = img.shape[1]
    height = img.shape[0]
    r_mesh = np.empty(img.shape, dtype=np.int64)
    dx_sq = (np.arange(width) - center_x)**2
    for y in range(height):
        r_mesh[y, :] = np.floor(np.sqrt(dx_sq + (y - center_y)**2))

    r = np.arange(r_mesh.max(), dtype=np.int64)
    cnt = np.zeros_like(r, dtype=np.int64)
    i = np.zeros_like(r, dtype=np.float64)
    for x in range(width):
        for y in range(height):
            if not img[y, x] >= threshold:
                continue
            cnt[r_mesh[y,x]] += 1
            i[r_mesh[y,x]] += img[y, x]

    for _r in r:
        if cnt[_r] > 0:
            i[_r] /= cnt[_r]
        else:
            i[_r] = np.nan

    return r + 0.5, i

@jit(nopython=True, cache=True)
def _mask_and_average(img:np.ndarray, mask:np.ndarray, center_x:float, center_y:float, threshold:int=2):
    """画像をマスクして、動径平均を計算する
    _radial_average(img*mask, center_x, center_y, threshold)と同じ
    """
    return _radial_average(img*mask, center_x, center_y, threshold)

def _readmask(src:str)->np.ndarray:
    """ファイルからマスクを読み込む
    ファイルを読み込んでuint8の二次元配列にして返す
    """
    if not os.path.exists(src):
        raise FileNotFoundError(f"{src} is not found.")
    mask = cv2.imread(src, cv2.IMREAD_UNCHANGED)
    if len(mask.shape) != 2:
        raise ValueError("mask file must be 2D single-channel image")
    mask[mask > 0] = 1
    return mask.astype(np.uint8)

def _get_stats(img:np.ndarray, mask:np.ndarray, center_x:float, center_y:float, prefix:str, r2q:Callable, threshold:int=2)->Tuple[np.ndarray, np.ndarray]:
    """動径平均と分散を求めてcsv, 散布図, エラーバー付きq-iプロットを保存する
    imgをマスクして動径平均と分散を計算して{prefix}_radial.csv, {prefix}_scatter.png, {prefix}_radial.pngを保存する

    Parameters
    ----------
    img : np.ndarray
        散乱強度の2次元配列
    mask : np.ndarray
        マスク画像の配列, 0の画素は無視される
    center_x : float
        ビームセンターのx座標
    center_y : float
        ビームセンターのy座標
    prefix : str
        保存するファイル名のプレフィックス
    r2q : Callable[[np.ndarray], np.ndarray]
        動径をqに変換する関数
    threshold : int
        この値より小さい画素は無視する

    Returns
    -------
    r : np.ndarray
        r[px]の配列
    i : np.ndarray
        散乱強度の配列
    """
    img = img * mask
    figsize = (img.shape[1]//80, 8)

    width = img.shape[1]
    height = img.shape[0]
    r_mesh = np.empty(img.shape, dtype=np.int64)
    dx_sq = (np.arange(width) - center_x)**2
    for y in range(height):
        r_mesh[y, :] = np.floor(np.sqrt(dx_sq + (y - center_y)**2))
    q_mesh = r2q(r_mesh)

    # 統計量の計算
    r = np.arange(r_mesh.max(), dtype=np.int64)
    cnt = np.zeros_like(r, dtype=np.int64)
    i = np.zeros_like(r, dtype=np.float64)
    i_sq = np.zeros_like(r, dtype=np.float64)
    for x in range(width):
        for y in range(height):
            if img[y, x] < threshold:
                continue
            cnt[r_mesh[y,x]] += 1
            i[r_mesh[y,x]] += img[y, x]
            i_sq[r_mesh[y,x]] += img[y, x]**2

    for _r in r:
        if cnt[_r] > 0:
            i[_r] /= cnt[_r]
            i_sq[_r] /= cnt[_r]
        else:
            i[_r] = np.nan
            i_sq[_r] = np.nan
    i_std = np.sqrt(i_sq - i**2)
    min_q, max_q = 0, q_mesh.max()
    min_i, max_i = np.nanmin(i), np.max(img)

    # 散布図
    savetxt(f"{prefix}_scatter.csv", np.array([q_mesh.flatten(), img.flatten()]).T,
            header=["q[nm^-1]", "i"], overwrite=True)
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(q_mesh, img, s=1)
    ax.set_xlabel(r"$q\,[\mathrm{nm}^{-1}]$")
    ax.set_xlim(min_q, max_q)
    ax.set_ylabel(r"$I(q)\,[\mathrm{a.u.}]$")
    ax.set_ylim(min_i, max_i)
    ax.set_yscale('log')
    fig.tight_layout()
    fig.savefig(f"{prefix}_scatter.png")
    print(f"{prefix}_scatter.png saved")

    # 動径平均とエラーバー
    q = r2q(r+0.5)
    savetxt(f"{prefix}_radial.csv", np.array([q, i, i_std, cnt]).T,
            header=["q[nm^-1]", "i", "i_std", "n"], overwrite=True)
    fig, ax = plt.subplots(figsize=figsize)
    ax.errorbar(q, i, yerr=i_std, fmt='o', markersize=2)
    ax.set_xlabel(r"$q\,[\mathrm{nm}^{-1}]$")
    ax.set_xlim(min_q, max_q)
    ax.set_ylabel(r"$I(q)\,[\mathrm{a.u.}]$")
    ax.set_ylim(min_i, max_i)
    ax.set_yscale('log')
    fig.tight_layout()
    fig.savefig(f"{prefix}_radial.png")
    print(f"{prefix}_radial.png saved")

    return r+0.5, i

def file_integrate(file:str, **kwargs):
    """SAXS画像を積分する

    see `saxs.series_integrate`
    """
    kwargs['verbose'] = False
    return series_integrate(file, **kwargs)

def series_integrate(src: list[str]|str, *,
                     param_src = '', mask_src: str='', mask: np.ndarray=np.array([]),
                     center_x=np.nan, center_y=np.nan,
                     camera_length=np.nan, wave_length=np.nan,
                     px_size=np.nan, detecter="",
                     slope=np.nan, intercept=np.nan,
                     flip='vertical',
                     statistics=False,
                     dst="", overwrite=False, verbose=True):
    """SAXS画像の系列を積分する

    Parameters
    ----------
    src : str
        画像ファイルのディレクトリまたｈファイル名
    mask_src : str
        マスク画像のファイル名
    mask : np.ndarray
        マスク画像の配列, 0の画素は無視される
    center_x : float
        ビームセンターのx座標
    center_y : float
        ビームセンターのy座標
    camera_length : float
        カメラ長[mm]
    wave_length : float
        X線の波長[nm]
    px_size : float
        1pxのサイズ[mm]
    detecter : str
        検出器名(`PILATUS` or `EIGER`)
    slope : float
        線形回帰の傾き[nm^-1/px]
    intercept : float
        線形回帰の切片[nm^-1]
    flip : str
        ''なら反転無し、'v'なら上下反転、'h'なら左右反転、'vh'なら上下左右反転
    statistics : bool
        Trueなら統計情報を出力する
    dst : str
        結果を保存するファイル名、指定がなければdir.csv
    overwrite : bool
        Trueなら上書きする
    verbose : bool
        Trueなら進捗バーを表示する
    """
    files:list[str] = []
    if isinstance(src, str):
        if src.endswith(".tif"):
            if not os.path.exists(src):
                raise FileNotFoundError(f"{src} is not found.")
            dst = dst if dst else re.sub(r"\.tif$", ".csv", src)
            files = [src]
        else:
            if not os.path.isdir(src):
                raise FileNotFoundError(f"{src} is not found.")
            files = [os.path.join(src, f) for f in listFiles(src, ext=".tif")]
            dst = dst if dst else src + ".csv"
    else:
        for file in src:
            if not file.endswith(".tif"):
                raise ValueError("Unsupported file format: only .tif is supported")
        if len(dst) == 0:
            raise ValueError("dst to save results must be set")
        files=src

    if param_src:
        if not os.path.exists(param_src):
            raise FileNotFoundError(f"{param_src} is not found.")
        params = Saxs2dParams.load(param_src)
        center_x = params.center_x
        center_y = params.center_y
        camera_length = params.camera_length
        wave_length = params.wave_length
        px_size = params.px_size
        slope = params.slope
        intercept = params.intercept
        flip = params.flip

    n_files = len(files)
    if n_files == 0:
        raise FileNotFoundError(f"No tif files in {src}.")
    if n_files == 1:
        verbose = False

    if verbose:
        bar = tqdm.tqdm(total=n_files)

    if not overwrite and os.path.exists(dst):
        raise FileExistsError(f"{dst} is already exists.")

    i_all = []
    headers = ["q[nm^-1]"]

    if detecter.upper() in DETECTER_PX_SIZES:
        px_size = DETECTER_PX_SIZES[detecter.upper()]
    elif detecter == '':
        if np.isnan(px_size):
            raise ValueError("either `px_size` or `detecter` must be set")
    else:
        raise ValueError(f'unrecognized detecter `{detecter}`')

    calibration = 'none'
    if is_numeric(camera_length) and is_numeric(wave_length):
        calibration = 'geometry'
    elif is_numeric(slope) and is_numeric(intercept):
        calibration = 'linear_regression'
    else:
        warnings.warn("no valid calibration parameter given")

    height, width = cv2.imread(files[0], cv2.IMREAD_UNCHANGED).shape
    mask_flg = False
    if mask.size > 0:
        if mask.shape != (height, width):
            raise ValueError("mask size not match.")
        mask_flg = True
    else:
        if mask_src:
            mask = _readmask(mask_src)
            if mask.shape != (height, width):
                raise ValueError(f"mask size not match. {mask_src}")
            mask_flg = True

    if calibration == 'geometry':
        def _r2q(r) -> np.ndarray:
            return r2q(r, camera_length, wave_length=wave_length, px_size=px_size)
    elif calibration == 'linear_regression':
        def _r2q(r) -> np.ndarray:
            return intercept + slope * r
    else:
        def _r2r(r:np.ndarray) -> np.ndarray:
            return r * px_size
        _r2q = _r2r

    r, i = np.array([]), np.array([])
    for file in files:
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        if img.shape != (height, width):
            raise ValueError(f"Image size is not match. {file}")
        if 'v' in flip:
            img = np.flipud(img)
        if 'h' in flip:
            img = np.fliplr(img)

        if statistics:
            if not mask_flg:
                mask = np.ones_like(img)
            r, i = _get_stats(img, mask, center_x, center_y, file.replace('.tif', ''), _r2q)
        else:
            if mask_flg:
                r, i = _mask_and_average(img, mask, center_x, center_y)
            else:
                r, i = _radial_average(img, center_x, center_y)
        i_all.append(i)
        headers.append(os.path.basename(file))
        if verbose:
            bar.update(1)

    q = _r2q(r)
    if calibration == 'none':
        headers[0] = "r[mm]"

    arr_out = np.hstack([q.reshape(-1, 1), np.array(i_all).T])
    savetxt(dst, arr_out, header=headers, overwrite=overwrite)

    if mask_flg:
        if mask_src == '':
            mask_src = dst.replace(".csv", "_mask.tif")
            cv2.imwrite(mask_src, mask)

    paramfile = dst.replace(".csv", "_params.json")

    if 'v' in flip and 'h' in flip:
        flip = 'vertical and horizontal'
    elif 'v' in flip:
        flip = 'vertical'
    elif 'h' in flip:
        flip = 'horizontal'
    else:
        flip = 'none'
    params = Saxs2dParams(center_x=center_x, center_y=center_y,
                              calibration_type=calibration, px_size=px_size,
                              camera_length=camera_length, wave_length=wave_length,
                              slope=slope, intercept=intercept, flip=flip,
                              mask_src=mask_src)
    if param_src == paramfile:
        overwrite = True
    params.save(paramfile, overwrite=overwrite)


    if verbose:
        bar.close()

    return dst

class Mask:
    """値が0の画素を無視するマスク"""
    def __init__(self, shape=(0,0), value:np.ndarray|None=None):
        """valueで初期化されたマスクを作成する
        valueが与えられなければshapeで指定されたサイズで1埋めのマスクを作成する
        """
        if value is not None:
            self.__mask = value.astype(np.uint8)
        else:
            if shape[0] <= 0 or shape[1] <= 0:
                raise ValueError("Invalid shape")
            self.__mask = np.ones(shape, dtype=np.uint8)
        return

    @property
    def value(self, dtype=np.uint8) -> np.ndarray:
        """格納された値をdtypeで指定した型で返す"""
        return (self.__mask > 0).astype(dtype)

    def apply(self, arr:np.ndarray):
        """arrで与えられた配列にマスクを適用した結果を返す"""
        return arr * (self.value>0).astype(arr.dtype)

    @property
    def shape(self):
        return self.__mask.shape

    def add(self, arr:np.ndarray):
        """arrがnon zeroの画素をマスクするように変える"""
        self.__mask[arr > 0] = 0
        return

    def add_rectangle(self, x: int, y: int, width: int, height: int):
        """マスクに長方形を加える

        Parameters
        ----------
        x : int
            長方形の左上のx座標
        y : int
            長方形の左上のy座標
        width : int
        height : int
        """
        self.__mask[y:y+height, x:x+width] = 0
        return

    def remove_rectangle(self, x: int, y: int, width: int, height: int):
        """マスクから長方形を取り除く

        Parameters
        ----------
        x : int
            長方形の左上のx座標
        y : int
            長方形の左上のy座標
        width : int
        height : int
        """

        self.__mask[y:y+height, x:x+width] = 1
        return

    def save(self, file: str='mask.pbm'):
        """マスクをファイルに保存する"""
        cv2.imwrite(file, self.__mask)
        return

    @classmethod
    def read(cls, src: str)->'Mask':
        """ファイルからマスクを読み込む"""
        return cls(value=_readmask(src))

class Saxs2d:
    """SAXSの2次元データを扱うクラス

    Attributes
    ----------
    __i : np.ndarray
        散乱強度の2次元配列
    __px2q : float
        1pxあたりのqの変化量[nm^-1/px]
    __center : Tuple[float, float]
        ビームセンターの座標(x,y)
    """
    def __init__(self, i: np.ndarray, px2q: float, center: Tuple[float, float]):
        self.__i = i  # floatの2次元配列 欠損値はnp.nan
        self.__px2q = px2q  # nm^-1/px
        self.__center = (center[0], center[1])  # (x,y)
        return

    @property
    def i(self) -> np.ndarray:
        return self.__i

    @property
    def center(self) -> Tuple[float, float]:
        return self.__center

    @property
    def px2q(self) -> float:
        return self.__px2q

    def radial_average(
        self, q_min: float = 0, q_max: float = np.inf
    ) -> Tuple[np.ndarray, np.ndarray]:
        """q_minからq_maxまでの範囲で動径平均を計算する
        インスタンスに格納された変数を使ってnumba.jitを使用せずに動径平均を計算する

        Parameters
        ----------
        q_min : float
            動径平均を計算する最小のq値[1/nm]
        q_max : float
            動径平均を計算する最大のq値[1/nm]

        Returns
        -------
        i : np.ndarray
            散乱強度の動径平均の配列
        q : np.ndarray
            qの配列
        """
        rx = np.arange(self.__i.shape[1]) - self.__center[0]
        ry = np.arange(self.__i.shape[0]) - self.__center[1]
        rxx, ryy = np.meshgrid(rx, ry)
        r = np.sqrt(rxx**2 + ryy**2)  # type: ignore

        r_min = int(np.floor(q_min / self.__px2q))
        r_max = int(np.ceil(min(q_max / self.__px2q, r.max())))
        r_bin = np.arange(r_min, r_max + 1, 1)

        r[np.isnan(self.__i)] = np.nan
        cnt = np.histogram(r, bins=r_bin)[0]
        i_sum = np.histogram(r, bins=r_bin, weights=self.__i)[0]
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        i = i_sum / cnt
        warnings.resetwarnings()

        q_bin = r_bin * self.__px2q
        return i, (q_bin[:-1] + q_bin[1:]) / 2
