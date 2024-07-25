import numpy as np
import warnings, re, os
from SpectraSpark.util import listFiles, write_json, ArrayLike
from typing import Tuple
from numba import jit
import tqdm
import cv2


PILATUS_PX_SIZE = 0.172  # mm
EIGER_PX_SIZE = 0.075  # mm

@jit(nopython=True, cache=True)
def _radial_average(img, center_x, center_y, threshold=2):
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
    r_mesh = np.empty(img.shape)
    dx_sq = (np.arange(width) - center_x)**2
    for y in range(height):
        r_mesh[y, :] = np.sqrt(dx_sq + (y - center_y)**2)

    min_r = int(r_mesh.min())
    max_r = int(r_mesh.max())+1
    r_range=max_r-min_r+1
    cnt = np.zeros(r_range, dtype=np.int64)
    i   = np.zeros(r_range, dtype=np.float64)
    for x in range(width):
        for y in range(height):
            if not img[y, x] >= threshold:
                continue
            idx = int(r_mesh[y, x]) - min_r
            cnt[idx] += 1
            i[idx] += img[y, x]

    for idx in range(len(cnt)):
        if cnt[idx] > 0:
            i[idx] /= cnt[idx]
        else:
            i[idx] = 0

    r = min_r + 0.5 + np.arange(len(cnt))
    return r, i

def _r2q(r, camera_length, px_size=PILATUS_PX_SIZE, wave_length=1.000):
    """画像上のpxをq[nm^-1]に変換する

    Parameters
    ----------
    r : np.ndarray
        動径[px]の配列
    camera_length : float
        カメラ長[mm]
    px_size : float
        pxサイズ[mm]
    wave_length : float
        X線波長[AA]
    """
    _tan = r * px_size / camera_length
    _theta = np.arctan(_tan) * 0.5
    return 4 * np.pi / (wave_length*0.1) * np.sin(_theta)

def file_integrate(file:str, **kwargs):
    """SAXS画像を積分する

    see `saxs.series_integrate`
    """
    kwargs['verbose'] = False
    series_integrate(file, **kwargs)

def series_integrate(src: list[str]|str, *,
                     center=(np.nan,np.nan),
                     camera_length=np.nan, wave_length=np.nan,
                     px_size=np.nan, detecter="",
                     slope=np.nan, intercept=np.nan,
                     flip='vertical',
                     dst="", overwrite=False, verbose=True):
    """SAXS画像の系列を積分する

    Parameters
    ----------
    dir : str
        画像ファイルのディレクトリ
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

    n_files = len(files)
    if n_files == 0:
        raise FileNotFoundError(f"No tif files in {src}.")

    if verbose:
        bar = tqdm.tqdm(total=n_files)

    if not overwrite and os.path.exists(dst):
        raise FileExistsError(f"{dst} is already exists.")

    i_all = []
    headers = ["q[nm^-1]"]

    if detecter in ('pilatus', 'PILATUS'):
        px_size = PILATUS_PX_SIZE
    elif detecter in ('eiger', 'EIGER'):
        px_size = EIGER_PX_SIZE
    elif detecter == '':
        if np.isnan(px_size):
            raise ValueError("either `px_size` or `detecter` must be set")
    else:
        raise ValueError(f'unrecognized detecter `{detecter}`')

    calibration = 'none'
    if camera_length > 0 and wave_length > 0:
        calibration = 'geometry'
    elif not np.isnan(slope) and not np.isnan(intercept):
        calibration = 'linear_regression'
    else:
        warnings.warn("no valid calibration parameter given")

    height, width = cv2.imread(files[0], cv2.IMREAD_UNCHANGED).shape
    r, i = np.array([]), np.array([])
    if verbose:
        bar.update(1)
    for file in files:
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        if img.shape != (height, width):
            raise ValueError(f"Image size is not match. {file}")
        if 'v' in flip:
            img = np.flipud(img)
        if 'h' in flip:
            img = np.fliplr(img)
        r, i = _radial_average(img, center[0], center[1])
        i_all.append(i)
        headers.append(os.path.basename(file))
        if verbose:
            bar.update(1)

    if calibration == 'geometry':
        q = _r2q(r, camera_length, px_size, wave_length)
    elif calibration == 'linear_regression':
        q = intercept + slope * r
    else:
        q = r * px_size
        headers[0] = "r[mm]"

    arr_out = np.hstack([q.reshape(-1, 1), np.array(i_all).T])
    np.savetxt(dst, arr_out, delimiter=",", header=",".join(headers))

    paramfile = dst.replace(".csv", "_params.json")

    params={
        'center_x[px]': center[0],
        'center_y[px]': center[1],
        'calibration_type': calibration,
        'px_size[mm]': px_size,
        'camera_length[mm]': camera_length,
        'wave_length[AA]': wave_length,
        'slope[nm^-1/px]': slope,
        'intercept[nm^-1]': intercept,
    }
    if 'v' in flip and 'h' in flip:
        flip = 'vertical and horizontal'
    elif 'v' in flip:
        flip = 'vertical'
    elif 'h' in flip:
        flip = 'horizontal'
    else:
        flip = 'none'
    params['flip'] = flip
    write_json(paramfile, params)

    if verbose:
        bar.close()
    return dst


class Saxs2d:
    def __init__(self, i: np.ndarray, px2q: float, center: ArrayLike):
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

    def rotate(self, angle: float):
        """画像を回転する"""
        raise NotImplementedError
