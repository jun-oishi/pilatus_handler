import numpy as np
import warnings
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

    cnt = np.zeros(int(r_mesh.max()+1), dtype=np.int64)
    i   = np.zeros(int(r_mesh.max()+1), dtype=np.float64)
    for x in range(width):
        for y in range(height):
            if not img[y, x] >= threshold:
                continue
            idx = int(r_mesh[y, x])
            cnt[idx] += 1
            i[idx] += img[y, x]

    for idx in range(len(cnt)):
        if cnt[idx] > 0:
            i[idx] /= cnt[idx]
        else:
            i[idx] = 0

    r = 0.5 + np.arange(len(i))
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

def series_integrate(dir, *,
                     center=(np.nan,np.nan), camera_length=np.nan,
                     wave_length=np.nan, px_size=PILATUS_PX_SIZE,
                     dst="", overwrite=False):
    """SAXS画像の系列を積分する

    Parameters
    ----------
    dir : str
        画像ファイルのディレクトリ
    """
    if not os.path.isdir(dir):
        raise FileNotFoundError(f"{dir} is not found.")

    files = listFiles(dir, ext=".tif")
    n_files = len(files)
    if n_files == 0:
        raise FileNotFoundError(f"No tif files in {dir}.")

    bar = tqdm.tqdm(total=n_files)

    dst = params["dst"] if dst else dir + "_integrated.csv"
    if not overwrite and os.path.exists(dst):
        raise FileExistsError(f"{dst} is already exists.")

    i_all = []
    headers = ["q[nm^-1]"]

    file = os.path.join(dir, files[0])
    img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    height, width = img.shape
    r, i = _radial_average(img, center[0], center[1])
    q = _r2q(r, camera_length, px_size, wave_length)
    i_all.append(i)
    headers.append(os.path.basename(files[0]))
    bar.update(1)
    for file in files[1:]:
        img = cv2.imread(os.path.join(dir,file), cv2.IMREAD_UNCHANGED)
        if img.shape != (height, width):
            raise ValueError(f"Image size is not match. {file}")
        r, i = _radial_average(img, center[0], center[1])
        i_all.append(i)
        headers.append(os.path.basename(file))
        bar.update(1)

    arr_out = np.hstack([q.reshape(-1, 1), np.array(i_all).T])
    np.savetxt(dst, arr_out, delimiter=",", header=",".join(headers))

    paramfile = dst.replace(".csv", ".par")
    with open(paramfile, "w") as f:
        f.write(f"camera_length={camera_length}\n")
        f.write(f"wave_length={wave_length}\n")
        f.write(f"px_size={px_size}\n")
        f.write(f"center_x={center[0]}\n")
        f.write(f"center_y={center[1]}\n")

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
