import numpy as np
import warnings
from SpectraSpark.util import listFiles, write_json, ArrayLike
from typing import Tuple


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
