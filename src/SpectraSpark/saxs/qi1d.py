import numpy as np


class Saxs1d:
    """1次元のSaxsデータを扱うクラス"""

    __DEFAULT_DELIMITER = ","

    def __init__(self, i: np.ndarray, q: np.ndarray):
        self.__i = i
        self.__q = q
        return

    @property
    def i(self) -> np.ndarray:
        return self.__i

    @property
    def q(self) -> np.ndarray:
        return self.__q

    @classmethod
    def load(cls, src: str) -> "Saxs1d":
        """csvファイルから読み込む
        P2M.toChiFile()で出力したファイル(ヘッダ3行)を読み込む
        第1列はq[nm^-1]
        """
        data = np.loadtxt(src, delimiter=cls.__DEFAULT_DELIMITER, skiprows=3)
        return Saxs1d(data[:, 1], data[:, 0])

    @classmethod
    def loadMatFile(cls, src: str, usecol: int) -> "Saxs1d":
        """データ列が複数のファイルから読み込む"""
        data = np.loadtxt(
            src, usecols=(0, usecol), delimiter=cls.__DEFAULT_DELIMITER, skiprows=1
        )
        return Saxs1d(data[:, 1], data[:, 0])

    def guinierRadius(self):
        """ギニエ半径を求める"""
        raise NotImplementedError

    def integratedIntensity(self):
        """積分強度を求める"""
        raise NotImplementedError


class Saxs1dSeries(Saxs1d):
    """時分割のSAXSデータ系列を扱うクラス"""

    __DEFAULT_DELIMITER = ","

    @classmethod
    def loadMatFile(cls, src: str) -> "Saxs1dSeries":
        data = np.loadtxt(src, delimiter=cls.__DEFAULT_DELIMITER, skiprows=3)
        q = data[:, 0]
        i = data[:, 1:].T  # q[i]がi番目のデータ
        return cls(i, q)

    @classmethod
    def load(cls, src: str) -> "Saxs1dSeries":
        return cls.loadMatFile(src)

    def peakHistory(self, q_min: float, q_max: float) -> np.ndarray:
        """ピーク強度の時系列を求める"""
        raise NotImplementedError
