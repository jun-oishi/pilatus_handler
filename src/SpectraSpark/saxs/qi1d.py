import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
import warnings

from ..util.io import loadtxt

class Saxs1d:
    """1次元のSaxsデータを扱うクラス"""

    def __init__(self, i: np.ndarray, q: np.ndarray):
        self.__intensity = i
        self.__q = q
        return

    @property
    def i(self) -> np.ndarray:
        """scattering intensity"""
        return self.__intensity

    @property
    def q(self) -> np.ndarray:
        """magnitude of scattering vector [nm^-1]"""
        return self.__q

    @classmethod
    def load(cls, src: str) -> "Saxs1d":
        """csvファイルから読み込む
        第1列はq[nm^-1], 第2列をI[q]として読み込む
        規格化因子n_factorを掛ける
        """
        data = loadtxt(src)
        return Saxs1d(data[:, 1], data[:, 0])

    @classmethod
    def loadMatFile(cls, src: str, usecol: int) -> "Saxs1d":
        """データ列が複数のファイルから読み込む"""
        data = loadtxt(src, usecols=(0, usecol))
        return Saxs1d(data[:, 1], data[:, 0])

    def guinierRadius(self):
        """ギニエ半径を求める"""
        raise NotImplementedError

    def integratedIntensity(self):
        """積分強度を求める"""
        raise NotImplementedError


class Saxs1dSeries(Saxs1d):
    """時分割のSAXSデータ系列を扱うクラス"""


    def __init__(self, src: str='', *, i: np.ndarray, q: np.ndarray):
        if src:
            self.loadMatFile(src)
        elif q.shape[0] == i.shape[1]:
            super().__init__(i, q)
        else:
            raise ValueError("src path or corresponding i and q needed")

    @classmethod
    def loadMatFile(cls, src: str) -> "Saxs1dSeries":
        data = loadtxt(src)
        q = data[:, 0]
        i = data[:, 1:].T  # i[k]がk番目のプロファイル
        print(f"{i.shape[0]} profiles along {q.shape[0]} q values loaded")
        return cls(i=i, q=q)

    @classmethod
    def load(cls, src: str) -> "Saxs1dSeries":
        """csvファイルを読み込んでSaxs1dSeriesを返す

        qi2d.series_integrateで出力したファイルを読み込む
        """
        return cls.loadMatFile(src)

    def load_temperature(self, src: str, *, usecol=4, skiprows=1):
        """温度データを読み込む"""
        values = loadtxt(src, usecols=usecol, skiprows=skiprows)
        if len(values) < self.i.shape[0]:
            raise ValueError("Temperature data size not match")
        elif len(values) > self.i.shape[0]:
            warnings.warn("Temperature data size exceeds the number of profiles: trailing data will be ignored")
        self.__temperature = values[:self.i.shape[0]]
        return

    @property
    def t(self) -> np.ndarray:
        """temperature history"""
        return self.__temperature

    def peakHistory(self, q_min: float, q_max: float) -> np.ndarray:
        """ピーク強度の時系列を求める"""
        raise NotImplementedError

    def heatmap(
            self, fig: Figure, ax: Axes,
            *,
            logscale: bool = True,
            x_label="$q$ [nm$^{-1}$]", x_lim=(-np.inf, np.inf),
            y_label: str = "", y_ticks = {}, y_lim=(0, None),
            v_min=np.nan, v_max=np.nan, extend: str = "min",
            cmap: str | Colormap = "jet",
            show_colorbar: bool = True,
            cbar_fraction: float = 0.02,
            cbar_pad: float = 0.08,
            cbar_aspect: float = 50,
    ):
        """ヒートマップを描画する

        Parameters
        ----------
        fig : Figure
        ax : Axes
        logscale : bool, default True
            Truethyならy軸を対数スケール
        x_label : str, default "$q$ [nm$^{-1}$]"
            x軸ラベル
        x_lim : Tuple[float, float], default (-np.inf, np.inf)
            x軸の範囲
        y_label : str, default ""
            y軸ラベル
        y_ticks : Dict[int, str], default {}
            y軸目盛り, 与えられなければファイル番号
        y_lim : Tuple[int, int], default (0, None)
            y軸の範囲
        v_min : float, default np.nan
            カラーバーの最小値
        v_max : float, default np.nan
            カラーバーの最大値
        extend : str, default "min"
            カラーバーの外側の表示方法("neither", "both", "min", "max")
            see matplotlib.pyplot.contourf
        cmap : str or Colormap, default "jet"
            カラーマップ
        show_colorbar : bool, default True
            Truethyならカラーバーを表示
        cbar_fraction : float, default 0.02
            カラーバーの幅
        cbar_pad : float, default 0.08
            カラーバーの位置
        cbar_aspect : float, default 50
            カラーバーのアスペクト比
        """
        q = self.q.copy()
        val = self.i.copy()
        if logscale:
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            val = np.log10(val)
            warnings.resetwarnings()
            isvalid = (np.isfinite(val).sum(axis=0)==val.shape[0])
            if not np.any(isvalid):
                raise ValueError("No valid data in the range")
            else:
                print(f"{val.shape[1]-isvalid.sum()} columns include NaN, so removed")
                q = q[isvalid]
                val = val[:, isvalid]

        idx = ((x_lim[0] < q) * (q < x_lim[1]))
        val = val[y_lim[0]:y_lim[1], idx]        # `arr[i:None]` interpreted as `arr[i:arr.size]``
        q = q[idx]
        y = np.arange(1, val.shape[0] + 1)
        if val.size == 0:
            if x_lim[0] > np.max(q) or x_lim[1] < np.min(q):
                raise ValueError("Specified q range is out of data range")
            elif y_lim[1] > val.shape[0] or y_lim[0] < 0:
                raise ValueError("Specified y range is out of data range")
            else:
                raise ValueError("No data in specified range: something wrong")
        v_min = np.nanmin(val) if np.isnan(v_min) else v_min
        v_max = np.nanmax(val) if np.isnan(v_max) else v_max
        levels = np.linspace(v_min, v_max, 256)
        cs = ax.contourf(q, y, val, levels=levels, cmap=cmap, extend=extend)

        if show_colorbar:
            cbar = fig.colorbar(
                cs, ax=ax,
                fraction=cbar_fraction, pad=cbar_pad, aspect=cbar_aspect
            )
            cbar.set_label(r"$\log[I(q)]\;[a.u.]$" if logscale \
                           else r"$I[q]\;[a.u.]$")

        ax.set_xlabel(x_label)
        if len(y_ticks) != 0:
            y, y_labels = zip(*y_ticks.items())
            ax.set_yticks(y, y_labels)
        elif not y_label:
            y_label = "file number"
        ax.set_ylabel(y_label)
        return
