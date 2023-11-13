import numpy as np
from scipy import interpolate
import os
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import util
import XafsData
from typing import Callable
from importlib import import_module

__version__ = "0.1.0"

_EMPTY = np.array([])


class Saxs1dProfile:
    """
    Attributes
    ----------
    _r : np.ndarray [px]
    _i : np.ndarray
    _theta : np.ndarray [deg]
    _energy : float [eV]
    _lambda : float [nm]
    _q : np.ndarray [nm^-1]
    m : float
        slope of linear regression of q=mr
    """

    def __init__(
        self,
        *,
        i: np.ndarray,
        r: np.ndarray = _EMPTY,
        theta: np.ndarray = _EMPTY,
        q: np.ndarray = _EMPTY,
    ):
        self._r: np.ndarray = r
        self._theta: np.ndarray = theta
        self._q: np.ndarray = q
        self._i: np.ndarray = i
        return

    @property
    def r(self) -> np.ndarray:
        return self._r

    @property
    def theta(self) -> np.ndarray:
        return self._theta

    @property
    def q(self) -> np.ndarray:
        return self._q

    @property
    def i(self) -> np.ndarray:
        return self._i

    @property
    def lambda_(self) -> float:
        return self._lambda

    @lambda_.setter
    def lambda_(self, l: float):
        self._lambda = l
        self._energy = 1_240 / l
        return

    @classmethod
    def load_csv(
        cls, path: str, *, delimiter: str | None = None, skiprows: int = 4, axis="r"
    ) -> "Saxs1dProfile":
        if delimiter == None:
            if path.endswith(".csv"):
                delimiter = ","
        try:
            table = np.loadtxt(path, delimiter=delimiter, skiprows=skiprows)
        except:
            table = np.loadtxt(
                path, delimiter=delimiter, skiprows=skiprows, encoding="cp932"
            )
        if axis == "r":
            return cls(r=table[:, 0], i=table[:, 1])
        elif axis == "theta":
            return cls(theta=table[:, 0], i=table[:, 1])
        elif axis == "q":
            return cls(q=table[:, 0], i=table[:, 1])
        else:
            raise ValueError("invalid axis")


class SaxsSeries:
    """
    attributes
    ----------
    dir : str
        path to directory containing csv files
    _r : np.ndarray
        1d array of radial coordinate[px] of the data (common to all files)
    _i : np.ndarray
        2d array of intensity data (i[n_file, r.size])
    """

    def __init__(self, dir: str, *, axis="r", ext=".csv"):
        self.dir = os.path.join(os.getcwd(), dir)

        filePaths = [
            os.path.join(self.dir, name) for name in util.listFiles(self.dir, ext=ext)
        ]
        files = [Saxs1dProfile.load_csv(f, axis=axis) for f in filePaths]
        self._r, self._theta, self._q = _EMPTY, _EMPTY, _EMPTY
        if axis == "r":
            self._r = files[0]._r
        elif axis == "theta":
            self._theta = files[0]._theta
        elif axis == "q":
            self._q = files[0]._q
        else:
            raise ValueError("invalid axis")
        self.axis = axis
        self._i = np.array([f._i for f in files], dtype=float)
        self._m = np.nan
        return

    @property
    def m(self) -> float:
        return self._m

    @m.setter
    def m(self, m: float):
        self._m = m
        self._q = m * self.r
        return

    @property
    def q(self) -> np.ndarray:
        return self._q

    @property
    def r(self) -> np.ndarray:
        return self._r

    @property
    def theta(self) -> np.ndarray:
        return self._theta

    @property
    def i(self) -> np.ndarray:
        return self._i

    def heatmap(
        self,
        ax: Axes,
        *,
        uselog: bool = True,
        x_axis="",
        x_lim=(np.nan, np.nan),
        y: np.ndarray = np.array([]),
        y_label: str = "file number",
        y_lim=(np.nan, np.nan),
        levels: int | np.ndarray = 128,
        cmap: str = "rainbow",
        show_colorbar: bool = False,
    ) -> Axes:
        """plot heatmap of i[file, r] on given ax"""
        i = np.log(self.i) if uselog else self.i

        if y.size == 0:
            y = np.arange(i.shape[0])
        elif y.size != i.shape[0]:
            raise ValueError("invalid y size")

        x_axis = x_axis if x_axis else self.axis
        if x_axis == "r":
            x = self.r
            x_label = "r[px]"
        elif x_axis == "theta":
            x = self.theta
            x_label = "2theta[deg]"
        elif x_axis == "q":
            x = self.q
            x_label = "q[nm^-1]"
        else:
            raise ValueError("invalid x_axis")

        if not np.isnan(x_lim[0]):
            ini = np.searchsorted(x, x_lim[0])
            i = i[:, ini:]
            x = x[ini:]
        if not np.isnan(x_lim[1]):
            fin = np.searchsorted(x, x_lim[1])
            x = x[:fin]
            i = i[:, :fin]
        if not np.isnan(y_lim[0]):
            ini = np.searchsorted(y, y_lim[0])
            y = y[ini:]
            i = i[ini:, :]
        if not np.isnan(y_lim[1]):
            fin = np.searchsorted(y, y_lim[1])
            y = y[:fin]
            i = i[:fin, :]

        contf = ax.contourf(x, y, i, levels=levels, cmap=cmap)

        if show_colorbar:
            cb_label = "ln(I)" if uselog else "I"
            plt.colorbar(contf).set_label(cb_label)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        return ax


class DafsData(SaxsSeries):
    """
    attributes
    ----------
    xafsfile : XafsData
        xafs file includes energy and i0 at each file
    name : str
        sample name fetched from xafsfile
    _i0 : np.ndarray
        intensity of incident beam at each energy [raw counts]
    _mu : np.ndarray
        absorption coefficient at each energy [a.u.]
    _fl : np.ndarray
        fluorescence yield at each energy [raw counts / i0]
    _energy : np.ndarray
        energy at each file
    _r : np.ndarray (n,2)
    _i : np.ndarray (n,2)
        normalized intensity of scattered beam at each energy [raw counts / i0]
    """

    def __init__(self, dir: str, xafsfile: str, *, xafscols=(3, 4, 5)):
        """load xafs file and fetch i0 and energy
        arguments
        ---------
        dir : str
            path to directory containing csv files
        xafsfile : str
            relative path to xafs file from `dir`
        xafscols : tuple[int, int]
            column numbers of i0 and i in xafs file
        """
        super().__init__(dir, axis="theta")
        xafsfile = os.path.join(dir, xafsfile)
        self.xafsfile = XafsData.XafsData(xafsfile, cols=xafscols)
        self.name = self.xafsfile.sampleinfo.split(" ")[0]
        if self.xafsfile.energy.size != len(util.listFiles(self.dir, ext=".csv")):
            raise ValueError("inconsistent number of files. incorrect xafs file ?")
        self._i0: np.ndarray = self.xafsfile._data[:, 0]
        self._mu = np.log(self._i0 / self.xafsfile._data[:, 1])
        self._fl = self.xafsfile._data[:, 2] / self._i0
        self._i = self._i / self._i0.reshape(-1, 1)
        self._energy: np.ndarray = self.xafsfile.energy
        self.n_e: int = self._energy.size

        _lambda = 1_240 / self._energy.reshape(-1, 1)
        thetaGrid, lambdaGrid = np.meshgrid(self._theta, _lambda)
        self._q = 4 * np.pi * np.sin(np.deg2rad(thetaGrid / 2)) / lambdaGrid

    @property
    def i0(self) -> np.ndarray:
        return self._i0

    @property
    def mu(self) -> np.ndarray:
        return self._mu

    @property
    def fl(self) -> np.ndarray:
        return self._fl

    @property
    def energy(self) -> np.ndarray:
        return self._energy

    @property
    def q(self) -> np.ndarray:
        return self._q

    def heatmap(
        self,
        ax: Axes,
        *,
        uselog: bool = True,
        levels: int = 128,
        cmap: str = "rainbow",
        show_colorbar: bool = False,
    ) -> Axes:
        return super().heatmap(
            ax,
            uselog=uselog,
            x_axis="theta",
            y=self.energy,
            y_label="energy[eV]",
            levels=levels,
            cmap=cmap,
            show_colorbar=show_colorbar,
        )

    def q_slice(self, q: float | np.ndarray) -> np.ndarray:
        """fetch e-i from all files at given q[nm^-1]"""
        arr = np.array(
            [
                np.interp(q, self.q[j, :], self.i[j, :], left=np.nan, right=np.nan)
                for j in range(self.energy.size)
            ]
        )
        return arr

    def e_slice(
        self, e: float, axis="theta", strictE=False
    ) -> tuple[np.ndarray, np.ndarray]:
        """fetch q, i from all files at given e[eV]"""
        if e < self.energy[0] or e > self.energy[-1]:
            raise ValueError(f"e:{e} is out of range")
        else:
            idx = np.argmin(np.abs(self.energy - e))
            if strictE and np.abs(self.energy[idx] - e) > 1e-3:
                raise ValueError(f"unlisted energy: {e}")
            i = self.i[idx]

        if axis == "theta":
            return self.theta, i
        else:
            return self.q[idx], i


def saveHeatmap(
    dir,
    *,
    overwrite=False,
    title="",
    load_axis="r",
    save_axis="r",
    ext=".csv",
    x_lim=(np.nan, np.nan),
) -> tuple[Figure, Axes]:
    """ヒートマップをpngで保存する

    Parameters
    ----------
    dir : str
        一次元プロファイルのファイルが入ったディレクトリ
    overwrite : bool, optional
        Trueならファイルを上書きする, by default False
    title : str, optional
        グラフのタイトル、未指定ならdirの最下層をそのまま使う, by default ""
    load_axis : str, optional
        "q" or "r", by default "r"
    save_axis : str, optional
        "q" or "r", by default "r"
    ext : str, optional
        拡張子 .csvなら区切り文字を`,`として、それ以外ならスペースとして読み込む, by default ".csv"

    Returns
    -------
    tuple[Figure, Axes]
        生成したFigure, Axes

    Raises
    ------
    FileExistsError
        _description_
    """
    dist = dir + ".png"
    if not overwrite and os.path.exists(dist):
        raise FileExistsError(f"{dir}.png already exists")
    fig, ax = plt.subplots()
    saxs = SaxsSeries(dir, axis=load_axis, ext=ext)
    saxs.heatmap(ax, show_colorbar=True, x_axis=save_axis, x_lim=x_lim)
    if title == "":
        title = os.path.basename(dir)
    elif title == None:
        title = ""
    ax.set_title(title)
    fig.savefig(dist)
    return fig, ax
