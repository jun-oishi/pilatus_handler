"""module for 2D SAXS profile"""

import numpy as np
import cv2
import os
import util
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.axes import Axes
import matplotlib
from datetime import datetime
from typing import overload
import tqdm
import re

# matplotlib.use("Qt5Agg")

__version__ = "0.1.0"


GREEN = (0, 255, 0)  # BGR

_EMPTY = np.array([])


def _compress(raw: np.ndarray, dtype=np.uint8, *, min=None, max=None) -> np.ndarray:
    """compress np.array into specified dtype"""
    if min is None:
        min = np.nanmin(raw)
    if max is None:
        max = np.nanmax(raw)
    zero2one: np.ndarray = (raw - min) / (max - min)  # type: ignore
    if dtype == np.uint8:
        return (zero2one * ((1 << 8) - 1)).astype(dtype)
    elif dtype == np.uint16:
        return (zero2one * ((1 << 16) - 1)).astype(dtype)
    else:
        raise ValueError("invalid dtype: only uint8 and uint16 are supported")


def _ellipse(
    img: np.ndarray,
    center: tuple[float, float],
    axes: tuple[float, float],
    tilt: float = 0,
    color: int = 255,
    thickness: int = 1,
) -> np.ndarray:
    return cv2.ellipse(img, (center, axes, tilt), color, thickness)  # type: ignore


class _Detector:
    """検出器を表現するクラス
    基本的にシングルトンで使う

    Attributes
    ----------
    name : str
    pixelSize : float [mm]
    """

    def __init__(self, pixelSize: float = np.nan, *, name: str = ""):
        self.__pixelSize: float = pixelSize
        self.__name: str = name

    @classmethod
    def unknown(cls, pixelSize=np.nan) -> "_Detector":
        if not hasattr(cls, "_unknown"):
            cls._unknown = cls(pixelSize, name="unknown")
        return cls._unknown

    @classmethod
    def Pilutus(cls) -> "_Detector":
        if not hasattr(cls, "_pilutus"):
            cls._pilutus = cls(0.172, name="pilatus")
        return cls._pilutus

    @classmethod
    def Eiger(cls) -> "_Detector":
        if not hasattr(cls, "_eiger"):
            cls._eiger = cls(0.075, name="eiger")
        return cls._eiger

    @property
    def name(self) -> str:
        return self.__name

    @property
    def pixelSize(self) -> float:
        return self.__pixelSize


class Saxs2dProfile:
    """SAXS 2D profile class

    attributes
    ----------
    _raw : np.ndarray
    center : tuple[float, float]
        (x, y) coordinate of beam center in THIS ORDER [px]
    _mask : np.ndarray
        0の画素は無視される
    detector : _Detector
    pixelSize : float [mm/px]
    _cameraLength : float [mm]
    _waveLength : float [nm]
    _AngleOfTiltPlane : float [deg]
    _AngleOfTilt : float [deg]
    _AngleOfRotation : float [deg]
    _PolarisationFactor : float
    """

    DEFAULT_MARK_COLOR = GREEN

    def __init__(self, raw: np.ndarray):
        self._raw: np.ndarray = raw
        self.__center: tuple[float, float] = (np.nan, np.nan)
        self._mask: np.ndarray = np.ones_like(raw, dtype=bool)
        self._detector: _Detector = _Detector.unknown()
        self._cameraLength: float = np.nan  # mm
        self._waveLength: float = np.nan  # nm
        self._AngleOfTiltPlane: float = np.nan  # deg
        self._AngleOfTilt: float = np.nan  # deg
        self._AngleOfRotation: float = np.nan  # deg
        self._PolarisationFactor: float = np.nan

    def loadFit2dPar(self, path: str) -> None:
        """fit2dのパラメタファイルからパラメタを読み込む"""
        lines = []
        with open(path) as f:
            lines = f.readlines()

        for line in lines:
            m = re.match(
                r"X/Y pixel sizes =[ ]+(\d+\.\d+),[ ]+(\d+\.\d+)[ ]+microns", line
            )
            if m:
                self._detector = _Detector(float(m.group(1)) / 1000)
                continue
            m = re.match(r"Sample to detector distance =[ ]+(\d+\.\d+)[ ]+mm", line)
            if m:
                self._cameraLength = float(m.group(1))
                continue
            m = re.match(r"Wavelength =[ ]+(\d+.\d+)[ ]+Angstroms", line)
            if m:
                self._waveLength = float(m.group(1)) / 10
                continue
            m = re.match(
                r"X/Y pixel position of beam =[ ]+(\d+\.\d+),[ ]+(\d+\.\d+)", line
            )
            if m:
                self.center = (float(m.group(1)), float(m.group(2)))
                continue
            m = re.match(r"Angle of tilt plane =[ ]+(\d+\.\d+)[ ]+degrees", line)
            if m:
                self._AngleOfTiltPlane = float(m.group(1))
                continue
            m = re.match(r"Angle of tilt =[ ]+(\d+\.\d+)[ ]+degrees", line)
            if m:
                self._AngleOfTilt = float(m.group(1))
                continue
            m = re.match(r"Detector rotation angle =[ ]+(\d+\.\d+)[ ]+degrees", line)
            if m:
                self._AngleOfRotation = float(m.group(1))
                continue
            m = re.match(r"Polarisation factor =[ ]+(\d+\.\d+)", line)
            if m:
                self._PolarisationFactor = float(m.group(1))
                continue
        return

    def setDetector(self, detector: str):
        if detector == "pilatus":
            self._detector = _Detector.Pilutus()
        elif detector == "eiger":
            self._detector = _Detector.Eiger()

    @property
    def pixelSize(self) -> float:
        return self._detector.pixelSize

    @property
    def raw(self) -> np.ndarray:
        return self._raw

    @property
    def i(self) -> np.ndarray:
        return self._raw * self._mask

    @property
    def shape(self) -> tuple[int, int]:
        return self._raw.shape  # type: ignore

    @property
    def width(self) -> int:
        return self._raw.shape[1]

    @property
    def height(self) -> int:
        return self._raw.shape[0]

    @classmethod
    def load_tiff(cls, path: str, flip="", paramfile=""):
        """load profile from tiff file

        Parameters
        ----------
        path: str
            path to tiff file

        Returns
        -------
        Saxs2dProfile
        """
        if not os.path.exists(path):
            raise FileNotFoundError("")
        if path[-4:] != ".tif":
            raise ValueError("invalid file type")
        flipped = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if "h" in flip or "horizontal" in flip:
            flipped = flipped[:, ::-1]
        if "v" in flip or "vertical" in flip:
            flipped = flipped[::-1, :]
        ret = cls(flipped)
        if paramfile:
            paramfile = os.path.join(os.path.dirname(path), paramfile)
            ret.loadFit2dPar(paramfile)
        return ret

    @property
    def center(self) -> tuple[float, float]:
        """center coordinate in (x,y) order"""
        return (self.__center[1], self.__center[0])

    @center.setter
    def center(self, center: tuple[float, float]):
        """set beam center by (x,y) coordinate"""
        try:
            if len(center) != 2:
                raise TypeError("")
        except TypeError:
            raise TypeError("center must be array-like of 2 floats")

        self.__center = (center[1], center[0])

    def clearMask(self) -> None:
        """clear mask"""
        self._mask = np.ones_like(self._raw, dtype=bool)
        return

    def thresholdMask(self, thresh=2) -> None:
        """閾値でマスクを作成する

        Parameters
        ----------
        thresh : int, optional
            画素値がこの値(を含まず)より小さい画素を無視するマスクをかける, by default 2
        """
        self._mask = self._mask * (self._raw >= thresh)
        return


class PatchedSaxsImage(Saxs2dProfile):
    def __init__(self, raw: np.ndarray):
        super().__init__(raw)
        self.cameraLength = np.nan  # mm

    @classmethod
    def load_tiff(cls, path: str, *, flip="v", paramfile="") -> "PatchedSaxsImage":
        ret = super().load_tiff(path, flip, paramfile=paramfile)
        ret.thresholdMask()
        return ret

    def detect_center(self) -> tuple[float, float]:
        """detect center and set to self.__center
        detect center by cv2.HoughCircles
        if no circle is detected, no error raised and self.__center is not updated

        Returns
        -------
        center: tuple[float,float]
            center of circle, (nan,nan) if no circle is detected
        """
        buf = self._raw.copy()
        cutoff = np.median(buf)
        buf[buf < cutoff] = 0
        buf = _compress(buf, dtype=np.uint8)
        circles = cv2.HoughCircles(
            buf,
            cv2.HOUGH_GRADIENT,
            1,
            20,
            param1=50,
            param2=30,
            minRadius=0,
            maxRadius=0,
        )

        if circles is None:
            raise ValueError("no circle detected")
        else:
            self.center = circles[0, 0]

        return self.center

    def radial_average(self, axis="r") -> tuple[np.ndarray, np.ndarray]:
        """compute radial average

        Parameters
        ----------
        axis : str
            "r" or "theta"
        dtheta : float
            bin width for theta axis, only used when axis="theta" [deg]

        Returns
        -------
        intensity : np.ndarray
            radial average of intensity
        bins : np.ndarray
            bins for radial average, has one more element than intensity
            when axis="r", unit is px and bin width is 1
        """
        buf = self.i
        dx = np.arange(self.width) - self.center[0]
        dy = np.arange(self.height) - self.center[1]
        dx, dy = np.meshgrid(dx, dy)
        r = np.sqrt(dx**2 + dy**2)

        dr = 1.0  # px
        r = r * self._mask.astype(r.dtype)  # maskされた画素は0になってhistogramから除外される
        r_min = int(np.floor(np.min(r[r > 0])))
        r_max = int(np.ceil(np.max(r)))

        bins = np.arange(r_min, r_max + dr, dr)

        intensity = np.empty(bins.size - 1)
        cnt = np.histogram(r, bins=bins)[0]
        cnt[cnt == 0] = -1  # 0除算を防ぐ
        sum = np.histogram(r, bins=bins, weights=buf)[0]
        intensity = sum / cnt

        if axis == "r":
            pass
        elif axis in ("2theta", "q"):
            bins = bins * self.pixelSize  # px -> mm
            bins = bins / self._cameraLength  #  mm -> tan
            bins = np.arctan(bins)  # tan -> rad
            if axis == "2theta":
                bins = np.rad2deg(bins)
            elif axis == "q":
                bins = 4 * np.pi * np.sin(bins / 2) / self._waveLength
        else:
            raise ValueError("invalid axis specifier")

        return intensity, bins

    def r2theta(self, r: np.ndarray) -> np.ndarray:
        """ビームセンターからの距離[px]を散乱角2θ[deg]に変換する

        Parameters
        ----------
        r : np.ndarray
            px

        Returns
        -------
        np.ndarray
            2θ[deg]
        """
        if np.isnan(self.pixelSize) or np.isnan(self.cameraLength):
            raise ValueError("detector parameters not set")
        return np.rad2deg(np.arcsin(r * self.pixelSize / self.cameraLength))


class TiltCameraCoordinate:
    """tilt camera coordinate
    Attributes
    ----------
    psi: float
        tilt angle[rad]
    l: float
        camera length along z-axis
    beamcenter: tuple[float, float]
        beam center in camera coordinate (x, y)
    """

    def __init__(self, psi: float, l: float, beamcenter):
        """
        Parameters
        ----------
        psi: float [rad]
        l : float [px]
        beamcenter : tuple[float, float] [px]
        """
        self.psi = psi  # rad
        self.l = l  # px
        self.beamcenter = beamcenter  # px

    def ellipseParam(self, theta, plane="camera"):
        """returns ellipse parameters
        Parameters
        ----------
        theta: float
            theta angle[rad]
        plane: str
            if "xy", returns ellipse parameters in xy plane
            if "camera", returns ellipse parameters in camera coordinate

        Returns
        -------
        a: float
            x axis length of ellipse
        b: float
            y axis length of ellipse
        center: tuple[float, float]
        """
        psi, l = self.psi, self.l
        s, t = np.tan(psi), np.tan(theta)
        d = 1 - s**2 * t**2
        a = l * t / d  # 楕円のxy平面投影のx軸方向の長さ
        b = l * t / np.sqrt(d)  # 楕円のxy平面投影のy軸方向の長さ
        x0 = -s * t**2 * l / d  # 楕円のxy平面投影の中心座標
        if plane == "xy":
            # xy平面への投影の楕円のパラメタ
            center = (x0, 0)
            return a, b, center
        elif plane == "camera":
            center = (x0 / np.cos(psi) + self.beamcenter[0], self.beamcenter[1])
            return a / np.cos(psi), b, center
        else:
            raise ValueError("plane must be 'xy' or 'camera'")

    def xy(self, theta, phi) -> np.ndarray:
        """returns x, y coordinate in camera coordinate
        Parameters
        ----------
        theta: float or array-like [rad]
        phi: float or array-like [rad]
        """
        if hasattr(phi, "__len__"):
            if hasattr(theta, "__len__"):
                return np.array([self.xy(t, p) for t, p in zip(theta, phi)])
            else:
                return np.array([self.xy(theta, p) for p in phi])
        if hasattr(theta, "__len__"):
            return np.array([self.xy(t, phi) for t in theta])

        psi, l = self.psi, self.l
        s, t = np.tan(psi), np.tan(theta)
        # 直接x(投影)を求める
        roots = np.roots(
            [
                1 + np.tan(phi) ** 2 - s**2 * t**2,
                2 * l * s * t**2,
                -(l**2) * t**2,
            ]
        )
        x = np.max(roots) if np.cos(phi) > 0 else np.min(roots)
        # 楕円の方程式からyを求める
        a, b, center = self.ellipseParam(theta, plane="xy")
        x0 = center[0]
        y = b * np.sqrt(1 - (x - x0) ** 2 / a**2)
        y = y if np.sin(phi) > 0 else -y
        # カメラ座標系に変換して返す
        return np.array([x / np.cos(psi) + self.beamcenter[0], y + self.beamcenter[1]])

    def sph(self, X, Y) -> np.ndarray:
        """returns theta, phi coordinate in spherical coordinate
        Parameters
        ----------
        X: float or array-like
        Y: float or array-like
            coordinate in camera coordinate
        """
        if hasattr(X, "__len__"):
            if hasattr(Y, "__len__"):
                return np.array([self.sph(x, y) for x, y in zip(X, Y)])
            else:
                return np.array([self.sph(x, Y) for x in X])
        if hasattr(Y, "__len__"):
            return np.array([self.sph(X, y) for y in Y])
        psi, l = self.psi, self.l
        # カメラ座標系をxy平面に投影
        x = (X - self.beamcenter[0]) * np.cos(self.psi)
        y = Y - self.beamcenter[1]
        theta = np.arctan2(np.sqrt(x**2 + y**2), l - x * np.tan(psi))
        phi = np.arctan2(y, x)
        return np.array([theta, phi])

    def drawPeak(
        self, canvas: np.ndarray, theta: float, value: int = 255, thickness: int = 1
    ) -> np.ndarray:
        """
        Parameters
        ----------
        canvas: np.ndarray
        theta: float [rad]
        value: int
            color value to draw peak
        thickness: int
        """
        a, b, ellippse_center = self.ellipseParam(theta, plane="camera")
        return _ellipse(canvas, ellippse_center, (2 * a, 2 * b), 0, value, thickness)

    def drawPeaks(
        self,
        canvas: np.ndarray,
        thetas: np.ndarray,
        value: int = 255,
        thickness: int = 1,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        canvas: np.ndarray
        thetas: float [rad]
        value: int
            color value to draw peak
        thickness: int
        """
        ret = canvas.copy()
        for theta in thetas:
            ret = self.drawPeak(ret, theta, value, thickness)
        return ret


class TiltedSaxsImage(Saxs2dProfile):
    """
    Attributes
    ----------
    psi : float
        tilt angle of detector [deg]
    cameraLength : float [px]
        camera length measured along beam direction [px]
    center : tuple[float, float]
        (x, y) coordinate of beam center [px]
        methods assume x < 0 so that small x corresponds to small theta
    e : float [eV]
    """

    def __init__(self, raw: np.ndarray):
        super().__init__(raw)
        self.psi: float = 0  # degree
        self.cameraLength: float = np.nan  # px
        self.__converted: np.ndarray = np.array([])
        self.__arr_theta_deg: np.ndarray = np.array([])

    @classmethod
    def load_tiff(cls, path: str, flip="") -> "TiltedSaxsImage":
        return super().load_tiff(path, flip)

    def __thetaGrid(self):
        """return theta grid for each pixel in degree"""
        coord = TiltCameraCoordinate(
            np.deg2rad(self.psi), self.cameraLength, self.center
        )
        sph = np.array(
            [
                [coord.sph(x, y) for x in range(self._raw.shape[1])]
                for y in range(self._raw.shape[0])
            ]
        )
        return np.rad2deg(sph[:, :, 0])

    def copyCameraParam(self, other: "TiltedSaxsImage"):
        self.psi, self.cameraLength = other.psi, other.cameraLength
        self.center = other.center
        return

    def convert2theta(
        self, dtheta=2**-6, min_theta=0, max_theta=90, *, autotrim=True
    ):
        """convert x-axis from pixel to theta
        keep longitudinal resolution and reset lateral axis to theta
        """
        theta = self.__thetaGrid()
        if autotrim:
            min_theta, max_theta = np.nanmin(theta), np.nanmax(theta)
            arr_theta = np.arange(min_theta, max_theta, dtheta)
        else:
            arr_theta = np.arange(min_theta + dtheta / 2, max_theta, dtheta)
        height = self._raw.shape[0]
        converted = np.array(
            [
                np.interp(arr_theta, theta[j], self._raw[j], left=np.nan, right=np.nan)
                for j in range(height)
            ],
            dtype=np.float32,
        )
        converted = converted / np.cos(np.deg2rad(arr_theta - self.psi))
        self.__converted = converted
        self.__arr_theta_deg = arr_theta
        return converted, arr_theta

    def radial_average(
        self, dtheta=2**-6, min_theta=0, max_theta=90, *, autotrim=True
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.__arr_theta_deg.size == 0:
            self.convert2theta(dtheta, min_theta, max_theta, autotrim=autotrim)
        return np.nanmean(self.__converted, axis=0), self.__arr_theta_deg


class DeterminCameraParam:
    """カメラパラメタをインタラクティブに決定するクラス
    Attronibutes
    ------------
        __raw (np.ndarray) : 画像データ
        energy (float) : X線のエネルギー [eV]
        ini_values (dict) : 初期値
        min_values (dict) : スライダの最小値
        max_values (dict) : スライダの最大値
    """

    def __init__(self, src: str, *, energy: float, flip=""):
        """
        Parameters
        ----------
        src : str
            path to tiff file
        energy : float
            energy of x-ray [eV]
        flip : str
            "h" or "horizontal" to flip image horizontally
        """
        self.__raw = _compress(cv2.imread(src, cv2.IMREAD_UNCHANGED), np.uint8)
        if "h" in flip or "horizontal" in flip:
            self.__raw = self.__raw[:, ::-1]

        self.energy = energy  # eV

        self.ini_values = {
            "cameraLength": 5300,
            "psi": 30.0,
            "center_x": -300.0,
            "center_y": 85.0,
        }

        self.min_values = {
            "cameraLength": 3500,
            "psi": 10.0,
            "center_x": -500.0,
            "center_y": 70.0,
        }

        self.max_values = {
            "cameraLength": 5500,
            "psi": 40.0,
            "center_x": -100.0,
            "center_y": 120.0,
        }

    @overload
    def __q2theta(self, q: np.ndarray) -> np.ndarray:
        ...

    @overload
    def __q2theta(self, q: float) -> float:
        ...

    def __q2theta(self, q):
        """散乱ベクトルq[nm^-1]から散乱角2theta[rad]を計算する

        Parameters
        ----------
        q : float | np.ndarray

        Returns
        -------
        float | np.ndarray
        """
        return 2 * np.arcsin(q * (1_240 / self.energy) / (4 * np.pi))

    def run(self, q: np.ndarray | str, imgpath: str = "") -> TiltCameraCoordinate:
        """インタラクティブにカメラパラメタを決定する
        TODO : 操作を終了しなくても(グラフは表示されたまま)処理が流れてしまう

        Parameters
        ----------
        q : np.ndarray | str
            描画するピークのq値[nm^-1]の配列または"agbeh"
        imgpath : str, optional
            空文字列以外が与えられれば終了時にそのパスに画像を保存する, by default ""

        Returns
        -------
        TiltCameraCoordinate
            決定したパラメタを持つTiltCameraCoordinateオブジェクト

        Raises
        ------
        ValueError
            qが"agbeh"以外の文字列の場合
        """
        base = np.zeros((400, 1000), dtype=np.uint8)
        im_height, im_width = self.__raw.shape
        x_shift, y_shift = 500, 200 - im_height // 2
        base[y_shift : y_shift + im_height, 500 : 500 + im_width] = self.__raw

        if isinstance(q, str):
            if q == "agbeh":
                q = np.array([3.22, 4.27, 5.37, 6.48, 7.57, 8.61, 9.67])
            else:
                raise ValueError(f"invalid q specifier: {q}")

        arr_theta = self.__q2theta(q)
        coord = TiltCameraCoordinate(
            np.deg2rad(self.ini_values["psi"]),
            self.ini_values["cameraLength"],
            (self.ini_values["center_x"], self.ini_values["center_y"]),
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.subplots_adjust(bottom=0.2)

        im = base.copy()
        im = coord.drawPeaks(im, arr_theta)

        ax.imshow(im, cmap="jet")

        ax_l = fig.add_axes((0.1, 0.16, 0.7, 0.03))
        l_slider = Slider(
            ax=ax_l,
            label="l[px]",
            valmin=self.min_values["cameraLength"],
            valmax=self.max_values["cameraLength"],
            valinit=self.ini_values["cameraLength"],
        )
        ax_psi = fig.add_axes((0.1, 0.13, 0.7, 0.03))
        psi_slider = Slider(
            ax=ax_psi,
            label="psi[deg]",
            valmin=self.min_values["psi"],
            valmax=self.max_values["psi"],
            valinit=self.ini_values["psi"],
        )

        ax_x0 = fig.add_axes((0.1, 0.10, 0.7, 0.03))
        x0_slider = Slider(
            ax=ax_x0,
            label="x0[px]",
            valmin=self.min_values["center_x"],
            valmax=self.max_values["center_x"],
            valinit=self.ini_values["center_x"],
        )

        ax_y0 = fig.add_axes((0.1, 0.07, 0.7, 0.03))
        y0_slider = Slider(
            ax=ax_y0,
            label="y0[px]",
            valmin=self.min_values["center_y"],
            valmax=self.max_values["center_y"],
            valinit=self.ini_values["center_y"],
        )

        ax_save = fig.add_axes((0.8, 0.02, 0.1, 0.03))
        save_button = Button(ax_save, "Save")

        def update(val):
            coord.beamcenter = (x0_slider.val + x_shift, y0_slider.val + y_shift)
            coord.l = l_slider.val
            coord.psi = np.deg2rad(psi_slider.val)
            im = base.copy()
            im = coord.drawPeaks(im, arr_theta)
            ax.imshow(im, cmap="jet")

        def save(event):
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            fig.savefig(f"ipynb/{timestamp}.png", dpi=300)

        l_slider.on_changed(update)
        psi_slider.on_changed(update)
        x0_slider.on_changed(update)
        y0_slider.on_changed(update)
        save_button.on_clicked(save)

        fig.show()

        if imgpath:
            fig.savefig(imgpath, dpi=300)

        return coord


def tif2chi(
    src: str,
    *,
    paramfile="",
    center=(np.nan, np.nan),
    axis="r",
    cameraLength=np.nan,
    psi=0.0,
    kind: str = "patched",
    overwrite=False,
    suffix="",
    flip="",
) -> str:
    """tifファイルを読み込んで動径積分を行い、csvファイルに保存する

    Parameters
    ----------
    src : str
        tifファイルのパス
    kind : str
        tilted or patched
    center : tuple, optional
        ビームセンターの座標(x[px],y[px]), by default (np.nan, np.nan)
    cameraLength : float, optional
        カメラ長[px], kind=tiltedの場合のみ, by default np.nan
    psi : float, optional
        カメラ平面の傾き, kind=tiltedの場合のみ, by default 0.0
    overwrite : bool, optional
        Trueなら同名のファイルがあっても上書きする, by default False
    suffix : str, optional
        書き出しファイル名は<元ファイル名(拡張子抜き)>_<suffix>.csvにする, by default ""
    flip : str, optional
        "h"か"horizontal"なら画像を左右反転して読み込む(kind=tiltedの場合のみ), by default ""

    Returns
    -------
    str
        書き出したファイルのパス

    Raises
    ------
    FileNotFoundError
        srcのファイルが見つからない場合
    FileExistsError
        overwrite=Falseかつ書き出し先のファイルが既に存在する場合
    ValueError
        kindが不正な場合
    """
    if not os.path.isfile(src):
        raise FileNotFoundError(f"{src} not found")
    dist = src.replace(".tif", suffix + ".csv")
    if not overwrite and os.path.exists(dist):
        raise FileExistsError(f"{dist} already exists")

    if kind == "tilted":
        profile = TiltedSaxsImage.load_tiff(src, flip=flip)
        profile.center = center
        profile.psi, profile.cameraLength = psi, cameraLength
        i, x = profile.radial_average()
        param = f'param,"cener=({center[0]}, {center[1]})", cameraLength={cameraLength}px, psi={psi}deg'
        labels = "theta[deg],i"
    elif kind == "patched":
        profile = PatchedSaxsImage.load_tiff(src, paramfile=paramfile)
        if paramfile == "":
            profile.center = center
        param = f"param,center=({profile.center[0]}, {profile.center[1]})"
        if axis == "r":
            labels = "r[px],i"
            i, x = profile.radial_average(axis="r")
        elif axis == "2theta":
            labels = "2theta[deg],i"
            i, x = profile.radial_average(axis="2theta")
        elif axis == "q":
            labels = "q[nm^-1],i"
            i, x = profile.radial_average(axis="q")
        else:
            raise ValueError(f"invalid axis specifier {axis}")
        x = (x[:-1] + x[1:]) / 2
    else:
        raise ValueError("invalid type")
    header = "\n".join([f"src,{src}", param, labels])
    data = np.vstack([x, i]).T
    np.savetxt(dist, data, delimiter=",", header=header, fmt="%.6f")
    return dist


def seriesIntegrate(
    dir: str,
    *,
    paramfile="",
    center=(np.nan, np.nan),
    cameraLength=np.nan,
    axis="r",
    psi=0.0,
    kind: str = "patched",
    overwrite=False,
    heatmap=True,
    heatmap_xlim=(np.nan, np.nan),
    verbose=False,
    suffix="",
    flip="",
):
    """指定ディレクトリ内のtifファイルを読み込んで動径積分を行い、csvファイルに保存する

    Parameters
    ----------
    dir : str
        ディレクトリへのパス
    kind : str
        tilted or patched
    center : tuple
        ビームセンターの座標(x[px],y[px]), by default (np.nan, np.nan)
    cameraLength : _type_, optional
        カメラ長[px], by default np.nan
    psi : float, optional
        カメラ平面の傾き, by default 0.0
    overwrite : bool, optional
        Trueなら同名のファイルを無視して上書き, by default False
    heatmap : bool, optional
        Trueなら全ファイル積分後ヒートマップを作成する, by default True
    verbose : bool, optional
        Trueなら処理したファイル名を逐一printする, by default False
    suffix : str, optional
        ファイル名の接尾辞, by default ""
    flip : str, optional
        "h"か"horizontal"なら左右反転する, by default ""

    Raises
    ------
    FileNotFoundError
        dirのパスが存在しない場合
    ValueError
        kindが不正な場合
    """
    no_error = True
    if not os.path.isdir(dir):
        raise FileNotFoundError(f"{dir} not found")
    files = util.listFiles(dir, ext=".tif")
    print(f"{len(files)} file found")
    bar = tqdm.tqdm(total=len(files), disable=verbose)
    for i, file in enumerate(files):
        src = os.path.join(dir, file)
        try:
            if kind in ("tilted", "patched"):
                dist = tif2chi(
                    src,
                    kind=kind,
                    paramfile=paramfile,
                    center=center,
                    axis=axis,
                    psi=psi,
                    cameraLength=cameraLength,
                    overwrite=overwrite,
                    suffix=suffix,
                    flip=flip,
                )
            else:
                raise ValueError("invalid type")

            if verbose:
                print(f"{src} => {dist}")
            else:
                bar.update(1)
        except FileExistsError as e:
            print(f"\n{src} skipped because csv file already exists")
            no_error = False
        except Exception as e:
            print(f"\n{src} skipped because error occured:")
            print("  ", e)
            no_error = False

    print()

    if heatmap:
        from Saxs1dProfile import saveHeatmap

        if not no_error:
            print("WARNING : some files skipped")
        saveHeatmap(
            dir, overwrite=overwrite, load_axis=axis, save_axis=axis, x_lim=heatmap_xlim
        )
