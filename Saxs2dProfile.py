"""module for 2D SAXS profile"""

import numpy as np
import cv2
import os
import util

__version__ = "0.1.0"


GREEN = (0, 255, 0)  # BGR


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


class Saxs2dProfile:
    """SAXS 2D profile class

    attributes
    ----------
    _raw : np.ndarray
    _center : tuple[float, float]
        (x, y) coordinate of beam center in THIS ORDER [px]
    detector : str
        "pilatus" or "eiger"
    pixelSize : float [mm]
        pixel size of detector, set by setDetector
    """

    DEFAULT_MARK_COLOR = GREEN

    def __init__(self, raw: np.ndarray):
        self._raw: np.ndarray = raw
        self._center: tuple[float, float] = (np.nan, np.nan)

    def setDetector(self, detector: str):
        if detector == "pilatus":
            self.detector = "pilatus"
            self.pixelSize = 172e-3  # mm
        elif detector == "eiger":
            self.detector = "eiger"
            self.pixelSize = 75e-3  # mm

    @property
    def raw(self) -> np.ndarray:
        return self._raw

    @property
    def shape(self) -> tuple[int, int]:
        return self._raw.shape  # type: ignore

    @classmethod
    def load_tiff(cls, path: str, flip=""):
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
        flipped = cv2.imread(path, cv2.IMREAD_UNCHANGED)[::-1, :]
        if "h" in flip or "horizontal" in flip:
            flipped = flipped[:, ::-1]
        if "v" in flip or "vertical" in flip:
            flipped = flipped[::-1, :]
        return cls(flipped)

    @property
    def center(self) -> tuple[float, float]:
        return self._center

    @center.setter
    def center(self, center: tuple[float, float]):
        """set beam center by (x,y) coordinate"""
        try:
            if len(center) != 2:
                raise TypeError("")
        except TypeError:
            raise TypeError("center must be array-like of 2 floats")

        self._center = (center[1], center[0])


class PatchedSaxsImage(Saxs2dProfile):
    """
    attributes
    ----------
    __mask : np.ndarray of bool
    cameraLength : float [mm]
    """

    def __init__(self, raw: np.ndarray):
        super().__init__(raw)
        self.__mask = np.zeros_like(raw, dtype=bool)
        self.cameraLength = np.nan  # mm

    @classmethod
    def load_tiff(cls, path: str, flip="") -> "PatchedSaxsImage":
        return super().load_tiff(path, flip)

    def auto_mask_invalid(self, thresh=2) -> None:
        """add mask for invalid pixels to self.__masks
        for data from pilatus sensor, negative values means invalid pixels
        """
        self.__mask = self.__mask | (self._raw < thresh)
        return

    def detect_center(self) -> tuple[float, float]:
        """detect center and set to self._center
        detect center by cv2.HoughCircles
        if no circle is detected, no error raised and self._center is not updated

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

    def radial_average(
        self, *, axis: str = "r", dtheta: float = 2**-6
    ) -> tuple[np.ndarray, np.ndarray]:
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
        buf = self._raw.copy()

        dx = np.ones_like(buf) * np.arange(buf.shape[1]) - self._center[1]
        dy = (
            np.ones_like(buf) * np.arange(buf.shape[0]).reshape(-1, 1) - self._center[0]
        )
        r = np.sqrt(dx**2 + dy**2)

        if axis == "r":
            dr = 1
            r_min = int(np.floor(np.min(r)))
            r_max = int(np.ceil(np.max(r)))
        elif axis == "theta":
            if self.cameraLength is np.nan:
                raise ValueError("cameraLength must be specified")
            dr = dtheta
            r = np.rad2deg(np.arctan(r / (self.cameraLength / self.pixelSize)))
            r_min = 0
            r_max = np.max(r)
        else:
            raise ValueError("invalid axis")

        bins = np.arange(r_min, r_max + dr, dr)
        buf = buf * self.__mask

        intensity = np.empty(bins.size - 1)
        cnt = np.histogram(r, bins=bins)[0]
        sum = np.histogram(r, bins=bins, weights=buf)[0]
        intensity = sum / cnt

        return intensity, bins


class TiltedSaxsImage(Saxs2dProfile):
    """
    Attributes
    ----------
    phi : float
        tilt angle of detector [deg]
    cameraLength : float [px]
        camera length measured along beam direction [px]
    center : tuple[float, float]
        (x, y) coordinate of beam center [px]
        methods assume x < 0 so that small x corresponds to small theta
    """

    def __init__(self, raw: np.ndarray):
        super().__init__(raw)
        self.phi: float = 0  # degree
        self.cameraLength: float = np.nan  # px
        self.__converted: np.ndarray = np.array([])
        self.__arr_theta_deg: np.ndarray = np.array([])

    @classmethod
    def load_tiff(cls, path: str, flip="") -> "TiltedSaxsImage":
        return super().load_tiff(path, flip)

    def __thetaGrid(self):
        """return theta grid for each pixel in degree"""
        phi = np.deg2rad(self.phi)
        x = np.arange(self._raw.shape[1]) - self._center[1]
        y = np.arange(self._raw.shape[0]) - self._center[0]
        xx, yy = np.meshgrid(x, y)
        t = np.sqrt((xx * np.cos(phi)) ** 2 + yy**2) / np.abs(
            xx * np.sin(phi) - self.cameraLength
        )
        return np.rad2deg(np.arctan(t))

    def copyCameraParam(self, other: "TiltedSaxsImage"):
        self.phi, self.cameraLength = other.phi, other.cameraLength
        self.center = (other.center[1], other.center[0])
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
        converted = converted / np.cos(np.deg2rad(arr_theta - self.phi))
        self.__converted = converted
        self.__arr_theta_deg = arr_theta
        return converted, arr_theta

    def radial_average(
        self, dtheta=2**-6, min_theta=0, max_theta=90, *, autotrim=True
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.__arr_theta_deg.size == 0:
            self.convert2theta(dtheta, min_theta, max_theta, autotrim=autotrim)
        return np.nanmean(self.__converted, axis=0), self.__arr_theta_deg


def tif2chi(
    src,
    *,
    center,
    cameraLength=np.nan,
    phi=0,
    kind,
    overwrite=False,
    suffix="",
    flip="",
) -> str:
    """二次元散乱プロファイルのtifファイルをintegrateしてcsvに保存する"""
    if not os.path.isfile(src):
        raise FileNotFoundError(f"{src} not found")
    dist = src.replace(".tif", suffix + ".csv")
    if not overwrite and os.path.exists(dist):
        raise FileExistsError(f"{dist} already exists")

    if kind == "tilted":
        profile = TiltedSaxsImage.load_tiff(src, flip=flip)
        profile.center = center
        profile.phi, profile.cameraLength = phi, cameraLength
        i, x = profile.radial_average()
        param = f'param,"cener=({center[0]}, {center[1]})", cameraLength={cameraLength}px, phi={phi}deg'
        labels = "theta[deg],i"
    elif kind == "patched":
        profile = PatchedSaxsImage.load_tiff(src)
        profile.auto_mask_invalid()
        profile.center = center
        axis = "x"
        param = f"param,center=({center[0]}, {center[1]})"
        labels = "r[px],i"
        if cameraLength is not np.nan:
            profile.cameraLength = cameraLength
            axis = "theta"
            param = param + f", cameraLength={cameraLength}mm"
            labels = "theta[deg],i"
        i, x = profile.radial_average(axis=axis)
        x = (x[:-1] + x[1:]) / 2
    else:
        raise ValueError("invalid type")
    header = "\n".join([f"src,{src}", param, labels])
    data = np.vstack([x, i]).T
    np.savetxt(dist, data, delimiter=",", header=header, fmt="%.6f")
    return dist


def seriesIntegrate(
    dir,
    *,
    center,
    cameraLength=np.nan,
    phi=0,
    kind,
    overwrite=False,
    heatmap=True,
    verbose=False,
    suffix="",
    flip="",
):
    """指定ディレクトリ内のtifファイルをintegrateしてcsvに保存する"""
    no_error = True
    if not os.path.isdir(dir):
        raise FileNotFoundError(f"{dir} not found")
    files = util.listFiles(dir, ext=".tif")
    print(f"{len(files)} file found")
    for i, file in enumerate(files):
        src = os.path.join(dir, file)
        try:
            if kind == "tilted":
                dist = tif2chi(
                    src,
                    kind=kind,
                    center=center,
                    phi=phi,
                    cameraLength=cameraLength,
                    overwrite=overwrite,
                    suffix=suffix,
                    flip=flip,
                )
            elif kind == "patched":
                dist = tif2chi(
                    src,
                    kind=kind,
                    center=center,
                    cameraLength=cameraLength,
                    overwrite=overwrite,
                    suffix=suffix,
                )
            else:
                raise ValueError("invalid type")

            if verbose:
                print(f"{src} => {dist}")
            else:
                print("#", end="" if (i + 1) % 40 else "\n", flush=True)
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
        saveHeatmap(dir, overwrite=overwrite)
