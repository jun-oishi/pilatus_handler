"""module for 2D SAXS profile"""

import numpy as np
import cv2
import os
import util

__version__ = "0.0.26"


_MASK_DTYPE = np.float32


class _Masks(np.ndarray):
    """array-like class for masks
    support operations like numpy.ndarray
    value is always 0 or 1, 0 for masked pixel

    Methods
    -------
    append(new_mask: np.ndarray)
        append new mask and update value of self
    """

    def __new__(cls, mold):
        value = np.ones_like(mold)
        self = np.asarray(value, dtype=_MASK_DTYPE).view(cls)
        self.__masks = []
        return self

    def __init__(self, mold):
        """initialize with mold for shape"""
        self.__masks = []
        return

    def __update(self) -> None:
        """update mask"""
        self[:] = self * 0 + 1
        for mask in self.__masks:
            self *= mask
        return

    def append(self, new_mask: np.ndarray) -> None:
        """append mask"""
        if new_mask.shape != self.shape:
            raise ValueError("invalid shape")
        toAppend = np.full_like(self, np.nan, dtype=_MASK_DTYPE)
        toAppend[new_mask > 0] = 1
        self.__masks.append(new_mask)
        self *= new_mask
        return


GREEN = (0, 255, 0)  # BGR


class Saxs2dProfile:
    """SAXS 2D profile class

    Methods
    -------
    values(log:bool=True, showMaskAsNan:bool=True, showCenterAsNan:bool=False)
        return modified values
    """

    DEFAULT_MARK_COLOR = GREEN

    def __new__(cls):
        raise NotImplementedError(f"{cls} default initializer not implemented")

    def __init__(self, raw: np.ndarray):
        self.__raw: np.ndarray = raw
        self.__buf: np.ndarray = np.zeros_like(raw)
        self.__masks: _Masks = _Masks(self.__raw)
        self.__center: tuple = (np.nan, np.nan)

    @property
    def shape(self) -> tuple[int, int]:
        return self.__raw.shape

    @classmethod
    def __internal_new(cls):
        return object.__new__(cls)

    @classmethod
    def load_tiff(cls, path: str, flip="") -> "Saxs2dProfile":
        """load profile from tiff file

        Parameters
        ----------
        path: str
            path to tiff file

        Returns
        -------
        Saxs2dProfile
        """
        ret = cls.__internal_new()
        if not os.path.exists(path):
            raise FileNotFoundError("")
        if path[-4:] != ".tif":
            raise ValueError("invalid file type")
        flipped = cv2.imread(path, cv2.IMREAD_UNCHANGED)[::-1, :]
        if "h" in flip or "horizontal" in flip:
            flipped = flipped[:, ::-1]
        if "v" in flip or "vertical" in flip:
            flipped = flipped[::-1, :]
        ret.__init__(flipped)
        return ret

    @classmethod
    def to_csv(cls, src: str, center: tuple[float, float], *, dr=1.0):
        """save radial average to csv file

        Parameters
        ----------
        path: str
            path to save
        center: tuple[float,float]
            center of circle
        """
        _self = cls.load_tiff(src)
        _self.auto_mask_invalid()
        _self.center = center
        i, bins = _self.radial_average(dr=dr)
        r = (bins[:-1] + bins[1:]) / 2
        dist = src.replace(".tif", ".csv")
        header = "\n".join(
            [f"src: {src}", f"center: ({center[0]}, {center[1]})", "r[px],i"]
        )
        data = np.vstack([r, i]).T
        np.savetxt(dist, data, delimiter=",", header=header, fmt="%.1f,%.10e")
        return dist

    @classmethod
    def to_series_csv(cls, dir: str, center: tuple[float, float], *, dr=1.0):
        files = util.listFiles(dir, ext=".tif")
        bins = np.arange(0, 0)
        intensity = np.empty((0, len(files)))
        for i, src in enumerate(files):
            _self = cls.load_tiff(os.path.join(dir, src))
            _self.auto_mask_invalid()
            _self.center = center
            if i == 0:
                _intensity, bins = _self.radial_average(dr=dr)
                intensity = np.empty((len(_intensity), len(files)))
                intensity[:, i] = _intensity
            else:
                intensity[:, i], _bins = _self.radial_average(bins=bins)

        dist = dir + "_series.csv"
        r = (bins[:-1] + bins[1:]) / 2
        header = "\n".join(
            [
                f"src: {dir}",
                f"center: ({center[0]}, {center[1]})",
                "r[px]," + ",".join(files),
            ]
        )
        np.savetxt(
            dist,
            np.hstack([r.reshape(-1, 1), intensity]),
            delimiter=",",
            header=header,
            fmt="%.1f," + ",".join(["%.10e"] * len(files)),
        )

    def values(
        self,
        *,
        log: bool = True,
        showCenterAsNan: bool = False,
        nan2zero: bool = False,
    ) -> np.ndarray:
        """get modified values

        Parameters
        ----------
        log: bool, default True
            if True, return ln(values), set nan for value <= 0
        showMaskAsNan: bool, default True
            if True, set nan for masked pixel
        showCenterAsNan: bool, default False
            if True, draw center mark(tilted cross) with nan value

        Returns
        -------
        np.ndarray
            the shape is same as raw data
        """
        self.__buf = self.__raw.copy()
        self.__buf = (self.__buf * self.__masks).astype(self.__buf.dtype)
        if log:
            self.__log()
        if showCenterAsNan:
            self.__draw_center()
        if nan2zero:
            self.__buf[np.isnan(self.__buf)] = 0

        return self.__buf

    @property
    def center(self) -> tuple[float, float]:
        return self.__center

    @center.setter
    def center(self, center: tuple[float, float]):
        try:
            if len(center) != 2:
                raise TypeError("")
        except TypeError:
            raise TypeError("center must be array-like of 2 floats")

        self.__center = (center[1], center[0])

    def setTiltParam(self, phi, l, x0, y0):
        """set parameters to calibrate tilt
        arguments:
            phi: tilt angle in deg
            l: distance from sample to detector [px]
            x0: x coordinate of center [px]
            y0: y coordinate of center [px]
        """
        self.__phi = phi
        self.__l = l
        self.center = (x0, y0)
        self.__tilt_calibrated = True

    def save(
        self,
        path: str,
        *,
        overwrite: bool = False,
        log: bool = True,
        color: bool = True,
        showMask: bool = True,
        showCenter: bool = False,
    ) -> int:
        """save modified image
        returns 0 if success

        Parameters
        ----------
        path: str
            path to save
        overwrite: bool, default False
            if True, overwrite existing file, if False, raise FileExistsError if file exists
        log: bool, default True
            if True, save ln(values), set nan for value <= 0
        color: bool, default True
            if True, save as color image
        showMask: bool, default True
            if True, show mask with green or nan, if False, mask is set as 0
        showCenter: bool, default False
            if True, draw center mark(tilted cross) with the same color as masked pixels
        """
        if (not overwrite) and (os.path.exists(path)):
            raise FileExistsError("")
        self.__buf = self.__raw.copy()
        self.__buf *= self.__masks
        if log:
            self.__log()
        if color:
            # shape=>(height,width,3), dtype=>uint8
            self.__toColor()
            if showMask:
                self.__buf[np.isnan(self.__masks)] = self.DEFAULT_MARK_COLOR
            if showCenter and self.__center[0] is not np.nan:
                self.__draw_center()
        cv2.imwrite(path, self.__buf)
        return 0

    def __log(self, nonPositiveValueAs=0) -> None:
        """update self.__buf with ln(self.__buf)

        Parameters
        ----------
        nonPositiveValueAs: 0 or np.nan, default 0
            value to set for non-positive value
        """
        if nonPositiveValueAs == 0:
            self.__buf = np.log(np.maximum(self.__buf, 1))
        elif nonPositiveValueAs == np.nan:
            self.__buf = np.log(self.__buf)
        else:
            raise ValueError("nonPositiveValueAs must be 0 or np.nan")
        return

    def __draw_center(
        self,
        *,
        color="Nan or green",
        markerType: int = cv2.MARKER_TILTED_CROSS,
        markerSize=100,
        thickness=2,
    ) -> None:
        """draw center mark on self.__buf

        Parameters
        ----------
        color: "Nan or green" or tuple of int, default "Nan or green"
            color of center mark, by default, green for colored image or nan for grayscale image
        markerType: int, default cv2.MARKER_TILTED_CROSS
            marker type of cv2.drawMarker
        markerSize: int, default 100
            marker size of cv2.drawMarker
        thickness: int, default 2
            thickness of cv2.drawMarker
        """
        if self.__center[0] is np.nan:
            return
        if color == "Nan or green":
            if len(self.__buf.shape) == 2:
                color = np.nan
            else:
                color = (0, 255, 0)
        center = (int(self.__center[1]), int(self.__center[0]))
        cv2.drawMarker(self.__buf, center, color, markerType, markerSize, thickness)
        return

    def __compress(self, dtype=np.uint8, *, min=None, max=None, setNanAs=None) -> None:
        """compress self.__buf to dtype

        Parameters
        ----------
        dtype: np.uint8 or np.uint16, default np.uint8
            dtype of compressed image
        min: numeric
            value to set as 0, by default, self.__buf.min()
        """
        if setNanAs is not None and np.any(self.__buf == np.nan):
            raise ValueError("self.__buf contains nan")

        if min is None:
            min = self.__buf.min()
        if max is None:
            max = self.__buf.max()

        zero2one: np.ndarray = (self.__buf - min) / (max - min)  # type: ignore
        if dtype == np.uint8:
            toCast = zero2one * ((1 << 8) - 1)
        elif dtype == np.uint16:
            toCast = zero2one * ((1 << 16) - 1)
        else:
            raise ValueError("invalid dtype: only uint8 and uint16 are supported")
        toCast[toCast == np.nan] = setNanAs
        self.__buf = toCast.astype(dtype)
        return

    def __toColor(self, cmap=cv2.COLORMAP_HOT) -> None:
        """convert self.__buf grayscale array to color image

        Parameters
        ----------
        cmap: cv2.COLORMAP_*, default cv2.COLORMAP_HOT
        """
        self.__compress()
        self.__buf = cv2.applyColorMap(self.__buf, cmap)
        return

    def auto_mask_invalid(self, thresh=2) -> None:
        """add mask for invalid pixels to self.__masks
        for data from pilatus sensor, negative values means invalid pixels
        """
        self.__masks.append(self.__raw >= thresh)  # nan=>0, otherwise=>1
        return

    def detect_center(self) -> tuple[float, float]:
        """detect center and set to self.__center
        detect center by cv2.HoughCircles
        if no circle is detected, no error raised and self.__center is not updated

        Returns
        -------
        center: tuple[float,float]
            center of circle, (nan,nan) if no circle is detected
        """
        self.__buf = self.__raw.copy()
        cutoff = np.median(self.__buf)
        self.__buf[self.__buf < cutoff] = 0
        self.__compress(dtype=np.uint8)
        circles = cv2.HoughCircles(
            self.__buf,
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
            self.__center = circles[0, 0, 1], circles[0, 0, 0]

        return self.center

    def radial_average(
        self,
        *,
        dr: float = np.nan,
        bins: np.ndarray = np.arange(0),
        range: tuple[float, float] = (np.nan, np.nan),
    ) -> tuple[np.ndarray, np.ndarray]:
        """compute radial average

        Parameters
        ----------
        dr: float
            radius step, ignored if bins is specified
        bins: np.ndarray
            edges of each segment with length (number of segments)+1, if specified, dr is ignored
        range: tuple[float, float]
            range of radius, ignored if bins is specified

        Returns
        -------
        intensity: np.ndarray
            integrated intensity for each segment
        bins: np.ndarray
            bin edges finally used, same as input bins if specified
        """
        buf = self.__raw.copy()

        dx = np.ones_like(buf) * np.arange(buf.shape[1]) - self.__center[1]
        dy = (
            np.ones_like(buf) * np.arange(buf.shape[0]).reshape(-1, 1)
            - self.__center[0]
        )
        r = np.sqrt(dx**2 + dy**2)
        buf = buf * self.__masks

        if bins.size == 0:
            if np.isnan(dr):
                raise ValueError("dr or bins must be specified")
            if range == (np.nan, np.nan):
                range = (np.nanmin(r), np.nanmax(r))  # type: ignore

            if (range[1] - range[0]) % dr > 1e-6:
                bins = np.arange(range[0], range[1] + dr, dr)
            else:
                bins = np.arange(range[0], range[1], dr)

        intensity = np.empty(bins.size - 1)
        cnt = np.histogram(r, bins=bins)[0]
        sum = np.histogram(r, bins=bins, weights=buf)[0]
        intensity = sum / cnt
        # for i in np.arange(bins.size - 1):
        #     filter = ((r >= bins[i]) & (r < bins[i + 1])).astype(np.float32)
        #     filter[filter == False] = np.nan
        #     tmp = buf * filter
        #     intensity[i] = np.nanmean(tmp)

        return intensity, bins

    def __thetaGrid(self):
        phi = np.deg2rad(self.__phi)
        x = np.arange(self.__raw.shape[1]) - self.__center[1]
        y = np.arange(self.__raw.shape[0]) - self.__center[0]
        xx, yy = np.meshgrid(x, y)
        t = np.sqrt((xx * np.cos(phi)) ** 2 + yy**2) / np.abs(
            xx * np.sin(phi) - self.__l
        )
        return np.rad2deg(np.arctan(t))

    def tiltCalibrate(
        self, dtheta=2**-6, min_theta=0, max_theta=90, *, autotrim=True
    ):
        """calibrate tilt
        keep longitudinal resolution and reset lateral axis to theta
        assume beam center is left outside of image
        """
        theta = self.__thetaGrid()
        if autotrim:
            min_theta, max_theta = np.nanmin(theta), np.nanmax(theta)
            arr_theta = np.arange(min_theta, max_theta, dtheta)
        else:
            arr_theta = np.arange(min_theta, max_theta + dtheta, dtheta)
        height = self.__raw.shape[0]
        width = len(arr_theta)
        calibrated = np.empty((height, width), dtype=np.float32)
        for y in range(height):
            calibrated[y] = np.interp(
                arr_theta,
                theta[y],
                self.__raw[y],
                left=np.nan,
                right=np.nan,
            )
        return calibrated, arr_theta


def tif2chi(src, center, dr=1.0, *, overwrite=False) -> str:
    """二次元散乱プロファイルのtifファイルをintegrateしてcsvに保存する"""
    if not os.path.isfile(src):
        raise FileNotFoundError(f"{src} not found")
    dist = src.replace(".tif", ".csv")
    if not overwrite and os.path.exists(dist):
        raise FileExistsError(f"{dist} already exists")

    profile = Saxs2dProfile.load_tiff(src)
    profile.auto_mask_invalid()
    profile.center = center
    i, bins = profile.radial_average(dr=dr)
    r = (bins[:-1] + bins[1:]) / 2
    header = "\n".join(
        [f"src,{src}", f'center,"({center[0]}, {center[1]})"', "r[px],i"]
    )
    data = np.vstack([r, i]).T
    np.savetxt(dist, data, delimiter=",", header=header)
    return dist


def seriesIntegrate(
    dir, center, dr=1.0, *, overwrite=False, heatmap=True, verbose=True
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
            dist = tif2chi(src, center, dr, overwrite=overwrite)
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

    if not verbose:
        print()

    if heatmap:
        from Saxs1dProfile import saveHeatmap

        if not no_error:
            print("WARNING : some files skipped")
        saveHeatmap(dir, overwrite=overwrite)
