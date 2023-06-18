#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import os
import util

_logger = util.getLogger(__name__, level=util.DEBUG)


def _compress(arr: np.ndarray, dtype, min: int | None = None, max: int | None = None):
    """
    compress array
    """
    if min is None:
        min = arr.min()
    if max is None:
        max = arr.max()

    zero2one: np.ndarray = (arr - min) / (max - min)  # type: ignore
    if dtype == np.uint8:
        return (zero2one * ((1 << 8) - 1)).astype(dtype)
    elif dtype == np.uint16:
        return (zero2one * ((1 << 16) - 1)).astype(dtype)
    else:
        _logger.error(f"invalid dtype: {dtype}")
        raise ValueError("invalid dtype: only uint8 and uint16 are supported")


class _Masks(np.ndarray):
    """Mask list class for SAXS profile

    attributes:
        _masks: list of masks
    """

    def __new__(cls, mold):
        value = np.ones_like(mold)
        self = np.asarray(value, dtype=bool).view(cls)
        self.__masks = []
        return self

    def __init__(self, mold):
        """
        initialize with mold for shape
        """
        self.__masks = []
        return

    def __update(self):
        """
        update mask
        """
        self[:] = self * 0 + 1
        for mask in self.__masks:
            self *= mask
        return

    def append(self, new_mask: np.ndarray):
        """
        append mask
        """
        _logger.debug(f"append mask: shape={new_mask.shape} ,dtype={new_mask.dtype}")
        if new_mask.shape != self.shape:
            raise ValueError("invalid shape")
        self.__masks.append(new_mask)
        self *= new_mask
        return

    def __pop(self, index: int):
        self.__masks.pop(index)
        self.__update()
        return

    def add_rectangle(self, top: int, bottom: int, left: int, right: int):
        arr = np.ones_like(self)
        arr[top:bottom, left:right] = 0
        self.append(arr)

    def undo(self):
        self.__pop(-1)
        return


GREEN = (0, 255, 0)


class Saxs2dProfile:
    """SAXS 2D profile class

    Attributes:
        __raw: raw image
        __img: image after processing
        __masks: mask of the image
        __center:
    """

    DEFAULT_MARK_COLOR = GREEN

    def __new__(cls):
        raise NotImplementedError(f"{cls} default initializer not implemented")

    def __init__(self, raw: np.ndarray):
        _logger.debug(f"initializing Saxs2dProfile with raw:{id(raw):x}, {raw.shape}")
        self.__raw: np.ndarray = raw
        self.__buf: np.ndarray = np.zeros_like(raw)
        self.__masks: _Masks = _Masks(self.__raw)
        self.__center: tuple = (np.nan, np.nan)
        _logger.debug(f"id(self.__raw): {id(self.__raw)}")

    def shape(self):
        return self.__raw.shape

    @classmethod
    def __internal_new(cls):
        return object.__new__(cls)

    @classmethod
    def load_tiff(cls, path: str):
        ret = cls.__internal_new()
        if not os.path.exists(path):
            raise FileNotFoundError("")
        if path[-4:] != ".tif":
            raise ValueError("invalid file type")
        ret.__init__(cv2.imread(path, cv2.IMREAD_UNCHANGED))
        _logger.debug(f"max: {ret.__raw.max()}, min: {ret.__raw.min()}")
        return ret

    def values(
        self,
        log: bool = True,
        showMaskAsNan: bool = True,
        showCenterAsNan: bool = False,
    ) -> np.ndarray:
        self.__buf = self.__raw.copy()
        if showMaskAsNan:
            self.__buf = self.__buf.astype(np.float32)
            self.__buf[self.__masks == 0] = np.nan
        else:
            self.__buf *= self.__masks
        if log:
            self.__log()
        if showCenterAsNan:
            self.__draw_center()
        return self.__buf

    def center(self):
        return self.__center

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
        """
        update and save image
        returns 0 if success
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
                self.__buf[self.__masks == 0] = self.DEFAULT_MARK_COLOR
            if showCenter and self.__center[0] is not np.nan:
                self.__draw_center()
        cv2.imwrite(path, self.__buf)
        return 0

    def __log(self, nonPositiveValueAs=1):
        if nonPositiveValueAs == 1:
            self.__buf = np.log(np.maximum(self.__buf, 1))
        elif nonPositiveValueAs == np.nan:
            self.__buf[self.__buf <= 0] = np.nan
            self.__bug = np.log(self.__buf)

    def __draw_center(
        self,
        *,
        color="Nan or green",
        markerType: int = cv2.MARKER_TILTED_CROSS,
        markerSize=100,
        thickness=2,
    ):
        if self.__center[0] is np.nan:
            return
        if color == "Nan or green":
            if len(self.__buf.shape) == 2:
                color = np.nan
            else:
                color = (0, 255, 0)
        center = (int(self.__center[0]), int(self.__center[1]))
        cv2.drawMarker(self.__buf, center, color, markerType, markerSize, thickness)
        return

    def __compress(self, dtype=np.uint8, min=None, max=None):
        if min is None:
            min = self.__buf.min()
        if max is None:
            max = self.__buf.max()

        zero2one: np.ndarray = (self.__buf - min) / (max - min)  # type: ignore
        if dtype == np.uint8:
            self.__buf = (zero2one * ((1 << 8) - 1)).astype(dtype)
        elif dtype == np.uint16:
            self.__buf = (zero2one * ((1 << 16) - 1)).astype(dtype)
        else:
            _logger.error(f"invalid dtype: {dtype}")
            raise ValueError("invalid dtype: only uint8 and uint16 are supported")
        return

    def __toColor(self, cmap=cv2.COLORMAP_HOT):
        self.__compress()
        self.__buf = cv2.applyColorMap(self.__buf, cmap)
        return

    def add_rectangle_mask(self, top: int, bottom: int, left: int, right: int):
        self.__masks.add_rectangle(top, bottom, left, right)

    def auto_mask(self):
        self.__masks.append(self.__raw >= 0)  # nan=>0, otherwise=>1

    def detect_center(self):
        toDetect = self.__raw.copy()
        cutoff = np.median(toDetect)
        toDetect[toDetect < cutoff] = 0
        _logger.debug(
            f"toDetect shape: {toDetect.shape}, dtype: {toDetect.dtype}, max:{toDetect.max()}"
        )
        toDetect = _compress(toDetect, dtype=np.uint8)
        circles = cv2.HoughCircles(
            toDetect,
            cv2.HOUGH_GRADIENT,
            1,
            20,
            param1=50,
            param2=30,
            minRadius=0,
            maxRadius=0,
        )
        if circles is None:
            _logger.info("no circle detected")
        else:
            _logger.debug(
                f"toDetect.shape: {toDetect.shape}, toDetect.dtype: {toDetect.dtype}"
            )
            _logger.info(f"circle detected: {circles.shape}")
            self.__center = circles[0, 0, 0], circles[0, 0, 1]
