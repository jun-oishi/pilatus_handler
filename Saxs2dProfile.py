#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import os
import util

_logger = util.getLogger(__name__, level=util.DEBUG)


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


class Saxs2dProfile:
    """SAXS 2D profile class

    Attributes:
        __raw: raw image
        __img: image after processing
        _Masks: mask of the image
    """

    def __new__(cls):
        raise NotImplementedError(f"{cls} default initializer not implemented")

    def __init__(self, raw: np.ndarray):
        _logger.debug(f"initializing Saxs2dProfile with raw:{id(raw):x}, {raw.shape}")
        self.__raw = raw
        self.__masks = _Masks(self.__raw)
        self.__update()
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

    def __update(self):
        """update self.__img"""
        self.__img = self.__raw
        self.__apply_mask()
        _logger.debug(
            f"self.__img.shape : {self.__img.shape}, self.__img.dtype : {self.__img.dtype}"
        )

    def values(self, log: bool = True, showMask: bool = False) -> np.ndarray:
        self.__update()
        ret = self.__img
        if log:
            ret = np.log(np.maximum(self.__img, 1))
        # if showMask:
        #     ret +=
        return ret

    def save(
        self,
        path: str,
        *,
        overwrite: bool = False,
        log: bool = True,
        color: bool = False,
    ) -> int:
        """
        update and save image
        returns 0 if success
        """
        if (not overwrite) and (os.path.exists(path)):
            raise FileExistsError("")
        self.__update()
        toWrite = self.__img
        if log:
            toWrite = np.log(np.maximum(self.__img, 1))
        if color:
            max = toWrite.max()
            min = toWrite.min()
            _logger.debug(f"max: {max}, min: {min}")
            toWrite = ((toWrite - min) / (max - min)) * ((1 << 8) - 1)
            toWrite = np.asarray(toWrite, dtype=np.uint8)
            _logger.debug(f"max: {toWrite.max()}, min: {toWrite.min()}")
            toWrite = cv2.applyColorMap(toWrite, cv2.COLORMAP_JET)
        cv2.imwrite(path, toWrite)
        return 0

    def __apply_mask(self):
        """apply self._Masks to self.__img"""
        self.__img = self.__img * self.__masks

    def add_rectangle_mask(self, top: int, bottom: int, left: int, right: int):
        self.__masks.add_rectangle(top, bottom, left, right)

    def auto_mask(self):
        self.__masks.append(self.__raw >= -1)
