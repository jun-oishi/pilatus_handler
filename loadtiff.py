from PIL import Image, TiffImagePlugin
import saxs_profile


class __Mask:
    def __init__(self, *args):
        # 禁止したいね
        pass

    @classmethod
    def createRectangle(cls, upper_left, lowwer_right):
        """
        create rectangle mask
        """
        mask = __Mask(upper_left, lowwer_right)
        return mask


class __Masks:
    """
    Mask class for SAXS profile

    attributes:
    - boundary: boundary of the mask
    - mask: mask
    """

    def __init__(self):
        """
        initialize
        """
        self.__masks = []
        return

    def add(self, mask: __Mask):
        """
        add mask
        """

        return


class TiffSaxsProfile:
    """
    SAXS profile class for TIFF images

    attributes:
    - raw: raw image
    - center: center of the image
    - mask: mask of the image
    """

    def __init__(self, path: str):
        self.__raw = Image.open(path)
        return

    def detect_center(self):
        """
        detect center of the profile
        """
        self.__center = type("pixel", (object,), {"x": 0, "y": 0})
        return

    def mask_boundary(self):
        """
        mask boundary of the profile
        """
        # self.__mask = mask
        return
