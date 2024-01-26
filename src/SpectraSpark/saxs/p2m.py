import os, re, json
from typing import Tuple
import numpy as np
import cv2
import tqdm
from ..util import listFiles
from .qi2d import Saxs2d


class P2MImage:
    """Pilatusによる小角散乱画像を表すクラス"""

    def __init__(self, src: str, param: str = ""):
        self.__src = src
        if not os.path.exists(src):
            raise FileNotFoundError(f"{src} is not found.")
        self.__raw: np.ndarray = cv2.imread(src, cv2.IMREAD_UNCHANGED)
        self.__pixel_size = np.nan  # mm/px
        self.__camera_length = np.nan  # mm
        self.__wave_length = np.nan  # nm
        self.__center = (np.nan, np.nan)  # x, y [px]
        if param == "":
            defaultpath = os.path.join(os.path.dirname(src), "param")
            if os.path.exists(defaultpath + ".par"):
                self.load_fit2d_par(defaultpath + ".par")
            elif os.path.exists(defaultpath + ".json"):
                self.load_json_par(defaultpath + ".json")

        if param != "":
            param = os.path.join(os.path.dirname(src), param)
            if param.endswith(".par"):
                self.load_fit2d_par(param)
            elif param.endswith(".json"):
                self.load_json_par(param)
        return

    @property
    def center(self) -> Tuple[float, float]:
        return self.__center

    def load_fit2d_par(self, src: str):
        """fit2dのパラメタファイルからパラメタを読み込む"""
        lines = []
        with open(src) as f:
            lines = f.readlines()

        for line in lines:
            m = re.match(
                r"X/Y pixel sizes =[ ]+(\d+\.\d+),[ ]+(\d+\.\d+)[ ]+microns", line
            )
            if m:
                self.__pixel_size = float(m.group(1)) / 1000  # mm/px
                continue
            m = re.match(r"Sample to detector distance =[ ]+(\d+\.\d+)[ ]+mm", line)
            if m:
                self.__camera_length = float(m.group(1))
                continue
            m = re.match(r"Wavelength =[ ]+(\d+.\d+)[ ]+Angstroms", line)
            if m:
                self.__wave_length = float(m.group(1)) / 10
                continue
            m = re.match(
                r"X/Y pixel position of beam =[ ]+(\d+\.\d+),[ ]+(\d+\.\d+)", line
            )
            if m:
                self.__center = (float(m.group(1)), float(m.group(2)))
                continue
        return

    def load_json_par(self, src):
        with open(src) as f:
            data = json.load(f)
        self.__pixel_size = data["pixel_size"] / 1000
        self.__camera_length = data["camera_length"]
        self.__wave_length = data["wave_length"] / 10
        self.__center = (data["center_x"], data["center_y"])
        return

    def radial_average(self) -> Tuple[np.ndarray, np.ndarray]:
        s2p = self.toSaxs2d()
        return s2p.radial_average()

    def __px2q(self) -> float:
        return 2 * np.pi / self.__wave_length / self.__camera_length * self.__pixel_size

    def toChiFile(self, dst: str):
        """chiファイルを出力する"""
        i, q = self.radial_average()
        header = "\n".join(
            [
                f"src,{self.__src}",
                f"param,center=({self.__center[0]},{self.__center[1]}),px2q={self.__px2q()}",
                "q [nm^-1],i [a.u.]",
            ]
        )
        np.savetxt(dst, np.array([q, i]).T, fmt="%.6f", header=header, delimiter=",")

    def toSaxs2d(self) -> Saxs2d:
        """Saxs2dに変換"""
        px2q = self.__px2q()
        mask = self.__raw < 2
        i = self.__raw.astype(np.float32)
        i[mask] = np.nan
        return Saxs2d(i, px2q, self.__center)

    def detectCenter(self):
        """中心を検出する"""
        # TODO
        raise NotImplementedError

    @classmethod
    def seriesIntegrate(
        cls, dir: str, param: str, dst: str = "", overwrite: bool = False
    ):
        """指定ディレクトリ内の画像すべてを積分して1ファイルに出力する"""
        dir = os.path.dirname(dir)
        if not os.path.exists(dir):
            raise FileNotFoundError(f"{dir} is not found.")

        if dst == "":
            dst = dir + ".csv"
            if not overwrite and os.path.exists(dst):
                raise FileExistsError(f"{dst} already exists.")

        files = listFiles(dir, ext=".tif")
        print(f"")
        bar = tqdm.tqdm(total=len(files))
        q = cls(os.path.join(dir, files[0]), param).radial_average()[1]
        iq = np.empty((len(files), len(q)))

        for _i, src in enumerate(files):
            path = os.path.join(dir, src)
            try:
                iq[_i] = cls(path, param).radial_average()[0]
            except Exception as e:
                print(f"\n{path} skipped because of {e}")
                iq[_i, :] = np.nan
            bar.update(1)
        files.insert(0, "q[nm^-1]")
        header = ",".join(files)
        data = np.hstack((q.reshape(-1, 1), iq.T))
        np.savetxt(dst, data, fmt="%.6f", header=header, delimiter=",")
        bar.close()
