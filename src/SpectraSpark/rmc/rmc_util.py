import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import cv2


NN_DIST = (
    0,  # 0NN
    1,  # 1NN
    np.sqrt(3),  # 2NN
    2,  # 3NN
    np.sqrt(7),
    3,
    2 * np.sqrt(3),
    np.sqrt(13),
    4,
    np.sqrt(19),
    np.sqrt(21),
    5,
    3 * np.sqrt(3),
    np.sqrt(28),  # 14NN
)
NN_COLORS = (
    "gray",  # 0NN
    "gray",  # 1NN
    "gray",  # 2NN
    "gray",  # 3NN
    "gray",  # 4NN
    "gray",  # 5NN
    "blue",  # 6NN
    "red",  # 7NN
    "lime",  # 8NN
    "orange",  # 9NN
    "cyan",  # 10NN
    "magenta",  # 11NN
    "yellow",  # 12NN
    "gray",  # 13NN
)
MAX_NN = len(NN_DIST) - 1

A_MG = 3.21  # Angstrom
C_MG = 5.21  # Angstrom

_EMPTY = np.empty(0, dtype=float)

def load_img(src:str) -> np.ndarray:
    if src.endswith("tiff") or src.endswith("tif"):
        return cv2.imread(src, cv2.IMREAD_UNCHANGED)
    else:
        """
        qx:
            width
            qx1, qx2, ...
        qy:
            height
            qy1, qy2, ...
        i:
            i[0,0], i[0,1], ...
            ...
            i[height-1,0], ...
        """
        return np.loadtxt(src, skiprows=7)

def write_img(dst:str, qx:np.ndarray, qy:np.ndarray, img:np.ndarray,
              *, overwrite=False) -> None:
    assert(qx.size == img.shape[1] and qy.size == img.shape[0])
    if not overwrite and os.path.exists(dst):
        raise FileExistsError(f"{dst} already exists")
    with open(dst, "w") as f:
        f.write("qx:\n")
        f.write(f"  {qx.size}\n")
        f.write("  " + " ".join([f"{x:.6f}" for x in qx]) + "\n")
        f.write("qy:\n")
        f.write(f"  {qy.size}\n")
        f.write("  " + " ".join([f"{y:.6f}" for y in qy]) + "\n")
        f.write("i:\n")
        for i in range(img.shape[0]):
            f.write("  " + " ".join([f"{x:.6f}" for x in img[i]]) + "\n")
    return

class Config:
    def __init__(self, *, src: str="",
                 La:int=0, Lb:int=0, pos_ab:np.ndarray=_EMPTY):
        """initialize instance
        initialize the instance by loading a file or giving array
        see `load` and `set_data`

        Parameters
        ----------
        src : str, optional, keyword-only
            see `load`, by default ""
        gamma : int, optional, keyword-only
            see `load`, by default 120
        La : int, optional, keyword-only
            see `set_data`, by default 0
        Lb : int, optional, keyword-only
            see `set_data`, by default 0
        pos_ab : np.ndarray, optional, keyword-only
            see `set_data`, by default _EMPTY
        """
        if src != "":
            self.load(src)
        elif pos_ab.size > 0:
            self.set_data(La, Lb, pos_ab)

        self.rdf = np.empty(len(NN_DIST) - 1, dtype=float)

    def load(self, src: str) -> None:
        """load configuration from a file
        file format:

        >>>
        comments:
          ...
        Lx Ly N:
          {Lx} {Ly} {N}
        x:
          x1 x2 x3 ...
        y:
          y1 y2 y3 ...

        Parameters
        ----------
        src : str
            file path
        gamma : int, optional
            angle formed by 2 lattice vectors in degree, by default 120
        """
        self.src = src
        with open(src) as f:
            lines = f.readlines()
        header = lines[3]
        self.Lx, self.Ly, self.n = [int(s) for s in header.strip().split()]
        _x, _y = lines[5], lines[7]
        self.x = np.array([int(s) for s in _x.strip().split()])
        self.y = np.array([int(s) for s in _y.strip().split()])
        assert len(self.x) == self.n and len(self.y) == self.n

        # neighbors[i][j] = [k1, k2, ...]  i番目のクラスタのj近接のクラスタのインデックスのリスト
        self.neighbors = [[[] for _nn in range(MAX_NN)] for _n in range(self.n)]

    def set_data(self, La:int, Lb:int, pos_ab:np.ndarray):
        """set configuration by array
        array interpreted as configration in 120deg lattice coordinate

        Parameters
        ----------
        La : int
            model size in x direction
        Lb : int
            model size in y direction
        pos_ab : np.ndarray
            2-column array of cluster positions
        """
        self.Lx = La
        self.Ly = Lb
        self.n = pos_ab.shape[0]
        self.x, self.y = pos_ab[:,0], pos_ab[:,1]
        self.expfile = ""

        # neighbors[i][j] = [k1, k2, ...]  i番目のクラスタのj近接のクラスタのインデックスのリスト
        self.neighbors = [[[] for _nn in range(MAX_NN)] for _n in range(self.n)]


    def __dist(self, i, j, periodic=True) -> float:
        """compute distance between i-th and j-th cluster

        Parameters
        ----------
        i : int
            index of cluster i
        j : int
            index of cluster j
        periodic : bool, optional
            if True, consider periodic boundary condition and return minimum distance, by default True

        Returns
        -------
        float
            distance between i-th and j-th cluster **IN LATTICE UNIT**
        """
        dx = self.x[i] - self.x[j]
        dy = self.y[i] - self.y[j]
        if not periodic:
            return np.sqrt((dx - dy * 0.5) ** 2 + (dy * np.sqrt(3) * 0.5) ** 2)

        dxx, dyy = np.meshgrid(
            (dx - self.Lx, dx, dx + self.Lx),
            (dy - self.Ly, dy, dy + self.Ly)
        )
        dst = np.sqrt((dxx - dyy / 2) ** 2 + (dyy * np.sqrt(3) / 2) ** 2)
        return np.min(dst)

    def compute_rdf(self):
        """compute radial distribution function
        count the number of pairs of clusters that are within each NN distance
        """
        _rdc = np.zeros((self.n, len(NN_DIST) - 1))
        for _i in range(self.n):
            for _j in range(_i + 1, self.n):
                d = self.__dist(_i, _j)
                for _k in range(len(NN_DIST) - 1):
                    thresh = (NN_DIST[_k] + NN_DIST[_k + 1]) / 2
                    if d < thresh:
                        _rdc[_i, _k] += 1
                        _rdc[_j, _k] += 1
                        if d - self.__dist(_i, _j, False) < 0:
                            break
                        self.neighbors[_i][_k].append(_j)
                        self.neighbors[_j][_k].append(_i)
                        break
        self.rdf[:] = np.mean(_rdc, axis=0)

    def plot(self, ax: Axes, fontsize: float = 16, *,
             showNN: tuple[int, ...] = ()):
        """plot the configuration

        Parameters
        ----------
        ax : Axes
            matplotlib axes
        fontsize : float, optional
            fontsize of the plot, by default 16
        showNN : tuple[int, ...], optional
            list of NN to show, by default ()
        """

        ax.clear()
        ax.set_aspect("equal")
        ax.axis("off")

        corners = (
            (0, 0),
            (self.Lx * A_MG, 0),
            (self.Lx * A_MG - self.Ly * A_MG / 2, self.Ly * A_MG * np.sqrt(3) / 2),
            (-self.Ly * A_MG / 2, self.Ly * A_MG * np.sqrt(3) / 2),
        )
        for i in range(4):
            j = (i + 1) % 4
            ax.plot(
                [corners[i][0], corners[j][0]], [corners[i][1], corners[j][1]], "k-"
            )
        scalebar_len = 100  # Angstrom
        scalebar_end = (-30, 30)
        ax.plot(
            (scalebar_end[0], scalebar_end[0] - scalebar_len),
            (scalebar_end[1], scalebar_end[1]),
            "k-",
        )
        ax.text(
            scalebar_end[0] - scalebar_len / 2,
            scalebar_end[1] - 5,
            f"{scalebar_len*0.1} nm",
            ha="center",
            va="top",
            fontsize=fontsize,
        )

        for i in range(self.n):
            x = self.x[i] * A_MG - self.y[i] * A_MG * 0.5
            y = self.y[i] * A_MG * np.sqrt(3) * 0.5
            ax.plot(x, y, "o", color="black")

            for k in showNN:
                for j in self.neighbors[i][k]:
                    if i < j:
                        continue
                    x2 = self.x[j] * A_MG - self.y[j] * A_MG * 0.5
                    y2 = self.y[j] * A_MG * np.sqrt(3) * 0.5
                    ax.plot([x, x2], [y, y2], color=NN_COLORS[k])

        for k in showNN:
            ax.plot([0, 0], [0, 0], color=NN_COLORS[k], label=f"{k}NN")
        ax.legend(fontsize=fontsize, loc="upper right")

    def save_rdf(self, title:str="SRC", *,
             figsize=(8, 5), dpi: int = 300, overwrite: bool = False,
             fontsize: float = 16, showNN: tuple[int, ...] = ()) -> None:
        """save rdf as a text and plot as a .png file

        Parameters
        ----------
        title : str, optional
            title of the figure, by default src file name used
        figsize : tuple, optional
            figure size, by default (8, 5)
        dpi : int, optional
            dpi of the figure, by default 300
        overwrite : bool, optional
            if True, overwrite the existing file, by default False
        fontsize : float, optional
            see `plot`, by default 16
        showNN : tuple[int, ...], optional
            see `plot`, by default ()
        """
        if title == "SRC":
            name = os.path.splitext(self.src)[0]
            title = name
        elif title:
            name = title
        else:
            raise ValueError("title must be given")

        dst = name + "_rdf.dat"
        if not overwrite and os.path.exists(dst):
            raise FileExistsError(f"{dst} already exists")

        with open(dst, "w") as f:
            f.write(f"comments:\n  {title}\n")
            f.write("Lx, Ly, N:\n  {self.Lx} {self.Ly} {self.n}\n")
            f.write("x:\n")
            f.write("  " + " ".join([f"{i:>3d}" for i in self.x]) + "\n")
            f.write("y:\n")
            f.write("  " + " ".join([f"{i:>3d}" for i in self.y]) + "\n")
            f.write("rdf:\n")
            f.write("  " + " ".join([f"{i:>6d}NN" for i in range(len(self.rdf))]) + "\n")
            f.write("  " + " ".join([f"{rd:.6f}" for rd in self.rdf]) + "\n")
            f.write("\n")
        print(f"Saved {dst}")

        dst = name + "_nn.png"
        if not overwrite and os.path.exists(dst):
            raise FileExistsError(f"{dst} already exists")

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        self.plot(ax, fontsize=fontsize)
        ax.set_title(title, fontsize=fontsize)
        fig.savefig(dst)

    def saveXtl(self, dst: str = "", *, overwrite: bool = False):
        """save configuration as a .xtl file

        Parameters
        ----------
        dst : str, optional
            filename to save, by default ""
        overwrite : bool, optional
            if True, overwrite the existing file, by default False
        """
        if dst == "":
            dst = self.src.replace(".txt", ".xtl")
        elif not dst.endswith(".xtl"):
            dst += ".xtl"

        if not overwrite and os.path.exists(dst):
            raise FileExistsError(f"{dst} already exists")

        x = self.x / self.Lx
        y = self.y / self.Ly
        a = self.Lx * A_MG
        b = self.Ly * A_MG
        c = 3 * C_MG
        alpha, beta, gamma = 90, 90, 120

        with open(dst, "w") as f:
            f.write(f"TITLE {dst}\n")
            f.write("CELL\n")
            f.write(f"  {a:.6f} {b:.6f} {c:.6f} {alpha:.6f} {beta:.6f} {gamma:.6f}\n")
            f.write("SYMMETRY NUMBER 1\n")
            f.write("SYMMETRY LABEL P1\n")
            f.write("ATOMS\n")
            f.write("NAME  X  Y  Z\n")
            for i in range(self.n):
                f.write(f"  L  {x[i]:.6f}  {y[i]:.6f}  0.500000\n")
            f.write("EOF\n")
        return


if __name__ == "__main__":
    args = sys.argv
    if len(args) < 2:
        print("Usage: python stats.py count [src]")
        print("       python stats.py xtl [src] [dst]")
        exit()

    if len(args) > 2:
        src = args[2]
    else:
        src = input("Input file: ")
    c = Config(src=src)

    if args[1] == "rdf":
        c.compute_rdf()
        c.save_rdf()

    elif args[1] == "xtl":
        if len(args) > 3:
            dst = args[3]
        else:
            dst = ""
        c.saveXtl(dst)
