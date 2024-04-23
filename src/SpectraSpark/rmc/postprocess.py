import numpy as np
import re
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

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

class Config:
    def __init__(self, *, src: str="", gamma: int = 120,
                 La:int=0, Lb:int=0, pos_ab:np.ndarray=_EMPTY):
        if src != "":
            self.load(src, gamma=gamma)
        elif pos_ab.size > 0:
            self.set_data(La, Lb, pos_ab)

        self.rdf = np.empty(len(NN_DIST) - 1, dtype=float)

    def load(self, src: str, gamma:int = 120):
        self.src = src
        header = open(src).readline().strip()
        self.raw_header = header
        match = re.search(r"Lx:(\d+)", header)
        self.Lx = int(match.group(1)) if match is not None else 0
        match = re.search(r"Ly:(\d+)", header)
        self.Ly = int(match.group(1)) if match is not None else 0
        match = re.search(r"N:(\d+)", header)
        self.n = int(match.group(1)) if match is not None else 0
        match = re.search(r"expfile:(\S+)", header)
        self.expfile = match.group(1) if match is not None else ""

        self.x, self.y = np.loadtxt(src, dtype=int, comments=">>>")
        assert len(self.x) == self.n
        if gamma == 60:  # xyが120degの格子座標に変換して持つ
            self.y = self.Ly - 1 - self.y

        # neighbors[i][j] = [k1, k2, ...]  i番目のクラスタのj近接のクラスタのインデックスのリスト
        self.neighbors = [[[] for _nn in range(MAX_NN)] for _n in range(self.n)]


    def set_data(self, La:int, Lb:int, pos_ab:np.ndarray):
        self.Lx = La
        self.Ly = Lb
        self.n = len(pos_ab)
        self.x, self.y = pos_ab[:,0], pos_ab[:,1]
        self.expfile = ""

        # neighbors[i][j] = [k1, k2, ...]  i番目のクラスタのj近接のクラスタのインデックスのリスト
        self.neighbors = [[[] for _nn in range(MAX_NN)] for _n in range(self.n)]


    def dist(self, i, j, periodic=True):
        dx = self.x[i] - self.x[j]
        dy = self.y[i] - self.y[j]
        if not periodic:
            return np.sqrt((dx - dy * 0.5) ** 2 + (dy * np.sqrt(3) * 0.5) ** 2)

        dxx = (dx - self.Lx, dx, dx + self.Lx)
        dyy = (dy - self.Ly, dy, dy + self.Ly)
        ret = np.inf
        for x in dxx:
            for y in dyy:
                d = np.sqrt((x - y / 2) ** 2 + (y * np.sqrt(3) / 2) ** 2)
                ret = min(ret, d)
        return ret

    def count(self):
        _rdc = np.zeros((self.n, len(NN_DIST) - 1))
        for _i in range(self.n):
            for _j in range(_i + 1, self.n):
                d = self.dist(_i, _j)
                for _k in range(len(NN_DIST) - 1):
                    thresh = (NN_DIST[_k] + NN_DIST[_k + 1]) / 2
                    if d < thresh:
                        _rdc[_i, _k] += 1
                        _rdc[_j, _k] += 1
                        if d - self.dist(_i, _j, False) < 0:
                            break
                        self.neighbors[_i][_k].append(_j)
                        self.neighbors[_j][_k].append(_i)
                        break
        self.rdf[:] = np.mean(_rdc, axis=0)

    def plot(
        self, ax: Axes, fontsize: float = 16, showNN: tuple[int, ...] = (6, 7, 8, 9, 10)
    ):
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

    def save(
        self, figsize=(8, 5), dpi: int = 300, fontsize: float = 16, title: str = ""
    ):
        name = os.path.splitext(self.src)[0]
        dst = name + "_rdf.dat"
        if os.path.exists(dst):
            if input(f"Overwrite {dst}? [y/n] : ") != "y":
                print(f"Skip {dst}")
                return

        with open(dst, "w") as f:
            f.write(self.raw_header + "\n")
            f.write("rdf:\n")
            for i, rd in enumerate(self.rdf):
                f.write(f"  {i:>2d}NN   {rd:.6f}\n")
            f.write("\n")

            f.write(f"cluster arrangement({self.n}):\n")
            for i in range(self.n):
                f.write(f"  {self.x[i]:>5d}  {self.y[i]:>5d}\n")
            f.write("\n")
        print(f"Saved {dst}")

        # dst = name + "_nn.png"
        # if os.path.exists(dst):
        #     if input(f"Overwrite {dst}? [y/n] : ") != "y":
        #         print(f"Skip {dst}")
        #         return

        # fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        # title = title if title != "" else self.src
        # self.plot(ax, fontsize=fontsize)
        # ax.set_title(title, fontsize=fontsize)
        # fig.savefig(dst)

    def saveXtl(self, dst: str = ""):
        x = self.x / self.Lx
        y = self.y / self.Ly
        a = self.Lx * A_MG
        b = self.Ly * A_MG
        c = 3 * C_MG
        alpha, beta, gamma = 90, 90, 120
        if dst == "":
            dst = self.src.replace(".txt", ".xtl")
        if os.path.exists(dst):
            if input(f"Overwrite {dst}? [y/n] : ") != "y":
                print(f"Skip {dst}")
                return

        with open(dst, "w") as f:
            f.write(f"TITLE {dst}\n")
            f.write(f"CELL\n")
            f.write(f"  {a:.6f} {b:.6f} {c:.6f} {alpha:.6f} {beta:.6f} {gamma:.6f}\n")
            f.write("SYMMETRY NUMBER 1\n")
            f.write("SYMMETRY LABEL P1\n")
            f.write("ATOMS\n")
            f.write("NAME  X  Y  Z\n")
            for i in range(self.n):
                f.write(f"  L  {x[i]:.6f}  {y[i]:.6f}  0.000000\n")
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
    c = Config(src)

    if args[1] == "count":
        c.count()
        c.save()

    elif args[1] == "xtl":
        if len(args) > 3:
            dst = args[3]
        else:
            dst = ""
        c.saveXtl(dst)
