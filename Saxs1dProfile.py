import numpy as np
import os
import matplotlib.pyplot as plt
import util

__version__ = "0.0.0"


class Saxs1dProfile:
    def __init__(self):
        self.r: np.ndarray = np.array([])
        self.i: np.ndarray = np.array([])

    @classmethod
    def load_csv(cls, path, *, delimiter=",", skiprows=4):
        obj = cls()
        try:
            table = np.loadtxt(path, delimiter=delimiter, skiprows=skiprows)
        except:
            table = np.loadtxt(
                path, delimiter=delimiter, skiprows=skiprows, encoding="cp932"
            )
        obj.r = table[:, 0]
        obj.i = table[:, 1]
        return obj


class FileSeries:
    def __init__(self, dir):
        self.dir = dir

    def loadFiles(self):
        self.fileNames = util.listFiles(self.dir, ext=".csv")
        self.files = [Saxs1dProfile.load_csv(f) for f in self.fileNames]

    def heatmap(self):
        self.loadFiles()

        x = self.files[0].r
        y = np.arange(len(self.files))
        z = np.array([np.log(f.i) for f in self.files])
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        contf = ax.contourf(x, y, z, levels=100, cmap="rainbow")
        ax.set_ylabel("file number")
        ax.set_xlabel("r[px]")
        plt.colorbar(contf).set_label("ln(I)")
        ax.set_title(os.path.basename(self.dir))
        dist = os.path.join(self.dir + ".png")
        fig.savefig(dist, dpi=300)
        return dist
