from Saxs2dProfile import Saxs2dProfile
from Saxs1dProfile import FileSeries
import util
import os
import numpy as np
import threading


class Cui:
    def __init__(self):
        self.x: float = np.nan
        self.y: float = np.nan
        self.overWrite = False

    def main(self):
        while True:
            try:
                if not self.wait():
                    break
            except Exception as e:
                print(f"error: {e}")

    def wait(self):
        raw = input("> ")
        command = raw.split(" ")[0]
        args = raw.split(" ")[1:]
        if command == "cd":
            os.chdir(args[0])
        elif command == "ls":
            print(os.listdir())
        elif command == "center":
            if len(args) == 2:
                self.x, self.y = float(args[0]), float(args[1])
            else:
                print(f"center set as {self.x, self.y}")
        elif command == "integrate":
            if self.confirmParam():
                self.integrate()
        elif command == "heatmap":
            self.headmap()
        elif command == "exit":
            return False
        else:
            print("invalid command")
        return True

    def confirmParam(self):
        if (np.isnan(self.x)) or (np.isnan(self.y)):
            print("center not set")
            return False
        print(f"dir       : {os.getcwd()}")
        print(f"center    : {self.x}, {self.y}")
        self.files = util.listFiles(os.getcwd())
        print(f"num files : {len(self.files)}")
        print(f"first file: {self.files[0]}")
        print(f"last file : {self.files[-1]}")
        confirm = input("confirm? (y/n)")
        if confirm == "y":
            return True
        else:
            return False

    def integrate(self):
        for file in self.files:
            print(f"processing {file}...", end="")
            profile = Saxs2dProfile.load_tiff(file)
            profile.auto_mask_invalid()
            profile.center = (self.x, self.y)
            i, bins = profile.radial_average(dr=1.0)
            r = (bins[:-1] + bins[1:]) / 2
            self.__saveCsv(file, r, i)
            print("done")
        return

    def __saveCsv(self, src, r, i):
        outfile = src.replace(".tif", ".csv")
        header = "\n".join([f"src,{src}", f"center,({self.x}, {self.y})", "r[px],i"])
        data = np.vstack([r, i]).T
        if not self.overWrite and os.path.exists(outfile):
            self.overWrite = (
                input(f"\n{outfile} already exists. overwrite? (y/n)") == "y"
            )
            if not self.overWrite:
                print("aborted")
                exit()
        np.savetxt(outfile, data, delimiter=",", header=header)

    def headmap(
        self,
    ):
        # TODO: I0による正規化
        files = FileSeries(os.getcwd())
        dist = files.heatmap()
        print(f"saved to {dist}")
        return


if __name__ == "__main__":
    cui = Cui()
    cui.main()
