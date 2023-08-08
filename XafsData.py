import numpy as np
import util
import re

__version__ = "0.0.1"

_logger = util.getLogger(__name__)


def _split_by_spaces(line: str):
    return re.split(r" +", line.strip())


class XafsData:
    """XAFS data class
    attributes
    ----------
    fileformat : str
        file format of the data (now only 9809 is supported)
    src : str
        path to the data file
    beamline : str
    sampleinfo : str
    ring : str
    mono : str
    param : list[str]
    energy : np.ndarray
    data : np.ndarray
        2d array of the data (data[n_point, n_columns])
    """

    def __init__(self, src: str, *, fileformat: str = "9809", cols: list[int] = []):
        if fileformat == "9809":
            self.fileformat = "9809"
            self.load_9809(src, cols=cols)
        else:
            raise ValueError(f"invalid fileformat: {fileformat}")
        return

    def load_9809(self, src: str, cols: list[int] = []):
        with open(src, "r") as f:
            lines = f.readlines()

        self.src = src
        self.beamline = lines[0].strip()
        self.sampleinfo = lines[1].strip()
        self.ring = lines[3].strip()
        self.mono = lines[4].strip()
        self.param = [lines[5].strip(), lines[6].strip()]
        if match := re.search(r"Points=( +)([0-9]+)", self.param[0]):
            n_points = int(match.group(2))
        else:
            raise ValueError(f"invalid file format: {src}\n  cannot find Points=...")
        if match := re.search(r"Block =( +)([0-9]+)", self.param[1]):
            n_blocks = int(match.group(2))
        else:
            raise ValueError(f"invalid file format: {src}\n  cannot find Block=...")

        self.__load_blocks([line.strip() for line in lines[9 : 9 + n_blocks]])
        if self.energy.size != n_points:
            raise ValueError(
                f"file may be damaged: {src}\n  energy.size:{self.energy.size} != n_points:{n_points}"
            )

        data_starts = 9 + n_blocks + 4
        channels = _split_by_spaces(lines[data_starts - 1])
        if cols == []:
            cols = list(range(len(channels)))
        self.data = np.loadtxt(src, skiprows=data_starts, usecols=cols)

        if self.data.shape[0] != n_points:
            raise ValueError(
                f"file may be damaged: {src}\n  data.shape[1]:{self.data.shape[1]} != n_points:{n_points}"
            )

        return

    def __load_blocks(self, lines: list[str]):
        table = [_split_by_spaces(line) for line in lines]
        ls_energy = []
        for row in table:
            init_energy = float(row[1])
            fin_energy = float(row[2])
            num = int(row[5])
            ls_energy.append(np.linspace(init_energy, fin_energy, num))
        self.energy = np.concatenate(ls_energy)
        return
