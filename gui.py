from Saxs2dProfile import Saxs2dProfile
import PySimpleGUI as sg
import util

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk  # type: ignore
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

logger = util.getLogger(__name__, util.DEBUG)

TMPDIR = "tmp/"

WINDOW_SIZE = (500, 750)
CANVAS_SIZE = (500, 600)

STATE_INIT = "init"
STATE_WAIT_AUTO_MASK = "wait_auto_mask"
STATE_WAIT_DETECT_CENTER = "wait_detect_center"
STATE_WAIT_SELECT_CENTER = "wait_select_center"
STATE_WAIT_INTEGRATE = "wait_integrate"


class FlushableFigCanvas:
    """canvas for matplotlib figure on tkinter"""

    def __init__(self, canvas: sg.tkinter.Canvas):
        self.__canvas = canvas
        self.__canvas_packed = {}
        self.__fig_agg: FigureCanvasTkAgg = None  # type: ignore
        return

    def draw(self, figure: Figure) -> None:
        """refresh canvas and draw figure

        Parameters
        ----------
        figure : mlp.figure.Figure
            figure to draw
        """
        if self.__fig_agg is not None:
            self.__flush()
        self.__fig_agg = FigureCanvasTkAgg(figure, self.__canvas)
        self.__fig_agg.draw()
        widget = self.__fig_agg.get_tk_widget()
        if widget not in self.__canvas_packed:
            self.__canvas_packed[widget] = True
            widget.pack(side="top", fill="both", expand=1)
        return

    def __flush(self) -> None:
        """remove figure"""
        self.__fig_agg.get_tk_widget().forget()
        try:
            self.__canvas_packed.pop(self.__fig_agg.get_tk_widget())
        except Exception as e:
            logger.error(f"error removing {self.__fig_agg}: {e}")
        plt.close("all")
        return

    def heatmap(self, data: np.ndarray) -> None:
        """flush canvas and draw heatmap"""
        fig = Figure()
        ax = fig.add_subplot(111)
        cmap: mpl.colormap.Colormap = mpl.colormaps.get_cmap("hot").copy()  # type: ignore
        cmap.set_bad("lime", alpha=1.0)
        im = ax.imshow(data, cmap=cmap)
        fig.colorbar(im, ax=ax)
        self.draw(fig)
        return

    def plot(self, x: np.ndarray, y: np.ndarray) -> None:
        """flush canvas and draw line plot"""
        fig = Figure()
        ax = fig.add_subplot(111)
        ax.plot(x, y)
        self.draw(fig)
        return


def main():
    layout = [
        [
            sg.Text(f"path:"),
            sg.InputText(
                default_text="testdata/s194sUn/s194sUn00000_00480.tif",
                key="-INPUT_FILEPATH-",
            ),
        ],
        [sg.Button("-", key="-BUTTON_ACTION-")],
        [
            sg.Text("status:", key="-TEXT_STATUS_HEADER-"),
            sg.Text("successfully initiated", key="-TEXT_STATUS-"),
        ],
        [sg.Canvas(size=CANVAS_SIZE, key="-CANVAS-")],
        [
            sg.Button("exit", key="-BUTTON_EXIT-"),
            sg.Button("save", key="-BUTTON_SAVE-"),
        ],
    ]

    window = sg.Window("test", layout, size=WINDOW_SIZE, finalize=True)

    action_button = window["-BUTTON_ACTION-"]
    status = window["-TEXT_STATUS-"]
    update_status = lambda mes: status.update(value=mes)

    figCanvas = FlushableFigCanvas(window["-CANVAS-"].TKCanvas)  # type: ignore

    state = STATE_INIT
    action_button.update(text="load")
    profile: Saxs2dProfile = None  # type: ignore
    while True:
        event, values = window.read(timeout=100)  # type: ignore
        if event != sg.TIMEOUT_KEY:
            logger.debug(f"state:{state} event: {event}")

        if event == "-BUTTON_EXIT-" or event == sg.WIN_CLOSED:
            break

        if event == "-BUTTON_SAVE-":
            savefile = TMPDIR + "test.png"
            profile.save(savefile, overwrite=True, showCenter=True)
            update_status(f"saved to `{savefile}`")

        if state == STATE_INIT and event == "-BUTTON_ACTION-":
            filepath = values["-INPUT_FILEPATH-"]
            try:
                profile = Saxs2dProfile.load_tiff(filepath)
            except FileNotFoundError:
                window["-TEXT_STATUS-"].update(value="file not found")
                update_status("file not found")
                continue
            except ValueError:
                window["-TEXT_STATUS-"].update(value="invalid file type")
                update_status("invalid file type")
                continue

            figCanvas.heatmap(profile.values(showMaskAsNan=False))
            update_status(f"`{filepath}` successfully loaded")
            state = STATE_WAIT_AUTO_MASK
            action_button.update(text="auto mask")
            continue

        if state == STATE_WAIT_AUTO_MASK and event == "-BUTTON_ACTION-":
            profile.auto_mask_invalid()
            figCanvas.heatmap(profile.values())
            update_status("auto mask done")
            state = STATE_WAIT_DETECT_CENTER
            action_button.update(text="detect center")
            continue
        if state == STATE_WAIT_DETECT_CENTER and event == "-BUTTON_ACTION-":
            profile.detect_center()
            if profile.center is None:
                update_status("center not detected.\twaiting for manual selection")
                state = STATE_WAIT_SELECT_CENTER
                continue
            else:
                figCanvas.heatmap(profile.values(showCenterAsNan=True))
                update_status("center detected")
                action_button.update(text="integrate")
                state = STATE_WAIT_INTEGRATE
                continue

        if state == STATE_WAIT_SELECT_CENTER and event == "-BUTTON_ACTION-":
            update_status("center selection called but not implemented yet")
            continue

        if state == STATE_WAIT_INTEGRATE and event == "-BUTTON_ACTION-":
            y, bins = profile.integrate(dr=5.0)
            x = (bins[:-1] + bins[1:]) / 2
            y = np.log(y)
            logger.debug(f"integrated: {x.shape}, {y.shape}")
            figCanvas.plot(x, y)
            update_status("integrated")
            continue

    window.close()


if __name__ == "__main__":
    main()
