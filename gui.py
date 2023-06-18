from Saxs2dProfile import Saxs2dProfile
import PySimpleGUI as sg
import util

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib as mpl
import numpy as np

logger = util.getLogger(__name__, util.DEBUG)

TMPDIR = "tmp/"

WINDOW_SIZE = (500, 750)
CANVAS_SIZE = (500, 600)

STATE_INIT = "init"
STATE_WAIT_AUTO_MASK = "wait_auto_mask"
STATE_WAIT_DETECT_CENTER = "wait_detect_center"
STATE_WAIT_SELECT_CENTER = "wait_select_center"
STATE_WAIT_1D_CONVERT = "wait_1d_convert"


class HeatmapFigCanvas(FigureCanvasTkAgg):
    def __init__(self, canvas: sg.tkinter.Canvas):
        self.canvas = canvas
        self.fig: Figure = Figure()
        self.ax: Axes = self.fig.add_subplot(111)
        self.cmap: mpl.colormap.Colormap = mpl.colormaps.get_cmap("hot").copy()  # type: ignore
        self.cmap.set_bad("lime", alpha=1.0)
        super().__init__(self.fig, canvas)
        self.colorbar = None
        self.draw()
        self.get_tk_widget().pack(side="top", fill="both", expand=1)
        return

    def refresh_with(self, data: np.ndarray):
        self.ax.clear()
        im = self.ax.imshow(data, cmap=self.cmap)
        self.draw()


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
        [sg.Text("successfully initiated", key="-TEXT_STATUS-")],
        [sg.Canvas(size=CANVAS_SIZE, key="-CANVAS-")],
        [
            sg.Button("exit", key="-BUTTON_EXIT-"),
            sg.Button("save", key="-BUTTON_SAVE-"),
        ],
    ]

    window = sg.Window("test", layout, size=WINDOW_SIZE, finalize=True)

    canvas_elem: sg.Canvas = window["-CANVAS-"]  # type: ignore
    action_button = window["-BUTTON_ACTION-"]
    status = window["-TEXT_STATUS-"]
    update_status = lambda mes: status.update(value=mes)

    canvas: sg.tkinter.Canvas = canvas_elem.TKCanvas  # type: ignore
    fig_agg = HeatmapFigCanvas(canvas)

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

            fig_agg.refresh_with(profile.values(showMaskAsNan=False))
            update_status(f"`{filepath}` successfully loaded")
            state = STATE_WAIT_AUTO_MASK
            action_button.update(text="auto mask")
            continue

        if state == STATE_WAIT_AUTO_MASK and event == "-BUTTON_ACTION-":
            profile.auto_mask()
            fig_agg.refresh_with(profile.values())
            update_status("auto mask done")
            state = STATE_WAIT_DETECT_CENTER
            action_button.update(text="detect center")
            continue
        if state == STATE_WAIT_DETECT_CENTER and event == "-BUTTON_ACTION-":
            profile.detect_center()
            if profile.center() is None:
                update_status("center not detected.\twaiting for manual selection")
                state = STATE_WAIT_SELECT_CENTER
            else:
                fig_agg.refresh_with(profile.values(showCenterAsNan=True))
                update_status("center detected")
                state = STATE_WAIT_1D_CONVERT

        if state == STATE_WAIT_SELECT_CENTER and event == "-BUTTON_ACTION-":
            update_status("select center")
            state = STATE_WAIT_1D_CONVERT
            continue

    window.close()


if __name__ == "__main__":
    main()
