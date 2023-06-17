from Saxs2dProfile import Saxs2dProfile
import PySimpleGUI as sg
import util
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

logger = util.getLogger(__name__, util.DEBUG)

TMPDIR = "tmp/"

WINDOW_SIZE = (500, 750)
CANVAS_SIZE = (500, 600)


def draw_figure(canvas, figure, loc=(0, 0)) -> FigureCanvasTkAgg:
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
    return figure_canvas_agg


def load_tiff(filepath):
    try:
        profile = Saxs2dProfile.load_tiff(filepath)
    except Exception as e:
        raise e
    return profile


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
        [sg.Button("exit", key="-BUTTON_EXIT-")],
    ]

    window = sg.Window("test", layout, size=WINDOW_SIZE, finalize=True)

    canvas_elem: sg.Canvas = window["-CANVAS-"]  # type: ignore
    action_button = window["-BUTTON_ACTION-"]
    status = window["-TEXT_STATUS-"]
    update_status = lambda mes: status.update(value=mes)

    canvas: sg.tkinter.Canvas = canvas_elem.TKCanvas  # type: ignore
    fig = Figure()
    ax = fig.add_subplot(111)
    fig_agg = draw_figure(canvas, fig)

    state = "init"
    action_button.update(text="load")
    profile: Saxs2dProfile = None  # type: ignore
    while True:
        event, values = window.read(timeout=100)  # type: ignore
        if event != sg.TIMEOUT_KEY:
            logger.debug(f"state:{state} event: {event}")

        if event == "-BUTTON_EXIT-" or event == sg.WIN_CLOSED:
            break

        if state == "init" and event == "-BUTTON_ACTION-":
            filepath = values["-INPUT_FILEPATH-"]
            try:
                profile = load_tiff(filepath)
            except FileNotFoundError:
                window["-TEXT_STATUS-"].update(value="file not found")
                update_status("file not found")
                continue
            except ValueError:
                window["-TEXT_STATUS-"].update(value="invalid file type")
                update_status("invalid file type")
                continue

            update_status("`{filepath}` successfully loaded")
            state = "loaded"
            ax.cla()
            ax.imshow(profile.values(log=True))
            fig_agg.draw()
            action_button.update(text="auto mask")
            continue

        if state == "loaded" and event == "-BUTTON_ACTION-":
            profile.auto_mask()
            ax.cla()
            ax.imshow(profile.values(log=True))
            fig_agg.draw()
            action_button.update(text="mask")
            state = "masked"
            continue

    window.close()


if __name__ == "__main__":
    main()
