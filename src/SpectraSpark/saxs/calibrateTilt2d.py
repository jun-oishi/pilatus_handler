import numpy as np
import cv2
import sys
from SpectraSpark.util.basic_calculation import q2theta, convert
from SpectraSpark.constants import Q_CeO2, Q_NaCl, Q_Si, Q_AgBeh, EIGER_PX_SIZE, PILATUS_PX_SIZE
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import datetime
import json

def drawPeaks(im, thetas, tilt, camera_length, center,
              color=255, thickness=1):
    """thetasに対応する散乱角のピークを描画する
    Parameters
    ----------
    im : ベースとなる画像
    thetas : 散乱角の配列[rad]
    tilt : tilt角[rad]
    camera_length : カメラ長-カメラ平面とサンプルの距離[px]
    center : ビームセンターの座標(x, y)[px]
    """
    s, t, u = np.tan(tilt), np.tan(thetas), np.cos(tilt)
    l = camera_length / u
    d = 1 - s**2 * t**2
    e = np.sqrt(1 + s**2 - s**2 * t**2)
    a = (e * l * t) / (d * u)
    b = (e * l * t) / np.sqrt(d)
    x0 = center[0] + (l*s*t**2) / (d*u)
    y0 = center[1]

    for _a, _b, _x0 in zip(a, b, x0):
        im = cv2.ellipse(im, ((_x0, y0), (_a, _b), 0), color, thickness) # type: ignore
    return im

def __add_slider(fig, top, bottom, left, right, label, valmin, valmax, valinit):
    """スライダーを追加する"""
    ax = fig.add_axes((left, bottom, right-left, top-bottom))
    slider = Slider(ax, label, valmin, valmax, valinit=valinit)
    return slider

def __add_button(fig, top, bottom, left, right, label):
    """ボタンを追加する"""
    ax = fig.add_axes((left, bottom, right-left, top-bottom))
    button = Button(ax, label)
    return button

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # パラメータ
    px_size = EIGER_PX_SIZE
    ini_tilt = 20.0
    ini_l = 200
    ini_x0 = -800

    if len(sys.argv) < 4:
        print("Usage: python calibrate.py [image path] [std type] [wave length/nm]")
        sys.exit(1)

    src = sys.argv[1]

    wave_length = float(sys.argv[3])

    if sys.argv[2] == "CeO2":
        Q = Q_CeO2
    elif sys.argv[2] == "NaCl":
        Q = Q_NaCl
    elif sys.argv[2] == "Si":
        Q = Q_Si
    elif sys.argv[2] == "AgBeh":
        Q = Q_AgBeh
    else:
        print("standard type must be CeO2, NaCl, Si, or AgBeh")
        sys.exit(1)
    arr_theta = np.array([q2theta(q, wave_length, unit="rad") for q in Q])

    raw = cv2.imread(src, cv2.IMREAD_UNCHANGED)
    raw = np.log(raw - raw.min() + 1)
    print(f"raw size: {raw.shape}")
    height, width = raw.shape
    margin = 0.2
    y_offset = int(height*margin)
    canvas = np.zeros(
        (height+2*int(height*margin), width),
        dtype=np.uint8
    )
    canvas[y_offset:y_offset+height, :] = convert(raw, dtype=np.uint8)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_axes((0.1, 0.2, 0.8, 0.7))

    left, right = 0.1, 0.9
    l_slider = __add_slider(
        fig, 0.16, 0.13, left, right, "camera length[mm]",
        valmin=0, valmax=500, valinit=ini_l
    )
    tilt_slider = __add_slider(
        fig, 0.13, 0.10, left, right, "tilt[deg]",
        valmin=0, valmax=60, valinit=ini_tilt
    )
    x0_slider = __add_slider(
        fig, 0.10, 0.07, left, right, "x0[px]",
        valmin=-10000, valmax=-500, valinit=ini_x0
    )
    y0_slider = __add_slider(
        fig, 0.07, 0.04, left, right, "y0[px]",
        valmin=0, valmax=height, valinit=height*0.5
    )
    save_button = __add_button(
        fig, 0.04, 0.01, left, right, "save"
    )

    def update(val):
        l = l_slider.val / px_size
        tilt = np.deg2rad(tilt_slider.val)
        center = (x0_slider.val, y0_slider.val+y_offset)
        im = drawPeaks(canvas.copy(), arr_theta, tilt, l, center)
        ax.clear()
        ax.imshow(im, cmap="jet")
        return

    def save(val):
        now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        fig.savefig(f"calibration_{now}.png")
        result = {
            "source" : src,
            "wave_length[nm]" : wave_length,
            "pixel_size[mm]" : px_size,
            "camera_length[mm]" : l_slider.val,
            "tilt[deg]" : tilt_slider.val,
            "center_x[px]" : x0_slider.val,
            "center_y[px]" : y0_slider.val
        }
        with open(f"calibration_{now}.json", "w") as f:
            json.dump(result, f, indent=2)
        return

    l_slider.on_changed(update)
    tilt_slider.on_changed(update)
    x0_slider.on_changed(update)
    y0_slider.on_changed(update)
    save_button.on_clicked(save)

    update(None)
    fig.show()
    plt.show()