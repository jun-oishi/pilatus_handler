import numpy as np
import cv2
import os
from typing import Tuple
from numba import jit, float64, int32
from util import theta2q

@jit(nopython=True, cache=True)
def xy2theta(xy, tilt, camera_length, x0=0, y0=0):
    """カメラ平面上の座標を極座標[rad]に変換する

    Parameters
    ----------
    xy : 元座標の配列(2次元ベクトルか2行の配列)
    tilt : チルト角[rad]
    camera_length : カメラ長-カメラ平面とサンプルの距離[px]
    x0 : ビームセンターのx座標[px]
    y0 : ビームセンターのy座標[px]
    """
    assert len(xy) == 2, "xy must be 2D vector or 2-row array"
    l = camera_length / np.cos(tilt)
    x = xy[0] - x0 * np.cos(tilt)
    y = xy[1] - y0
    theta = np.arctan2(np.sqrt(x**2+y**2), l)
    # phi = np.arctan2(y, x)
    return theta

@jit(nopython=True, cache=True)
def integrate(img:np.ndarray, tilt_angle:float,
              camera_length:float, x0:float, y0:float,
              wave_length:float, px_size:float) -> Tuple[np.ndarray, np.ndarray]:
    """チルト角がある画像を一次元化する

    Parameters
    ----------
    img : 元画像
    tilt_angle : チルト角[deg], 0度がビームに垂直
    camera_length : カメラ長-カメラ平面とサンプルの距離[mm]
    x0 : ビームセンターのx座標[px]
    y0 : ビームセンターのy座標[px]
    wave_length: X線の波長[nm]
    px_size : 画素サイズ[mm]

    Returns
    -------
    q : q[nm^-1]
    i : 強度
    """
    tilt = np.deg2rad(tilt_angle)
    camera_length = camera_length / px_size
    height, width = img.shape

    xy = np.array([[0.0, width-1], [y0, y0]]) # 型を揃える
    theta_lim = xy2theta(xy, tilt, camera_length, x0, y0)
    q_lim = theta2q(theta_lim, wave_length, "rad")
    q_ret = np.linspace(q_lim[0], q_lim[1], width)
    dq = q_ret[1] - q_ret[0]
    q_bin = np.empty(len(q_ret)+1, np.float64)
    q_bin[0] = q_ret[0] - dq/2
    q_bin[1:] = q_ret + dq/2

    x = np.arange(width)
    i_ret = np.zeros_like(q_ret)
    n_ret = np.zeros_like(q_ret, np.int64)
    for y in range(height):
        xy = np.empty((2, width))
        xy[0,:] = x
        xy[1,:] = y
        theta = xy2theta(xy, tilt, camera_length, x0, y0)
        q = theta2q(theta, wave_length, "rad")
        i = img[y]
        idx = np.digitize(q, q_bin)
        for j in range(width):
            if idx[j] == 0 or idx[j] == len(q_bin):
                continue
            i_ret[idx[j]-1] += i[j]
            n_ret[idx[j]-1] += 1
    i_ret = i_ret / n_ret
    return q_ret, i_ret


if __name__ == "__main__":
    import sys, os, json

    if len(sys.argv) < 3:
        print("Usage: python integrateTilt2d.py [image path] [param file path]")
        sys.exit(1)
    args = sys.argv[1:]

    if "-O" in args or "--overwrite" in args:
        overwrite=True
        if "-O" in args:
            args.remove("-O")
        if "--overwrite" in args:
            args.remove("--overwrite")
    else:
        overwrite=False


    if not os.path.exists(args[1]):
        raise FileNotFoundError(f"{args[1]} is not found.")
    params = json.load(open(args[1]))
    pixel_size:float = params["pixel_size[mm]"]
    wave_length:float = params["wave_length[nm]"]
    camera_length:float = params["camera_length[mm]"]
    tilt_angle:float = params["tilt[deg]"]
    x0 = params["center_x[px]"]
    y0 = params["center_y[px]"]

    if not os.path.exists(args[0]):
        raise FileNotFoundError(f"{args[0]} is not found.")
    if not overwrite and os.path.exists(dst):
        raise FileExistsError(f"{dst} is already exists.")

    dst = os.path.splitext(args[0])[0] + ".csv"
    if os.path.isdir(args[0]):
        files = [f for f in os.listdir(args[0]) if f.endswith(".tif")]
        files = [os.path.join(args[0], f) for f in files]
        files.sort()

        width = cv2.imread(files[0], cv2.IMREAD_UNCHANGED).shape[1]
        q = np.empty(width)
        i = np.empty((len(files), width), np.float64)
        for _i, file in enumerate(files):
            im = cv2.imread(file, cv2.IMREAD_UNCHANGED)
            q, i[_i] = integrate(im, tilt_angle, camera_length, x0, y0,
                                wave_length, pixel_size)
        header = f"# {json.dumps(params)}\n" \
                 + "# q[nm^-1] " \
                 + " ".join([f"{os.path.basename(file)}" for file in files])
        np.savetxt(dst, np.hstack([q.reshape(-1, 1), i.T]),
                   header=header, comments="")
        print(f"Saved to {dst}")
    else:
        im = cv2.imread(args[0], cv2.IMREAD_UNCHANGED)
        q, i = integrate(im, tilt_angle, camera_length, x0, y0,
                        wave_length, pixel_size)
        header = f"# {json.dumps(params)}\n# q[nm^-1] i"
        np.savetxt(dst, np.array([q, i]).T, header=header, comments="")
        print(f"Saved to {dst}")