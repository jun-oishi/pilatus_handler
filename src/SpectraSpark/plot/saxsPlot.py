import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from typing import Iterable

from ..util import ArrayLike
from ..saxs import Saxs2d, Saxs1d, Saxs1dSeries

_EMPTY = np.array([])

_q2d = lambda q: 2 * np.pi / q
_d2q = lambda d: 2 * np.pi / d


def trSaxsHeatmap(
    fig: Figure,
    ax: Axes,
    saxsData: Saxs1dSeries,
    *,
    logscale: bool = True,
    x_lim: tuple[float, float] = (np.nan, np.nan),
    secondary_xaxis: bool = True,
    y_label: str = "file number",
    y_ticks: ArrayLike = _EMPTY,
    y_tick_labels: Iterable[str] = [],
    y_lim: tuple[float, float] = (0, np.inf),
    n_levels: int = 128,
    min_val: float = np.nan,
    max_val: float = np.nan,
    cmap: str | Colormap = "jet",
    show_colorbar: bool = True,
    extend: str = "min",
    cbar_fraction: float = 0.01,
    cbar_pad: float = 0.09,
) -> Axes:
    """時分割のSAXSデータ系列をヒートマップで表示する"""
    q = saxsData.q
    i = saxsData.i if not logscale else np.log(saxsData.i)
    y = np.arange(i.shape[0])

    if x_lim[0] > q[0]:
        i = i[:, q >= x_lim[0]]
        q = q[q >= x_lim[0]]
    if x_lim[1] < q[-1]:
        i = i[:, q <= x_lim[1]]
        q = q[q <= x_lim[1]]

    if y_lim[0] > y[0]:
        i = i[y >= y_lim[0]]
        y = y[y >= y_lim[0]]
    if y_lim[1] < y[-1]:
        i = i[y <= y_lim[1]]
        y = y[y <= y_lim[1]]

    min_val = np.nanmin(i) if np.isnan(min_val) else min_val
    max_val = np.nanmax(i) if np.isnan(max_val) else max_val
    levels = np.linspace(min_val, max_val, n_levels)

    cs = ax.contourf(q, y, i, levels=levels, cmap=cmap, extend=extend)

    if secondary_xaxis:
        top_ax = ax.secondary_xaxis("top", functions=(_q2d, _d2q))
        top_ax.set_xlabel("$d\;[\mathrm{nm}]$")

    if len(y_ticks) != 0:
        ax.set_yticks(y_ticks, y_tick_labels)

    if y_label != "":
        ax.set_ylabel(y_label)

    if show_colorbar:
        cbar = fig.colorbar(
            cs,
            ax=ax,
            orientation="horizontal",
            location="bottom",
            pad=cbar_pad,
            fraction=cbar_fraction,
            aspect=1 / cbar_fraction,
        )
        if logscale:
            cbar.set_label("$\ln[I(q)]\;[a.u.]$")
        else:
            cbar.set_label("$I(q)\;[a.u.]$")

    return ax


def saveHeatmap(saxs_series: Saxs1dSeries, dst: str, figsize=None, **kwargs):
    """時分割のSAXSデータ系列をヒートマップで表示して保存する"""
    fig, ax = plt.subplots(figsize=figsize)
    trSaxsHeatmap(fig, ax, saxs_series, **kwargs)
    fig.savefig(dst)
    plt.close(fig)
    return


def showQIimage(
    fig: Figure,
    ax: Axes,
    img: Saxs2d,
    *,
    logscale: bool = True,
    lim: float = np.nan,
    x_lim: tuple[float, float] = (np.nan, np.nan),
    y_lim: tuple[float, float] = (np.nan, np.nan),
    max_val: float = np.nan,
    min_val: float = np.nan,
    show_colorbar: bool = True,
    extend: str = "min",
    cbar_fraction: float = 0.01,
    cbar_pad: float = 0.12,
    secondary_axis: bool = True,
    sx_ticks: ArrayLike = _EMPTY,
    sx_xticks: ArrayLike = _EMPTY,
    sx_yticks: ArrayLike = _EMPTY,
    _cmap: str | Colormap = "hot",
    nan_color: str = "skyblue",
) -> Axes:
    """SAXS画像を表示する"""
    i = img.i if not logscale else np.log(img.i)
    px2q = img.px2q
    center_x, center_y = img.center

    x = (np.arange(i.shape[1]) - center_x) * px2q
    y = (np.arange(i.shape[0]) - center_y) * px2q

    if not np.isnan(lim):
        x_lim = (-lim, lim)
        y_lim = (-lim, lim)

    if x_lim[0] > x[0]:
        i = i[:, x >= x_lim[0]]
        x = x[x >= x_lim[0]]
    if x_lim[1] < x[-1]:
        i = i[:, x <= x_lim[1]]
        x = x[x <= x_lim[1]]
    if y_lim[0] > y[0]:
        i = i[:, y >= y_lim[0]]
        y = y[y >= y_lim[0]]
    if y_lim[1] < y[-1]:
        i = i[:, y <= y_lim[1]]
        y = y[y <= y_lim[1]]

    min_val = np.nanmin(i) if np.isnan(min_val) else min_val
    max_val = np.nanmax(i) if np.isnan(max_val) else max_val

    dq = px2q
    left, right = x[0] - dq / 2, x[-1] + dq / 2
    bottom, top = y[0] - dq / 2, y[-1] + dq / 2
    extent = (left, right, bottom, top)
    cmap: Colormap = plt.get_cmap(_cmap)
    cmap.set_bad(nan_color)
    im = ax.imshow(
        i,
        cmap=cmap,
        extent=extent,
        vmin=min_val,
        vmax=max_val,
        aspect="equal",
        origin="lower",
    )

    if show_colorbar:
        cbar = fig.colorbar(
            im,
            ax=ax,
            orientation="vertical",
            location="right",
            pad=cbar_pad,
            fraction=cbar_fraction,
            aspect=1 / cbar_fraction,
        )
        if logscale:
            cbar.set_label("$\ln[I(q)]\;[a.u.]$")
        else:
            cbar.set_label("$I(q)\;[a.u.]$")

    ax.set_xlabel("$q\;[\mathrm{nm}^{-1}]$")
    ax.set_ylabel("$q\;[\mathrm{nm}^{-1}]$")

    if secondary_axis:
        sx_x = ax.secondary_xaxis("top", functions=(_q2d, _d2q))
        sx_x.set_xlabel("$d\;[\mathrm{nm}]$")
        sx_y = ax.secondary_yaxis("right", functions=(_q2d, _d2q))
        sx_y.set_ylabel("$d\;[\mathrm{nm}]$")

        if len(sx_ticks) != 0:
            sx_xticks = sx_ticks
            sx_yticks = sx_ticks
        if len(sx_xticks) != 0:
            sx_x.set_xticks(sx_xticks)
        if len(sx_yticks) != 0:
            sx_y.set_yticks(sx_yticks)

    return ax
