from cProfile import label
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from cycler import Cycler, cycler

from anndata import AnnData
import numpy as np
import scvelo as scv
import matplotlib.pyplot as plt
from matplotlib import patheffects, rcParams
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

cm_data = [
    # [0.2422, 0.1504, 0.6603],
    # [0.2444, 0.1534, 0.6728],
    # [0.2464, 0.1569, 0.6847],
    # [0.2484, 0.1607, 0.6961],
    # [0.2503, 0.1648, 0.7071],
    # [0.2522, 0.1689, 0.7179],
    # [0.254, 0.1732, 0.7286],
    # [0.2558, 0.1773, 0.7393],
    # [0.2576, 0.1814, 0.7501],
    # [0.2594, 0.1854, 0.761],
    # [0.2611, 0.1893, 0.7719],
    # [0.2628, 0.1932, 0.7828],
    # [0.2645, 0.1972, 0.7937],
    # [0.2661, 0.2011, 0.8043],
    # [0.2676, 0.2052, 0.8148],
    # [0.2691, 0.2094, 0.8249],
    # [0.2704, 0.2138, 0.8346],
    # [0.2717, 0.2184, 0.8439],
    # [0.2729, 0.2231, 0.8528],
    # [0.274, 0.228, 0.8612],
    # [0.2749, 0.233, 0.8692],
    # [0.2758, 0.2382, 0.8767],
    # [0.2766, 0.2435, 0.884],
    # [0.2774, 0.2489, 0.8908],
    # [0.2781, 0.2543, 0.8973],
    # [0.2788, 0.2598, 0.9035],
    # [0.2794, 0.2653, 0.9094],
    # [0.2798, 0.2708, 0.915],
    # [0.2802, 0.2764, 0.9204],
    # [0.2806, 0.2819, 0.9255],
    # [0.2809, 0.2875, 0.9305],
    # [0.2811, 0.293, 0.9352],
    # [0.2813, 0.2985, 0.9397],
    # [0.2814, 0.304, 0.9441],
    # [0.2814, 0.3095, 0.9483],
    # [0.2813, 0.315, 0.9524],
    # [0.2811, 0.3204, 0.9563],
    # [0.2809, 0.3259, 0.96],
    # [0.2807, 0.3313, 0.9636],
    [0.2803, 0.3367, 0.967],
    [0.2798, 0.3421, 0.9702],
    [0.2791, 0.3475, 0.9733],
    [0.2784, 0.3529, 0.9763],
    [0.2776, 0.3583, 0.9791],
    [0.2766, 0.3638, 0.9817],
    [0.2754, 0.3693, 0.984],
    [0.2741, 0.3748, 0.9862],
    [0.2726, 0.3804, 0.9881],
    [0.271, 0.386, 0.9898],
    [0.2691, 0.3916, 0.9912],
    [0.267, 0.3973, 0.9924],
    [0.2647, 0.403, 0.9935],
    [0.2621, 0.4088, 0.9946],
    [0.2591, 0.4145, 0.9955],
    [0.2556, 0.4203, 0.9965],
    [0.2517, 0.4261, 0.9974],
    [0.2473, 0.4319, 0.9983],
    [0.2424, 0.4378, 0.9991],
    [0.2369, 0.4437, 0.9996],
    [0.2311, 0.4497, 0.9995],
    [0.225, 0.4559, 0.9985],
    [0.2189, 0.462, 0.9968],
    [0.2128, 0.4682, 0.9948],
    [0.2066, 0.4743, 0.9926],
    [0.2006, 0.4803, 0.9906],
    [0.195, 0.4861, 0.9887],
    [0.1903, 0.4919, 0.9867],
    [0.1869, 0.4975, 0.9844],
    [0.1847, 0.503, 0.9819],
    [0.1831, 0.5084, 0.9793],
    [0.1818, 0.5138, 0.9766],
    [0.1806, 0.5191, 0.9738],
    [0.1795, 0.5244, 0.9709],
    [0.1785, 0.5296, 0.9677],
    [0.1778, 0.5349, 0.9641],
    [0.1773, 0.5401, 0.9602],
    [0.1768, 0.5452, 0.956],
    [0.1764, 0.5504, 0.9516],
    [0.1755, 0.5554, 0.9473],
    [0.174, 0.5605, 0.9432],
    [0.1716, 0.5655, 0.9393],
    [0.1686, 0.5705, 0.9357],
    [0.1649, 0.5755, 0.9323],
    [0.161, 0.5805, 0.9289],
    [0.1573, 0.5854, 0.9254],
    [0.154, 0.5902, 0.9218],
    [0.1513, 0.595, 0.9182],
    [0.1492, 0.5997, 0.9147],
    [0.1475, 0.6043, 0.9113],
    [0.1461, 0.6089, 0.908],
    [0.1446, 0.6135, 0.905],
    [0.1429, 0.618, 0.9022],
    [0.1408, 0.6226, 0.8998],
    [0.1383, 0.6272, 0.8975],
    [0.1354, 0.6317, 0.8953],
    [0.1321, 0.6363, 0.8932],
    [0.1288, 0.6408, 0.891],
    [0.1253, 0.6453, 0.8887],
    [0.1219, 0.6497, 0.8862],
    [0.1185, 0.6541, 0.8834],
    [0.1152, 0.6584, 0.8804],
    [0.1119, 0.6627, 0.877],
    [0.1085, 0.6669, 0.8734],
    [0.1048, 0.671, 0.8695],
    [0.1009, 0.675, 0.8653],
    [0.0964, 0.6789, 0.8609],
    [0.0914, 0.6828, 0.8562],
    [0.0855, 0.6865, 0.8513],
    [0.0789, 0.6902, 0.8462],
    [0.0713, 0.6938, 0.8409],
    [0.0628, 0.6972, 0.8355],
    [0.0535, 0.7006, 0.8299],
    [0.0433, 0.7039, 0.8242],
    [0.0328, 0.7071, 0.8183],
    [0.0234, 0.7103, 0.8124],
    [0.0155, 0.7133, 0.8064],
    [0.0091, 0.7163, 0.8003],
    [0.0046, 0.7192, 0.7941],
    [0.0019, 0.722, 0.7878],
    [0.0009, 0.7248, 0.7815],
    [0.0018, 0.7275, 0.7752],
    [0.0046, 0.7301, 0.7688],
    [0.0094, 0.7327, 0.7623],
    [0.0162, 0.7352, 0.7558],
    [0.0253, 0.7376, 0.7492],
    [0.0369, 0.74, 0.7426],
    [0.0504, 0.7423, 0.7359],
    [0.0638, 0.7446, 0.7292],
    [0.077, 0.7468, 0.7224],
    [0.0899, 0.7489, 0.7156],
    [0.1023, 0.751, 0.7088],
    [0.1141, 0.7531, 0.7019],
    [0.1252, 0.7552, 0.695],
    [0.1354, 0.7572, 0.6881],
    [0.1448, 0.7593, 0.6812],
    [0.1532, 0.7614, 0.6741],
    [0.1609, 0.7635, 0.6671],
    [0.1678, 0.7656, 0.6599],
    [0.1741, 0.7678, 0.6527],
    [0.1799, 0.7699, 0.6454],
    [0.1853, 0.7721, 0.6379],
    [0.1905, 0.7743, 0.6303],
    [0.1954, 0.7765, 0.6225],
    [0.2003, 0.7787, 0.6146],
    [0.2061, 0.7808, 0.6065],
    [0.2118, 0.7828, 0.5983],
    [0.2178, 0.7849, 0.5899],
    [0.2244, 0.7869, 0.5813],
    [0.2318, 0.7887, 0.5725],
    [0.2401, 0.7905, 0.5636],
    [0.2491, 0.7922, 0.5546],
    [0.2589, 0.7937, 0.5454],
    [0.2695, 0.7951, 0.536],
    [0.2809, 0.7964, 0.5266],
    [0.2929, 0.7975, 0.517],
    [0.3052, 0.7985, 0.5074],
    [0.3176, 0.7994, 0.4975],
    [0.3301, 0.8002, 0.4876],
    [0.3424, 0.8009, 0.4774],
    [0.3548, 0.8016, 0.4669],
    [0.3671, 0.8021, 0.4563],
    [0.3795, 0.8026, 0.4454],
    [0.3921, 0.8029, 0.4344],
    [0.405, 0.8031, 0.4233],
    [0.4184, 0.803, 0.4122],
    [0.4322, 0.8028, 0.4013],
    [0.4463, 0.8024, 0.3904],
    [0.4608, 0.8018, 0.3797],
    [0.4753, 0.8011, 0.3691],
    [0.4899, 0.8002, 0.3586],
    [0.5044, 0.7993, 0.348],
    [0.5187, 0.7982, 0.3374],
    [0.5329, 0.797, 0.3267],
    [0.547, 0.7957, 0.3159],
    [0.5609, 0.7943, 0.305],
    [0.5748, 0.7929, 0.2941],
    [0.5886, 0.7913, 0.2833],
    [0.6024, 0.7896, 0.2726],
    [0.6161, 0.7878, 0.2622],
    [0.6297, 0.7859, 0.2521],
    [0.6433, 0.7839, 0.2423],
    [0.6567, 0.7818, 0.2329],
    [0.6701, 0.7796, 0.2239],
    [0.6833, 0.7773, 0.2155],
    [0.6963, 0.775, 0.2075],
    [0.7091, 0.7727, 0.1998],
    [0.7218, 0.7703, 0.1924],
    [0.7344, 0.7679, 0.1852],
    [0.7468, 0.7654, 0.1782],
    [0.759, 0.7629, 0.1717],
    [0.771, 0.7604, 0.1658],
    [0.7829, 0.7579, 0.1608],
    [0.7945, 0.7554, 0.157],
    [0.806, 0.7529, 0.1546],
    [0.8172, 0.7505, 0.1535],
    [0.8281, 0.7481, 0.1536],
    [0.8389, 0.7457, 0.1546],
    [0.8495, 0.7435, 0.1564],
    [0.86, 0.7413, 0.1587],
    [0.8703, 0.7392, 0.1615],
    [0.8804, 0.7372, 0.165],
    [0.8903, 0.7353, 0.1695],
    [0.9, 0.7336, 0.1749],
    [0.9093, 0.7321, 0.1815],
    [0.9184, 0.7308, 0.189],
    [0.9272, 0.7298, 0.1973],
    [0.9357, 0.729, 0.2061],
    [0.944, 0.7285, 0.2151],
    [0.9523, 0.7284, 0.2237],
    [0.9606, 0.7285, 0.2312],
    [0.9689, 0.7292, 0.2373],
    [0.977, 0.7304, 0.2418],
    [0.9842, 0.733, 0.2446],
    [0.99, 0.7365, 0.2429],
    [0.9946, 0.7407, 0.2394],
    [0.9966, 0.7458, 0.2351],
    [0.9971, 0.7513, 0.2309],
    [0.9972, 0.7569, 0.2267],
    [0.9971, 0.7626, 0.2224],
    [0.9969, 0.7683, 0.2181],
    [0.9966, 0.774, 0.2138],
    [0.9962, 0.7798, 0.2095],
    [0.9957, 0.7856, 0.2053],
    [0.9949, 0.7915, 0.2012],
    [0.9938, 0.7974, 0.1974],
    [0.9923, 0.8034, 0.1939],
    [0.9906, 0.8095, 0.1906],
    [0.9885, 0.8156, 0.1875],
    [0.9861, 0.8218, 0.1846],
    [0.9835, 0.828, 0.1817],
    [0.9807, 0.8342, 0.1787],
    [0.9778, 0.8404, 0.1757],
    [0.9748, 0.8467, 0.1726],
    [0.972, 0.8529, 0.1695],
    [0.9694, 0.8591, 0.1665],
    [0.9671, 0.8654, 0.1636],
    [0.9651, 0.8716, 0.1608],
    [0.9634, 0.8778, 0.1582],
    [0.9619, 0.884, 0.1557],
    [0.9608, 0.8902, 0.1532],
    [0.9601, 0.8963, 0.1507],
    [0.9596, 0.9023, 0.148],
    [0.9595, 0.9084, 0.145],
    [0.9597, 0.9143, 0.1418],
    [0.9601, 0.9203, 0.1382],
    [0.9608, 0.9262, 0.1344],
    [0.9618, 0.932, 0.1304],
    [0.9629, 0.9379, 0.1261],
    [0.9642, 0.9437, 0.1216],
    [0.9657, 0.9494, 0.1168],
    [0.9674, 0.9552, 0.1116],
    [0.9692, 0.9609, 0.1061],
    [0.9711, 0.9667, 0.1001],
    [0.973, 0.9724, 0.0938],
    [0.9749, 0.9782, 0.0872],
    [0.9769, 0.9839, 0.0805],
]
parula_map = LinearSegmentedColormap.from_list("parula", cm_data)
parula_map_r = LinearSegmentedColormap.from_list("parula_r", cm_data[::-1])


def default_palette(palette=None):
    if palette is None:
        return rcParams["axes.prop_cycle"]
    elif not isinstance(palette, Cycler):
        return cycler(color=palette)
    else:
        return palette


def statplot(
    arrays,
    style="box",
    showfliers=False,
    alpha=0.5,
    colors=None,
    linewidth=1,
    labels=None,
    xlabel=None,
    ylabel=None,
    xlim=None,
    ylim=None,
    xscale=None,
    yscale=None,
    xticks=None,
    yticks=None,
    title=None,
    fontsize=None,
    figsize=None,
    ax=None,
    dpi=None,
    show=True,
    **kwargs,
):
    """
    Plot a histogram.

    Arguments
    ---------
    arrays: : `list` or `array`.
        List of arrays to be plotted.
    style: `str`, choice of `box` or `violin` (default: `box`)
        Style of the plot.
    showfliers: `bool` (default: `False`)
        Show outliers. Only relevant for `box` style.
    alpha: `list`, `np.ndarray` or `None` (default: `None`)
        Alpha of the colors. Must be same length as colors.
    colors: : `list` or `array` (default `['royalblue', 'white', 'forestgreen']`)
        List of colors, either as names or rgb values.
    linewidth: `float` (default: `1`)
        Line width.
    labels : str or None (default: `None`)
            String, or sequence of strings to label clusters.
    xlabel: `str` (default: `None`)
        Label of x-axis.
    ylabel: `str` (default: `None`)
        Label of y-axis.
    xlim: tuple, e.g. [0,1] or `None` (default: `None`)
        Restrict x-limits of the axis.
    ylim: tuple, e.g. [0,1] or `None` (default: `None`)
        Restrict y-limits of the axis.
    cutoff: tuple, e.g. [0,1] or `float` or `None` (default: `None`)
        Bins will be cut off below and above the cutoff values.
    xscale: `log` or `None` (default: `None`)
        Scale of the x-axis.
    yscale: `log` or `None` (default: `None`)
        Scale of the y-axis.
    xticks: `list` or `None` (default: `None`)
        List of x-ticks.
    yticks: `list` or `None` (default: `None`)
        List of y-ticks.
    fontsize: `float` or `dict` (default: `None`)
        Label font size.
    figsize: tuple (default: `(7,5)`)
        Figure size.
    ax: `matplotlib.Axes`, optional (default: `None`)
        A matplotlib axes object. Only works if plotting a single component.
    dpi: `int` (default: 80)
        Figure dpi.
    show: `bool`, optional (default: `None`)
        Show the plot, do not return axis.

    Returns
    -------
    If `show==False` a `matplotlib.Axis`
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    palette = default_palette(None).by_key()["color"][::-1]
    colors = palette if colors is None or len(colors) < len(arrays) else colors

    if isinstance(labels, str):
        labels = [labels]

    if style == "box":
        bplot = ax.boxplot(
            arrays,
            showfliers=showfliers,
            patch_artist=True,
        )
        for i, patch in enumerate(bplot["boxes"]):
            patch.set_facecolor(colors[i])
            patch.set_alpha(alpha)
            patch.set_linewidth(linewidth)
            # patch.set_edgecolor(colors[i])
            # patch.set_linestyle("solid")
            # patch.set_capstyle("butt")
            # patch.set_joinstyle("miter")
            # patch.set_hatch(None)
            # patch.set_fill(True)
            # patch.set_zorder(1)
            # patch.set_path_effects(patheffects.Stroke(linewidth=1, foreground="black"))
    elif style == "violin":
        vplot = ax.violinplot(arrays)
        for i, patch in enumerate(vplot["bodies"]):
            patch.set_facecolor(colors[i])
            patch.set_alpha(alpha)
            patch.set_linewidth(linewidth)
    else:
        raise ValueError(f"Unknown style: {style}")

    if xlabel is None:
        xlabel = ""
    ax.set_xlabel(
        xlabel,
        fontsize=fontsize.get("xlabel", None)
        if isinstance(fontsize, dict)
        else fontsize,
    )
    if ylabel is None:
        ylabel = ""
    ax.set_ylabel(
        ylabel,
        fontsize=fontsize.get("ylabel", None)
        if isinstance(fontsize, dict)
        else fontsize,
    )

    if xscale is not None:
        ax.set_xscale(xscale)
    if yscale is not None:
        ax.set_yscale(yscale)

    if xticks is not None:
        ax.set_xticks(
            xticks,
            fontsize=fontsize.get("xticks", None)
            if isinstance(fontsize, dict)
            else fontsize,
        )
    if yticks is not None:
        ax.set_yticks(
            yticks,
            fontsize=fontsize.get("yticks", None)
            if isinstance(fontsize, dict)
            else fontsize,
        )

    if labels is not None:
        if len(labels) == len(arrays) and isinstance(arrays, list):
            ax.set_xticks(
                [y + 1 for y in range(len(arrays))],
                labels=labels,
                fontsize=fontsize.get("xsticks", None)
                if isinstance(fontsize, dict)
                else fontsize,
            )
        else:
            raise ValueError("labels must be same length as arrays")

    if title is not None:
        ax.set_title(
            title,
            fontsize=fontsize.get("title", None)
            if isinstance(fontsize, dict)
            else fontsize,
        )

    if rcParams["savefig.transparent"]:
        ax.patch.set_alpha(0)
    # ax.yaxis.grid(True)

    if show:
        plt.show()
    return ax


def compare_plot(
    *adatas: List[AnnData],
    labels: List[str] = ["DeepVelo", "Dynamical"],
    data: List[Any] = None,
    vkey: str = "velocity",
    method: str = "cosine",
    ylabel: str = "Overall consistency",
    add_title: bool = False,
    right_top_axis: bool = False,
) -> Tuple[plt.Axes]:
    """
    Plot and compare arrays of values from multiple AnnData objects.

    Args:
        adatas (List[AnnData]): List of AnnData objects.
        labels (List[str], optional): List of labels for each AnnData object.
            Defaults to ["DeepVelo", "Dynamical"].
        data (List[Any], optional): List of arrays of values to compare. If data
            is provided, vkey and method are ignored. Defaults to None.
        vkey (str, optional): Velocity key. Defaults to "velocity".
        method (str, optional): Velocity consistency method. Defaults to "cosine".
        ylabel (str, optional): Y-axis label. Defaults to "Overall consistency".
        add_title (bool, optional): Add title to plot. Defaults to False.
        right_top_axis (bool, optional): Show right top axis. Defaults to False.

    Returns:
        Tuple[plt.Axes]: Tuple of matplotlib axes.
    """
    assert len(adatas) == len(labels), "Must provide same number of labels as adatas"

    if data is None:
        data = [adata.obs[vkey + "_confidence_" + method].dropna() for adata in adatas]
        xlim_ = [0, 1]
        for adata, label in zip(adatas, labels):
            print(
                f"{label} mean score: {adata.obs[vkey + '_confidence_' + method].mean():.4f},"
                f" std: {adata.obs[vkey + '_confidence_' + method].std():.4f}"
            )
    else:
        xlim_ = [data[0].min(), data[0].max()]

    assert len(data) == len(labels)

    ax = scv.pl.hist(
        data,
        labels=labels,
        kde=True,
        normed=True,
        bins=200,
        xlim=xlim_,
        fontsize=18,
        legend_fontsize=16,
        show=False,
    )

    if not right_top_axis:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    ax_hist = ax

    ax = statplot(
        data,
        style="box",
        labels=labels,
        title=f"{vkey}_confidence_{method}" if add_title else None,
        ylabel=ylabel,
        fontsize={"xsticks": 12, "ylabel": 12, "title": 18},
        show=False,
    )

    if not right_top_axis:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    return ax_hist, ax


def dist_plot(
    *data: Any,
    bins: int = 10,
    labels: Optional[List[str]] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    fontsize: Optional[Union[int, Dict[str, int]]] = None,
    figsize: Optional[Tuple[int, int]] = None,
    dpi: Optional[int] = 150,
    show: Optional[bool] = True,
    save: Optional[str] = None,
) -> Union[plt.Axes, List[plt.Axes]]:
    """
    Plot a distribution of data in a row of subplots.
    Parameters
    ----------
    *data: `list` of `np.ndarray`
        Data to plot.
    bins: `int`, optional (default: `10`)
        Number of bins for histogram.
    labels: `list` of `str`, optional (default: `None`)
        Labels for each data.
    title: `str`, optional (default: `None`)
        Title of the plot.
    xlabel: `str`, optional (default: `None`)
        Label for the x-axis.
    ylabel: `str`, optional (default: `None`)
        Label for the y-axis.
    fontsize: `int` or `dict`, optional (default: `None`)
        Fontsize for the plot.
    figsize: `tuple`, optional (default: `(7, 5)`)
        Figure size.
    dpi: `int`, optional (default: `80`)
        Figure dpi.
    show: `bool`, optional (default: `True`)
        Show the plot, do not return axis.
    save: `str`, optional (default: `None`)
        Save the plot to file.
    Returns
    -------
    `matplotlib.axes.Axes` or `list` of `matplotlib.axes.Axes`
    """
    if fontsize is None:
        fontsize = {"xsticks": 12, "ylabel": 12, "title": 18}
    if isinstance(fontsize, int):
        fontsize = {"xsticks": fontsize, "ylabel": fontsize, "title": fontsize}
    if labels is None:
        labels = [str(i) for i in range(len(data))]
    assert len(data) == len(labels), "Must provide same number of labels as data"
    if figsize is None:
        figsize = (7 * len(data), 5)

    cols = len(data)
    rows = 1
    fig, axs = plt.subplots(
        nrows=rows, ncols=cols, figsize=figsize, dpi=dpi, sharey=True
    )
    axs = axs.flatten()

    if title is not None:
        fig.subplots_adjust(top=0.85)
        # set the suptitle postion to the top center
        fig.suptitle(title, fontsize=fontsize.get("title", None), y=0.9)

    for i, (d, label, ax) in enumerate(zip(data, labels, axs)):
        ax.hist(d, bins=bins, label=label, alpha=0.5)
        ax.set_title(label, fontsize=fontsize.get("title", None))
        ax.set_xlabel(xlabel, fontsize=fontsize.get("xlabel", None))
        ax.set_ylabel(ylabel, fontsize=fontsize.get("ylabel", None))
        ax.tick_params(labelsize=fontsize.get("xticks", None))
        ax.legend(fontsize=fontsize.get("legend", None))

    if show:
        plt.show()
    if save is not None:
        fig.savefig(save)
    return axs


def gene_scatter(
    adata: AnnData,
    coords: "np.ndarray",
    var_keys: Union[str, List[str]] = ["gene_continuity", "gene_corr"],
    size=30,
    nrows: int = 1,
    titles: Optional[Union[str, List[str]]] = None,
    legend_titles: Optional[Union[str, List[str]]] = None,
    legend_title_size: Union[int, str] = "small",
    legend_size: Union[int, str] = "small",
):
    """gene-wise scatter plot"""

    if isinstance(var_keys, str):
        var_keys = [var_keys]

    if titles is None:
        titles = [f"plot of {vkey}" for vkey in var_keys]

    if legend_titles is None:
        legend_titles = [f"{vkey}" for vkey in var_keys]

    num_var = len(var_keys)
    ncols = num_var // nrows

    # plot on coords
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 8, nrows * 6))
    axes = axes.flatten() if num_var > 1 else [axes]

    for i, vkey in enumerate(var_keys):
        scatter_plot = axes[i].scatter(
            coords[:, 0],
            coords[:, 1],
            c=adata.var[vkey],
            s=(adata.layers["Ms"].mean(axis=0) * size).tolist(),
            alpha=0.5,
        )
        # color bar lim to 2%, 98% of continuity
        # continuity_lim = np.percentile(adata.var["gene_continuity"], [2, 98])
        # scatter_plot.set_clim(continuity_lim)
        axes[i].set_title(titles[i])
        axes[i].add_artist(
            axes[i].legend(
                *scatter_plot.legend_elements(num=7),
                loc="lower left",
                title=legend_titles[i],
                title_fontsize=legend_title_size,
                fontsize=legend_size,
            )
        )
        axes[i].legend(
            *scatter_plot.legend_elements(
                prop="sizes", alpha=0.6, func=lambda s: s / size, num=7
            ),
            loc="best",
            title="expression",
            title_fontsize=legend_title_size,
            fontsize=legend_size,
        )


def draw_var_dist(
    adata,
    var_name,
    quantile=0.05,
    find_value=None,
    save_dir=None,
):
    if save_dir is not None:
        save_dir = Path(save_dir)
    fig = plt.figure(figsize=(6, 4))
    sns.histplot(adata.var[var_name], bins=50, kde=True, stat="density")
    # show the vertical line at the 5%, text the value
    plt.axvline(adata.var[var_name].quantile(quantile), color="red")
    plt.text(
        adata.var[var_name].quantile(quantile),
        1.0,
        f"{adata.var[var_name].quantile(quantile):.2f} at {quantile*100}%",
        color="red",
    )

    # find the quantile of the value and show the vertical line
    if find_value is not None:
        find_quantile = (adata.var[var_name] < find_value).sum() / adata.var.shape[0]
        plt.axvline(find_value, color="green", linestyle="--")
        plt.text(
            find_value,
            0.7,
            f"{find_value:d} at {find_quantile*100}%"
            if find_value == 0
            else f"{find_value:.2f} at {find_quantile*100:.2f}%",
            color="green",
        )

    plt.title(f"{var_name} distribution")
    if save_dir is not None:
        plt.savefig(save_dir / f"{var_name}_distribution.png")
        print(f"save {var_name} distribution to {save_dir}")
