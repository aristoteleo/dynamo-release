import colorcet
import matplotlib
from matplotlib import rcParams, cm, colors
from cycler import cycler
import matplotlib.pyplot as plt

# create cmap
zebrafish_colors = ['#4876ff', '#85C7F2', '#cd00cd', '#911eb4', '#000080', '#808080', '#008080', '#ffc125', '#262626',
                    '#3cb44b', '#ff4241', '#b77df9']
zebrafish_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("zebrafish", zebrafish_colors)

fire_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("fire", colorcet.fire)
darkblue_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "darkblue", colorcet.kbc
)
darkgreen_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "darkgreen", colorcet.kgy
)
darkred_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "darkred", colors=colorcet.linear_kry_5_95_c72[:192], N=256
)
darkpurple_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "darkpurple", colorcet.linear_bmw_5_95_c89
)
# add gkr theme for velocity
div_blue_black_red_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "div_blue_black_red", colorcet.diverging_gkr_60_10_c40
)
# add RdBu_r theme for velocity
div_blue_red_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "div_blue_red", colorcet.diverging_bwr_55_98_c37
)
# add glasbey_bw for cell annotation in white background
glasbey_white_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "glasbey_white", colorcet.glasbey_bw_minc_20
)
# add glasbey_bw_minc_20_maxl_70 theme for cell annotation in dark background
glasbey_dark_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "glasbey_dark", colorcet.glasbey_bw_minc_20_maxl_70
)

# register cmap
plt.register_cmap("zebrafish", zebrafish_cmap)
plt.register_cmap("fire", fire_cmap)
plt.register_cmap("darkblue", darkblue_cmap)
plt.register_cmap("darkgreen", darkgreen_cmap)
plt.register_cmap("darkred", darkred_cmap)
plt.register_cmap("darkpurple", darkpurple_cmap)
plt.register_cmap("div_blue_black_red", div_blue_black_red_cmap)
plt.register_cmap("div_blue_red", div_blue_red_cmap)
plt.register_cmap("glasbey_white", glasbey_white_cmap)
plt.register_cmap("glasbey_dark", glasbey_dark_cmap)


_themes = {
    "fire": {
        "cmap": "fire",
        "color_key_cmap": "rainbow",
        "background": "black",
        "edge_cmap": "fire",
    },
    "viridis": {
        "cmap": "viridis",
        "color_key_cmap": "Spectral",
        "background": "white",
        "edge_cmap": "gray",
    },
    "inferno": {
        "cmap": "inferno",
        "color_key_cmap": "Spectral",
        "background": "black",
        "edge_cmap": "gray",
    },
    "blue": {
        "cmap": "Blues",
        "color_key_cmap": "tab20",
        "background": "white",
        "edge_cmap": "gray_r",
    },
    "red": {
        "cmap": "Reds",
        "color_key_cmap": "tab20b",
        "background": "white",
        "edge_cmap": "gray_r",
    },
    "green": {
        "cmap": "Greens",
        "color_key_cmap": "tab20c",
        "background": "white",
        "edge_cmap": "gray_r",
    },
    "darkblue": {
        "cmap": "darkblue",
        "color_key_cmap": "rainbow",
        "background": "black",
        "edge_cmap": "darkred",
    },
    "darkred": {
        "cmap": "darkred",
        "color_key_cmap": "rainbow",
        "background": "black",
        "edge_cmap": "darkblue",
    },
    "darkgreen": {
        "cmap": "darkgreen",
        "color_key_cmap": "rainbow",
        "background": "black",
        "edge_cmap": "darkpurple",
    },
    "div_blue_black_red": {
        "cmap": "div_blue_black_red",
        "color_key_cmap": "div_blue_black_red",
        "background": "black",
        "edge_cmap": "gray_r",
    },
    "div_blue_red": {
        "cmap": "div_blue_red",
        "color_key_cmap": "div_blue_red",
        "background": "white",
        "edge_cmap": "gray_r",
    },
    "glasbey_dark": {
        "cmap": "glasbey_dark",
        "color_key_cmap": "glasbey_dark",
        "background": "black",
        "edge_cmap": "gray",
    },
    "glasbey_white_zebrafish": {
        "cmap": "zebrafish",
        "color_key_cmap": "zebrafish",
        "background": "white",
        "edge_cmap": "gray_r",
    },
    "glasbey_white": {
        "cmap": "glasbey_white",
        "color_key_cmap": "glasbey_white",
        "background": "white",
        "edge_cmap": "gray_r",
    },
}

# https://github.com/vega/vega/wiki/Scales#scale-range-literals
cyc_10 = list(map(colors.to_hex, cm.tab10.colors))
cyc_20 = list(map(colors.to_hex, cm.tab20c.colors))
zebrafish_256 = list(map(colors.to_hex, zebrafish_colors))

# ideally let us convert the following ggplot theme for Nature publisher group into matplotlib.rcParams
# nm_theme <- function() {
#   theme(strip.background = element_rect(colour = 'white', fill = 'white')) +
#     theme(panel.border = element_blank(), axis.line = element_line()) +
#     theme(panel.grid.minor.x = element_blank(), panel.grid.minor.y = element_blank()) +
#     theme(panel.grid.major.x = element_blank(), panel.grid.major.y = element_blank()) +
#     theme(panel.background = element_rect(fill='white')) +
#     #theme(text = element_text(size=6)) +
#     theme(axis.text.y=element_text(size=6)) +
#     theme(axis.text.x=element_text(size=6)) +
#     theme(axis.title.y=element_text(size=6)) +
#     theme(axis.title.x=element_text(size=6)) +
#     theme(panel.border = element_blank(), axis.line = element_line(size = .1), axis.ticks = element_line(size = .1)) +
#     theme(legend.position = "none") +
#     theme(strip.text.x = element_text(colour="black", size=6)) +
#     theme(strip.text.y = element_text(colour="black", size=6)) +
#     theme(legend.title = element_text(colour="black", size = 6)) +
#     theme(legend.text = element_text(colour="black", size = 6)) +
#     theme(plot.margin=unit(c(0,0,0,0), "lines"))
# }


def dyn_theme(background="white"):
    # https://github.com/matplotlib/matplotlib/blob/master/lib/matplotlib/mpl-data/stylelib/dark_background.mplstyle

    if background == "black":
        rcParams.update(
            {
                "lines.color": "w",
                "patch.edgecolor": "w",
                "text.color": "w",
                "axes.facecolor": background,
                "axes.edgecolor": "white",
                "axes.labelcolor": "w",
                "xtick.color": "w",
                "ytick.color": "w",
                "figure.facecolor": background,
                "figure.edgecolor": background,
                "savefig.facecolor": background,
                "savefig.edgecolor": background,
                "grid.color": "w",
                "axes.grid": False,
            }
        )
    else:
        rcParams.update(
            {
                "lines.color": "k",
                "patch.edgecolor": "k",
                "text.color": "k",
                "axes.facecolor": background,
                "axes.edgecolor": "black",
                "axes.labelcolor": "k",
                "xtick.color": "k",
                "ytick.color": "k",
                "figure.facecolor": background,
                "figure.edgecolor": background,
                "savefig.facecolor": background,
                "savefig.edgecolor": background,
                "grid.color": "k",
                "axes.grid": False,
            }
        )


def config_dynamo_rcParams(
    background="white", prop_cycle=zebrafish_256, fontsize=8, color_map=None, frameon=None
):
    """Configure matplotlib.rcParams to dynamo defaults (based on ggplot style and scanpy).

    Parameters
    ----------
        background: `str` (default: `white`)
            The background color of the plot. By default we use the white ground
            which is suitable for producing figures for publication. Setting it to `black` background will
            be great for presentation.
        prop_cycle: `list` (default: zebrafish_256)
            A list with hex color codes
        fontsize: float (default: 6)
            Size of font
        color_map: `plt.cm` or None (default: None)
            Color map
        frameon: `bool` or None (default: None)
            Whether to have frame for the figure.
    Returns
    -------
        Nothing but configure the rcParams globally.
    """

    # from http://www.huyng.com/posts/sane-color-scheme-for-matplotlib/

    rcParams["patch.linewidth"] = 0.5
    rcParams["patch.facecolor"] = "348ABD"  # blue
    rcParams["patch.edgecolor"] = "EEEEEE"
    rcParams["patch.antialiased"] = True

    rcParams["font.size"] = 10.0

    rcParams["axes.facecolor"] = "E5E5E5"
    rcParams["axes.edgecolor"] = "white"
    rcParams["axes.linewidth"] = 1
    rcParams["axes.grid"] = True
    # rcParams['axes.titlesize'] =  "x-large"
    # rcParams['axes.labelsize'] = "large"
    rcParams["axes.labelcolor"] = "555555"
    rcParams[
        "axes.axisbelow"
    ] = True  # grid/ticks are below elements (e.g., lines, text)

    # rcParams['axes.prop_cycle'] = cycler('color', ['E24A33', '348ABD', '988ED5', '777777', 'FBC15E', '8EBA42', 'FFB5B8'])
    # # E24A33 : red
    # # 348ABD : blue
    # # 988ED5 : purple
    # # 777777 : gray
    # # FBC15E : yellow
    # # 8EBA42 : green
    # # FFB5B8 : pink

    # rcParams['xtick.color'] = "555555"
    rcParams["xtick.direction"] = "out"

    # rcParams['ytick.color'] = "555555"
    rcParams["ytick.direction"] = "out"

    rcParams["grid.color"] = "white"
    rcParams["grid.linestyle"] = "-"  # solid line

    rcParams["figure.facecolor"] = "white"
    rcParams["figure.edgecolor"] = "white"  # 0.5

    # the following code is modified from scanpy
    # https://github.com/theislab/scanpy/blob/178a0981405ba8ccfd5031eb15bc07b3a45d2730/scanpy/plotting/_rcmod.py

    # dpi options (mpl default: 100, 100)
    rcParams["figure.dpi"] = 100
    rcParams["savefig.dpi"] = 150

    # figure (default: 0.125, 0.96, 0.15, 0.91)
    rcParams["figure.figsize"] = (6, 4)
    rcParams["figure.subplot.left"] = 0.18
    rcParams["figure.subplot.right"] = 0.96
    rcParams["figure.subplot.bottom"] = 0.15
    rcParams["figure.subplot.top"] = 0.91

    # lines (defaults:  1.5, 6, 1)
    rcParams["lines.linewidth"] = 1.5  # the line width of the frame
    rcParams["lines.markersize"] = 6
    rcParams["lines.markeredgewidth"] = 1

    # font
    rcParams["font.sans-serif"] = [
        "Arial",
        "sans-serif",
        "Helvetica",
        "DejaVu Sans",
        "Bitstream Vera Sans",
    ]
    fontsize = fontsize
    labelsize = 0.90 * fontsize

    # fonsizes (default: 10, medium, large, medium)
    rcParams["font.size"] = fontsize
    rcParams["legend.fontsize"] = labelsize
    rcParams["axes.titlesize"] = fontsize
    rcParams["axes.labelsize"] = labelsize

    # legend (default: 1, 1, 2, 0.8)
    rcParams["legend.numpoints"] = 1
    rcParams["legend.scatterpoints"] = 1
    rcParams["legend.handlelength"] = 0.5
    rcParams["legend.handletextpad"] = 0.4

    # color cycle
    rcParams["axes.prop_cycle"] = cycler(color=prop_cycle)  # use tab20c by default

    # lines
    rcParams["axes.linewidth"] = 0.8
    rcParams["axes.edgecolor"] = "black"
    rcParams["axes.facecolor"] = "white"

    # ticks (default: k, k, medium, medium)
    rcParams["xtick.color"] = "k"
    rcParams["ytick.color"] = "k"
    rcParams["xtick.labelsize"] = labelsize
    rcParams["ytick.labelsize"] = labelsize

    # axes grid (default: False, #b0b0b0)
    rcParams["axes.grid"] = False
    rcParams["grid.color"] = ".8"

    # color map
    rcParams["image.cmap"] = "RdBu_r" if color_map is None else color_map

    dyn_theme(background)

    # frame (default: True)
    frameon = False if frameon is None else frameon
    global _frameon
    _frameon = frameon


def set_figure_params(
    dynamo=True,
    background="white",
    fontsize=8,
    figsize=(6, 4),
    dpi=None,
    dpi_save=None,
    frameon=None,
    vector_friendly=True,
    color_map=None,
    format="pdf",
    transparent=False,
    ipython_format="png2x",
):
    """Set resolution/size, styling and format of figures.
       This function is adapted from: https://github.com/theislab/scanpy/blob/f539870d7484675876281eb1c475595bf4a69bdb/scanpy/_settings.py
    Arguments
    ---------
        dynamo: `bool` (default: `True`)
            Init default values for :obj:`matplotlib.rcParams` suited for dynamo.
        background: `str` (default: `white`)
            The background color of the plot. By default we use the white ground
            which is suitable for producing figures for publication. Setting it to `black` background will
            be great for presentation.
        fontsize: `[float, float]` or None (default: `6`)
        figsize: `(float, float)` (default: `(6.5, 5)`)
            Width and height for default figure size.
        dpi: `int` or None (default: `None`)
            Resolution of rendered figures - this influences the size of figures in notebooks.
        dpi_save: `int` or None (default: `None`)
            Resolution of saved figures. This should typically be higher to achieve
            publication quality.
        frameon: `bool` or None (default: `None`)
            Add frames and axes labels to scatter plots.
        vector_friendly: `bool` (default: `True`)
            Plot scatter plots using `png` backend even when exporting as `pdf` or `svg`.
        color_map: `str` (default: `None`)
            Convenience method for setting the default color map.
        format: {'png', 'pdf', 'svg', etc.} (default: 'pdf')
            This sets the default format for saving figures: `file_format_figs`.
        transparent: `bool` (default: `False`)
            Save figures with transparent back ground. Sets `rcParams['savefig.transparent']`.
        ipython_format : list of `str` (default: 'png2x')
            Only concerns the notebook/IPython environment; see
            `IPython.core.display.set_matplotlib_formats` for more details.
    """

    try:
        import IPython

        if isinstance(ipython_format, str):
            ipython_format = [ipython_format]
        IPython.display.set_matplotlib_formats(*ipython_format)
    except Exception:
        pass

    from matplotlib import rcParams

    global _vector_friendly, file_format_figs

    _vector_friendly = vector_friendly
    file_format_figs = format

    if dynamo:
        config_dynamo_rcParams(
            background=background, fontsize=fontsize, color_map=color_map
        )
    if figsize is not None:
        rcParams["figure.figsize"] = figsize

    if dpi is not None:
        rcParams["figure.dpi"] = dpi
    if dpi_save is not None:
        rcParams["savefig.dpi"] = dpi_save
    if transparent is not None:
        rcParams["savefig.transparent"] = transparent
    if frameon is not None:
        global _frameon
        _frameon = frameon


def reset_rcParams():
    """Reset `matplotlib.rcParams` to defaults."""
    from matplotlib import rcParamsDefault

    rcParams.update(rcParamsDefault)


def set_pub_style(scaler=1):
    """formatting helper function that can be used to save publishable figures"""
    set_figure_params('dynamo', background='white')
    matplotlib.use('cairo')
    matplotlib.rcParams.update({'font.size': 4 * scaler})
    params = {'font.size': 4 * scaler,
              'legend.fontsize': 4 * scaler,
              'legend.handlelength': 0.5 * scaler,
              'axes.labelsize': 6 * scaler,
              'axes.titlesize': 6 * scaler,
              'xtick.labelsize': 6 * scaler,
              'ytick.labelsize': 6 * scaler,
              'axes.titlepad': 1 * scaler,
              'axes.labelpad': 1 * scaler
    }
    matplotlib.rcParams.update(params)
