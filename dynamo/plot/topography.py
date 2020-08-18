import warnings
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt

from ..vectorfield.topography import topography as _topology  # , compute_separatrices
from ..tools.utils import update_dict
from ..external.hodge import ddhodge
from ..vectorfield.vector_calculus import curl, divergence
from .utils import default_quiver_args
from .scatters import scatters
from .scatters import docstrings
from .utils import (
    set_arrow_alpha,
    set_stream_line_alpha,
    _plot_traj,
    quiver_autoscaler,
    save_fig,
    _select_font_color,
)
from ..configuration import _themes


def plot_flow_field(
    vecfld,
    x_range,
    y_range,
    n_grid=100,
    start_points=None,
    integration_direction="both",
    background=None,
    density=1,
    linewidth=1,
    streamline_color=None,
    streamline_alpha=0.4,
    color_start_points=None,
    save_show_or_return='return',
    save_kwargs={},
    ax=None,
    **streamline_kwargs,
):
    """Plots the flow field with line thickness proportional to speed.
    code adapted from: http://be150.caltech.edu/2017/handouts/dynamical_systems_approaches.html

    Parameters
    ----------
    vecfld: :class:`~VectorField2D`
        An instance of the VectorField2D class which presumably has fixed points computed and stored.
    x_range: array_like, shape (2,)
        Range of values for x-axis.
    y_range: array_like, shape (2,)
        Range of values for y-axis.
    n_grid : int, default 100
        Number of grid points to use in computing
        derivatives on phase portrait.
    start_points: np.ndarray (default: None)
        The initial points from which the streamline will be draw.
    integration_direction:  {'forward', 'backward', 'both'} (default: `both`)
        Integrate the streamline in forward, backward or both directions. default is 'both'.
    background: `str` or None (default: None)
        The background color of the plot.
    density: `float` or None (default: 1)
        density of the plt.streamplot function.
    linewidth: `float` or None (default: 1)
        multiplier of automatically calculated linewidth passed to the plt.streamplot function.
    streamline_color: `str` or None (default: None)
        The color of the vector field stream lines.
    streamline_alpha: `float` or None (default: 0.4)
        The alpha value applied to the vector field stream lines.
    color_start_points: `float` or None (default: `None`)
        The color of the starting point that will be used to predict cell fates.
    save_show_or_return: {'show', 'save', 'return'} (default: `return`)
        Whether to save, show or return the figure.
    save_kwargs: `dict` (default: `{}`)
        A dictionary that will passed to the save_fig function. By default it is an empty dictionary and the save_fig function
        will use the {"path": None, "prefix": 'plot_flow_field', "dpi": None, "ext": 'pdf', "transparent": True, "close":
        True, "verbose": True} as its parameters. Otherwise you can provide a dictionary that properly modify those keys
        according to your needs.
    ax : Matplotlib Axis instance
        Axis on which to make the plot

    Returns
    -------
    output : Matplotlib Axis instance
        Axis with streamplot included.
    """

    from matplotlib import rcParams, patches

    from matplotlib.colors import to_hex

    if background is None:
        _background = rcParams.get("figure.facecolor")
        _background = to_hex(_background) if type(_background) is tuple else _background
    else:
        _background = background

    if _background in ["#ffffff", "black"]:
        color, color_start_points = "white" if streamline_color is None else streamline_color, "red" if color_start_points is None else color_start_points
    else:
        color, color_start_points = "black" if streamline_color is None else streamline_color, "red" if color_start_points is None else color_start_points

    # Set up u,v space
    u = np.linspace(x_range[0], x_range[1], n_grid)
    v = np.linspace(y_range[0], y_range[1], n_grid)
    uu, vv = np.meshgrid(u, v)

    # Compute derivatives
    u_vel = np.empty_like(uu)
    v_vel = np.empty_like(vv)
    for i in range(uu.shape[0]):
        for j in range(uu.shape[1]):
            u_vel[i, j], v_vel[i, j] = vecfld.func(np.array([uu[i, j], vv[i, j]]))

    # Compute speed
    speed = np.sqrt(u_vel ** 2 + v_vel ** 2)

    # Make linewidths proportional to speed,
    # with minimal line width of 0.5 and max of 3
    # lw = lw_min + (lw_max - lw_min) * speed / speed.max()

    streamplot_kwargs = {
        "density": density * 2,
        "linewidth": None,
        "cmap": None,
        "norm": None,
        "arrowsize": 1,
        "arrowstyle": "fancy",
        "minlength": 0.1,
        "transform": None,
        "maxlength": 4.0,
        "zorder": 3,
    }
    linewidth *= 2 * speed / speed[~np.isnan(speed)].max()
    streamplot_kwargs.update({"linewidth": linewidth})

    streamplot_kwargs = update_dict(streamplot_kwargs, streamline_kwargs)

    # Make stream plot
    if ax is None:
        ax = plt.gca()
    if start_points is None:
        s = ax.streamplot(
            uu, vv, u_vel, v_vel, color=color, **streamplot_kwargs,

        )
        set_arrow_alpha(ax, streamline_alpha)
        set_stream_line_alpha(s, streamline_alpha)
    else:
        if len(start_points.shape) == 1:
            start_points.reshape((1, 2))
        ax.scatter(*start_points, marker="*", zorder=100)

        s = ax.streamplot(
            uu,
            vv,
            u_vel,
            v_vel,
            start_points=start_points,
            integration_direction=integration_direction,
            color=color_start_points,
            **streamplot_kwargs,
        )
        set_arrow_alpha(ax, streamline_alpha)
        set_stream_line_alpha(s, streamline_alpha)

    if save_show_or_return == "save":
        s_kwargs = {"path": None, "prefix": 'plot_flow_field', "dpi": None,
                    "ext": 'pdf', "transparent": True, "close": True, "verbose": True}
        s_kwargs = update_dict(s_kwargs, save_kwargs)

        save_fig(**s_kwargs)
    elif save_show_or_return == "show":
        plt.tight_layout()
        plt.show()
    elif save_show_or_return == "return":
        return ax


def plot_nullclines(vecfld,
                    lw=3,
                    background=None,
                    save_show_or_return='return',
                    save_kwargs={},
                    ax=None):
    """Plot nullclines stored in the VectorField2D class.

    Arguments
    ---------
        vecfld: :class:`~VectorField2D`
            An instance of the VectorField2D class which presumably has fixed points computed and stored.
        lw: `float` (default: 3)
            The linewidth of the nullcline.
        background: `str` or None (default: None)
            The background color of the plot.
        save_show_or_return: {'show', 'save', 'return'} (default: `return`)
            Whether to save, show or return the figure.
        save_kwargs: `dict` (default: `{}`)
            A dictionary that will passed to the save_fig function. By default it is an empty dictionary and the save_fig function
            will use the {"path": None, "prefix": 'plot_nullclines', "dpi": None, "ext": 'pdf', "transparent": True, "close":
            True, "verbose": True} as its parameters. Otherwise you can provide a dictionary that properly modify those keys
            according to your needs.
        ax: :class:`~matplotlib.axes.Axes`
            The matplotlib axes used for plotting. Default is to use the current axis.
    """
    from matplotlib import rcParams

    from matplotlib.colors import to_hex

    if background is None:
        _background = rcParams.get("figure.facecolor")
        _background = to_hex(_background) if type(_background) is tuple else _background
    else:
        _background = background

    if _background in ["#ffffff", "black"]:
        colors = ["#189e1a", "#1f77b4"]
    else:
        colors = ["#189e1a", "#1f77b4"]

    if ax is None:
        ax = plt.gca()
    for ncx in vecfld.NCx:
        ax.plot(*ncx.T, c=colors[0], lw=lw)
    for ncy in vecfld.NCy:
        ax.plot(*ncy.T, c=colors[1], lw=lw)

    if save_show_or_return == "save":
        s_kwargs = {"path": None, "prefix": 'plot_nullclines', "dpi": None,
                    "ext": 'pdf', "transparent": True, "close": True, "verbose": True}
        s_kwargs = update_dict(s_kwargs, save_kwargs)

        save_fig(**s_kwargs)
    elif save_show_or_return == "show":
        plt.tight_layout()
        plt.show()
    elif save_show_or_return == "return":
        return ax


def plot_fixed_points(
    vecfld,
    marker="o",
    markersize=200,
    cmap=None,
    filltype=["full", "top", "none"],
    background=None,
    save_show_or_return='return',
    save_kwargs={},
    ax=None,
):
    """Plot fixed points stored in the VectorField2D class.

    Arguments
    ---------
        vecfld: :class:`~VectorField2D`
            An instance of the VectorField2D class which presumably has fixed points computed and stored.
        marker: `str` (default: `o`)
            The marker type. Any string supported by matplotlib.markers.
        markersize: `float` (default: 200)
            The size of the marker.
        cmap: string (optional, default 'Blues')
            The name of a matplotlib colormap to use for coloring or shading the confidence of fixed points. If None, the
            default color map will set to be viridis (inferno) when the background is white (black).
        filltype: list
            The fill type used for stable, saddle, and unstable fixed points. Default is 'full', 'top' and 'none',
            respectively.
        background: `str` or None (default: None)
            The background color of the plot.
        save_show_or_return: {'show', 'save', 'return'} (default: `return`)
            Whether to save, show or return the figure.
        save_kwargs: `dict` (default: `{}`)
            A dictionary that will passed to the save_fig function. By default it is an empty dictionary and the save_fig function
            will use the {"path": None, "prefix": 'plot_fixed_points', "dpi": None, "ext": 'pdf', "transparent": True, "close":
            True, "verbose": True} as its parameters. Otherwise you can provide a dictionary that properly modify those keys
            according to your needs.
        ax: :class:`~matplotlib.axes.Axes`
            The matplotlib axes used for plotting. Default is to use the current axis.
    """

    from matplotlib import rcParams, markers
    import matplotlib.patheffects as PathEffects
    from matplotlib.colors import to_hex

    if background is None:
        _background = rcParams.get("figure.facecolor")
        _background = to_hex(_background) if type(_background) is tuple else _background
    else:
        _background = background

    if _background in ["#ffffff", "black"]:
        _theme_ = (
            "inferno"
        )
    else:
        _theme_ = (
            "viridis"
        )
    _cmap = _themes[_theme_]["cmap"] if cmap is None else cmap

    Xss, ftype = vecfld.get_fixed_points(get_types=True)
    confidence = vecfld.get_Xss_confidence()
    if ax is None:
        ax = plt.gca()

    for i in range(len(Xss)):
        cur_ftype = ftype[i]
        marker_ = markers.MarkerStyle(marker=marker, fillstyle=filltype[int(cur_ftype + 1)])
        ax.scatter(
            *Xss[i],
            marker=marker_,
            s=markersize,
            c=confidence[i],
            edgecolor=_select_font_color(_background),
            linewidths=1,
            cmap=_cmap,
            vmin=0,
            vmax=1,
            zorder=1000
        )
        txt = ax.text(*Xss[i], repr(i), c=('black' if cur_ftype == -1 else 'blue' if cur_ftype == 0 else 'red'),
                horizontalalignment='center', verticalalignment='center', zorder=1001, weight='bold',
        )
        txt.set_path_effects(
            [
                PathEffects.Stroke(linewidth=1.5, foreground=_background, alpha=0.8),
                PathEffects.Normal(),
            ]
        )

    if save_show_or_return == "save":
        s_kwargs = {"path": None, "prefix": 'plot_fixed_points', "dpi": None,
                    "ext": 'pdf', "transparent": True, "close": True, "verbose": True}
        s_kwargs = update_dict(s_kwargs, save_kwargs)

        save_fig(**s_kwargs)
    elif save_show_or_return == "show":
        plt.tight_layout()
        plt.show()
    elif save_show_or_return == "return":
        return ax


def plot_traj(f,
              y0,
              t,
              args=(),
              lw=2,
              background=None,
              integration_direction="both",
              save_show_or_return='return',
              save_kwargs={},
              ax=None
):
    """Plots a trajectory on a phase portrait.
    code adapted from: http://be150.caltech.edu/2017/handouts/dynamical_systems_approaches.html

    Parameters
    ----------
    f : function for form f(y, t, *args)
        The right-hand-side of the dynamical system.
        Must return a 2-array.
    y0 : array_like, shape (2,)
        Initial condition.
    t : array_like
        Time points for trajectory.
    args : tuple, default ()
        Additional arguments to be passed to f
    lw : `float`, (default: 2)
        The line width of the trajectory.
    background: `str` or None (default: None)
        The background color of the plot.
    integration_direction: `str` (default: `forward`)
        Integrate the trajectory in forward, backward or both directions. default is 'both'.
    save_show_or_return: {'show', 'save', 'return'} (default: `return`)
        Whether to save, show or return the figure.
    save_kwargs: `dict` (default: `{}`)
        A dictionary that will passed to the save_fig function. By default it is an empty dictionary and the save_fig function
        will use the {"path": None, "prefix": 'plot_traj', "dpi": None, "ext": 'pdf', "transparent": True, "close":
        True, "verbose": True} as its parameters. Otherwise you can provide a dictionary that properly modify those keys
        according to your needs.
    ax : Matplotlib Axis instance
        Axis on which to make the plot

    Returns
    -------
    output : Matplotlib Axis instance
        Axis with streamplot included.
    """
    from matplotlib import rcParams
    from matplotlib.colors import to_hex

    if background is None:
        _background = rcParams.get("figure.facecolor")
        _background = to_hex(_background) if type(_background) is tuple else _background
    else:
        _background = background

    if background in ["#ffffff", "black"]:
        color = ["#ffffff"]
    else:
        color = "black"

    if len(y0.shape) == 1:
        ax = _plot_traj(y0, t, args, integration_direction, ax, color, lw, f)
    else:
        for i in range(y0.shape[0]):
            cur_y0 = y0[i, None]  # don't drop dimension
            ax = _plot_traj(cur_y0, t, args, integration_direction, ax, color, lw, f)

    if save_show_or_return == "save":
        s_kwargs = {"path": None, "prefix": 'plot_traj', "dpi": None,
                    "ext": 'pdf', "transparent": True, "close": True, "verbose": True}
        s_kwargs = update_dict(s_kwargs, save_kwargs)

        save_fig(**s_kwargs)
    elif save_show_or_return == "show":
        plt.tight_layout()
        plt.show()
    elif save_show_or_return == "return":
        return ax


def plot_separatrix(vecfld,
                    x_range,
                    y_range,
                    t,
                    noise=1e-6,
                    lw=3,
                    background=None,
                    save_show_or_return='return',
                    save_kwargs={},
                    ax=None
):
    """Plot separatrix on phase portrait.

    Parameters
    ----------
        vecfld: :class:`~VectorField2D`
            An instance of the VectorField2D class which presumably has fixed points computed and stored.
        x_range: array_like, shape (2,)
            Range of values for x-axis.
        y_range: array_like, shape (2,)
        t : array_like
            Time points for trajectory.
        noise : tuple, default ()
            A small noise added to steady states for drawing the separatrix.
        lw : `float`, (default: 2)
            The line width of the trajectory.
        background: `str` or None (default: None)
            The background color of the plot.
        save_show_or_return: {'show', 'save', 'return'} (default: `return`)
            Whether to save, show or return the figure.
        save_kwargs: `dict` (default: `{}`)
            A dictionary that will passed to the save_fig function. By default it is an empty dictionary and the save_fig function
            will use the {"path": None, "prefix": 'plot_separatrix', "dpi": None, "ext": 'pdf', "transparent": True, "close":
            True, "verbose": True} as its parameters. Otherwise you can provide a dictionary that properly modify those keys
            according to your needs.
        ax : Matplotlib Axis instance
            Axis on which to make the plot

        code adapted from: http://be150.caltech.edu/2017/handouts/dynamical_systems_approaches.html

    """
    from matplotlib import rcParams
    from matplotlib.colors import to_hex

    if background is None:
        _background = rcParams.get("figure.facecolor")
        _background = to_hex(_background) if type(_background) is tuple else _background
    else:
        _background = background

    if _background in ["#ffffff", "black"]:
        color = ["#ffffff"]
    else:
        color = "tomato"

    # No saddle point, no separatrix.
    fps, ftypes = vecfld.get_fixed_points(get_types=True)
    J = vecfld.Xss.get_J()
    saddle = fps[ftypes == 0]
    Jacobian = J[[ftypes == 0]]
    if len(saddle) > 0:
        # Negative time function to integrate to compute separatrix
        def rhs(ab, t):
            # Unpack variables
            a, b = ab
            # Stop integrating if we get the edge of where we want to integrate
            if x_range[0] < a < x_range[1] and y_range[0] < b < y_range[1]:
                return -vecfld.func(ab)
            else:
                return np.array([0, 0])

        # Parameters for building separatrix
        # t = np.linspace(0, t_max, 400)
        all_sep_a, all_sep_b = None, None
        if ax is None:
            ax = plt.gca()
        for i in range(len(saddle)):
            fps = saddle[i]
            J = Jacobian[i]
            # Build upper right branch of separatrix
            ab0 = fps + noise
            ab_upper = scipy.integrate.odeint(rhs, ab0, t)

            # Build lower left branch of separatrix
            ab0 = fps - noise
            ab_lower = scipy.integrate.odeint(rhs, ab0, t)

            # Concatenate, reversing lower so points are sequential
            sep_a = np.concatenate((ab_lower[::-1, 0], ab_upper[:, 0]))
            sep_b = np.concatenate((ab_lower[::-1, 1], ab_upper[:, 1]))

            # Plot
            ax.plot(sep_a, sep_b, "-", color=color, lw=lw)

            all_sep_a = (
                sep_a if all_sep_a is None else np.concatenate((all_sep_a, sep_a))
            )
            all_sep_b = (
                sep_b if all_sep_b is None else np.concatenate((all_sep_b, sep_b))
            )

    if save_show_or_return == "save":
        s_kwargs = {"path": None, "prefix": 'plot_separatrix', "dpi": None,
                    "ext": 'pdf', "transparent": True, "close": True, "verbose": True}
        s_kwargs = update_dict(s_kwargs, save_kwargs)

        save_fig(**s_kwargs)
    elif save_show_or_return == "show":
        plt.tight_layout()
        plt.show()
    elif save_show_or_return == "return":
        return ax


@docstrings.with_indent(4)
def topography(
    adata,
    basis="umap",
    x=0,
    y=1,
    color='ntr',
    layer="X",
    highlights=None,
    labels=None,
    values=None,
    theme=None,
    cmap=None,
    color_key=None,
    color_key_cmap=None,
    background='white',
    ncols=4,
    pointsize=None,
    figsize=(6, 4),
    show_legend='on data',
    use_smoothed=True,
    xlim=None,
    ylim=None,
    t=None,
    terms=("streamline", "fixed_points"),
    init_cells=None,
    init_states=None,
    quiver_source="raw",
    fate="both",
    approx=False,
    quiver_size=None,
    quiver_length=None,
    density=1,
    linewidth=1,
    streamline_color=None,
    streamline_alpha=0.4,
    color_start_points=None,
    markersize=200,
    marker_cmap=None,
    save_show_or_return='show',
    save_kwargs={},
    aggregate=None,
    show_arrowed_spines=False,
    ax=None,
    sort='raw',
    frontier=False,
    s_kwargs_dict={},
    q_kwargs_dict={},
    **streamline_kwargs_dict
):
    """Plot the streamline, fixed points (attractor / saddles), nullcline, separatrices of a recovered dynamic system
    for single cells. The plot is created on two dimensional space.

    Topography function plots the full vector field topology including streamline, fixed points, characteristic lines. A key
    difference between dynamo and Velocyto or scVelo is that we learn a functional form of a vector field which can be
    used to predict cell fate over arbitrary time and space. On states near observed cells, it retrieves the key kinetics
    dynamics from the data observed and smoothes them. When time and state is far from your observed single cell RNA-seq
    datasets, the accuracy of prediction will decay. Vector field can be efficiently reconstructed in high dimension or
    lower pca/umap space. Since we learn a vector field function, we can plot the full vector via streamline on the entire
    domain as well as predicts cell fates by providing a set of initial cell states (via `init_cells`, `init_states`). The
    nullcline and separatrix provide topological information about the reconstructed vector field. By definition, the
    x/y-nullcline is a set of points in the phase plane so that dx/dt = 0 or dy/dt=0. Geometrically, these are the points
    where the vectors are either straight up or straight down. Algebraically, we find the x-nullcline by solving
    f(x,y) = 0. The boundary different different attractor basis is the separatrix because it separates the regions into
    different subregions with a specific behavior. To find them is a very difficult problem and separatrix calculated by
    dynamo requres manual inspection.

    Here is more details on the fixed points drawn on the vector field:  Fixed points are concepts introduced in dynamic
    systems theory. There are three types of fixed points: 1) repeller: a repelling state that only has outflows, which
    may correspond to a pluripotent cell state (ESC) that tends to differentiate into other cell states automatically or
    under small perturbation; 2) unstable fixed points or saddle points. Those states have attraction on some dimension
    (genes or reduced dimensions) but diverge in at least one other dimension. Saddle may correspond to progenitors, which
    are differentiated from ESC/pluripotent cells and relatively stable, but can further differentiate into multiple
    terminal cell types / states; 3) lastly, stable fixed points / cell type or attractors, which only have inflows and
    attract all cell states nearby, which may correspond to stable cell types and can only be kicked out of its cell
    state under extreme perturbation or in very rare situation. Fixed points are numbered with each number color coded.
    The mapping of the color of the number to the type of fixed point are: red: repellers; blue: saddle points; black:
    attractors. The scatter point itself also has filled color, which corresponds to confidence of the estimated fixed
    point. The lighter, the more confident or the fixed points are are closer to the sequenced single cells. Confidence
    of each fixed points can be used in conjunction with the Jacobian analysis for investigating regulatory network with
    spatiotemporal resolution.

    By default, we plot a figure with three subplots , each colors cells either with `potential`, `curl` or `divergence`.
    `potential` is related to the intrinsic time, where a small potential is related to smaller intrinsic time and vice
    versa. Divergence can be used to indicate the state of each cell is in. Negative values correspond to potential sink
    while positive corresponds to potential source. https://en.wikipedia.org/wiki/Divergence. Curl may be related to cell
    cycle or other cycling cell dynamics. On 2d, negative values correspond to clockwise rotation while positive corresponds
    to anticlockwise rotation. https://www.khanacademy.org/math/multivariable-calculus/greens-theorem-and-stokes-theorem/formal-definitions-of-divergence-and-curl/a/defining-curl
    In conjunction with cell cycle score (dyn.pp.cell_cycle_scores), curl can be used to identify cells under active cell
    cycle progression.

    Parameters
    ----------
        %(scatters.parameters.no_show_legend|kwargs|save_kwargs)s
        xlim: `numpy.ndarray`
            The range of x-coordinate
        ylim: `numpy.ndarray`
            The range of y-coordinate
        t:  t_end: `float` (default 1)
            The length of the time period from which to predict cell state forward or backward over time. This is used
            by the odeint function.
        terms: `tuple` (default: ('streamline', 'fixed_points'))
            A tuple of plotting items to include in the final topography figure.  ('streamline', 'nullcline', 'fixed_points',
             'separatrix', 'trajectory', 'quiver') are all the items that we can support.
        init_cells: `list` (default: None)
            Cell name or indices of the initial cell states for the historical or future cell state prediction with numerical integration.
            If the names in init_cells are not find in the adata.obs_name, it will be treated as cell indices and must be integers.
        init_state: `numpy.ndarray` (default: None)
            Initial cell states for the historical or future cell state prediction with numerical integration. It can be
            either a one-dimensional array or N x 2 dimension array. The `init_state` will be replaced to that defined by init_cells if
            init_cells are not None.
        quiver_source: `numpy.ndarray` {'raw', 'reconstructed'} (default: None)
            The data source that will be used to draw the quiver plot. If `init_cells` is provided, this will set to be the projected RNA
            velocity before vector field reconstruction automatically. If `init_cells` is not provided, this will set to be the velocity
            vectors calculated from the reconstructed vector field function automatically. If quiver_source is `reconstructed`, the velocity
            vectors calculated from the reconstructed vector field function will be used.
        fate: `str` {"history", 'future', 'both'} (default: `both`)
            Predict the historial, future or both cell fates. This corresponds to integrating the trajectory in forward,
            backward or both directions defined by the reconstructed vector field function. default is 'both'.
        approx: `bool` (default: False)
            Whether to use streamplot to draw the integration line from the init_state.
        quiver_size: `float` or None (default: None)
            The size of quiver. If None, we will use set quiver_size to be 1. Note that quiver quiver_size is used to calculate
            the head_width (10 x quiver_size), head_length (12 x quiver_size) and headaxislength (8 x quiver_size) of the quiver.
            This is done via the `default_quiver_args` function which also calculate the scale of the quiver (1 / quiver_length).
        quiver_length: `float` or None (default: None)
            The length of quiver. The quiver length which will be used to calculate scale of quiver. Note that befoe applying
            `default_quiver_args` velocity values are first rescaled via the quiver_autoscaler function. Scale of quiver indicates
            the number of data units per arrow length unit, e.g., m/s per plot width; a smaller scale parameter makes the arrow longer.
        density: `float` or None (default: 1)
            density of the plt.streamplot function.
        linewidth: `float` or None (default: 1)
            multiplier of automatically calculated linewidth passed to the plt.streamplot function.
        streamline_color: `str` or None (default: None)
            The color of the vector field stream lines.
        streamline_alpha: `float` or None (default: 0.4)
            The alpha value applied to the vector field stream lines.
        color_start_points: `float` or None (default: `None`)
            The color of the starting point that will be used to predict cell fates.
        markersize: `float` (default: 200)
            The size of the marker.
        marker_cmap: string (optional, default 'Blues')
            The name of a matplotlib colormap to use for coloring or shading the confidence of fixed points. If None, the
            default color map will set to be viridis (inferno) when the background is white (black).
        save_show_or_return: {'show', 'save', 'return'} (default: `show`)
            Whether to save, show or return the figure.
        save_kwargs: `dict` (default: `{}`)
            A dictionary that will passed to the save_fig function. By default it is an empty dictionary and the save_fig function
            will use the {"path": None, "prefix": 'topography', "dpi": None, "ext": 'pdf', "transparent": True, "close":
            True, "verbose": True} as its parameters. Otherwise you can provide a dictionary that properly modify those keys
            according to your needs.
        aggregate: `str` or `None` (default: `None`)
            The column in adata.obs that will be used to aggregate data points.
        show_arrowed_spines: bool (optional, default False)
            Whether to show a pair of arrowed spines representing the basis of the scatter is currently using.
        ax: Matplotlib Axis instance
            Axis on which to make the plot
        frontier: `bool` (default: `False`)
            Whether to add the frontier. Scatter plots can be enhanced by using transparency (alpha) in order to show area
            of high density and multiple scatter plots can be used to delineate a frontier. See matplotlib tips & tricks
            cheatsheet (https://github.com/matplotlib/cheatsheets). Originally inspired by figures from scEU-seq paper:
            https://science.sciencemag.org/content/367/6482/1151.
        s_kwargs_dict: `dict` (default: {})
            The dictionary of the scatter arguments.
        q_kwargs_dict:
            Additional parameters that will be passed to plt.quiver function
        streamline_kwargs_dict:
            Additional parameters that will be passed to plt.streamline function

    Returns
    -------
        Plot the streamline, fixed points (attractors / saddles), nullcline, separatrices of a recovered dynamic system
        for single cells or return the corresponding axis, depending on the plot argument.

    See also:: :func:`pp.cell_cycle_scores`
    """

    from matplotlib import rcParams
    from matplotlib.colors import to_hex

    if color is None:
        color = ['potential', 'curl', 'divergence'] if adata.n_obs < 5000 else ['curl', 'divergence']

    if type(color) == str: color = [color]
    if len(set(adata.obs.columns).intersection(color)) == 0:
        if adata.obs.keys().isin(['potential']).sum() == 0:
            ddhodge(adata, basis=basis)
        if adata.obs.keys().isin(['curl']).sum() == 0:
            curl(adata, basis=basis)
        if adata.obs.keys().isin(['divergence']).sum() == 0:
            divergence(adata, basis=basis)

    if background is None:
        _background = rcParams.get("figure.facecolor")
        _background = to_hex(_background) if type(_background) is tuple else _background
    else:
        _background = background

    terms = list(terms) if type(terms) is tuple else [terms] if type(terms) is str else terms
    if approx:
        if "streamline" not in terms:
            terms.append("streamline")
        if "trajectory" in terms:
            terms = list(set(terms).difference("trajectory"))

    if init_cells is not None or init_states is not None:
        terms.extend('trajectory')

    uns_key = "VecFld" if basis == "X" else "VecFld_" + basis

    if uns_key not in adata.uns.keys():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            _topology(adata, basis, VecFld=None)
    elif "VecFld2D" not in adata.uns[uns_key].keys():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            _topology(adata, basis, VecFld=None)        
    elif (
        "VecFld2D" in adata.uns[uns_key].keys()
        and type(adata.uns[uns_key]["VecFld2D"]) == str
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            _topology(adata, basis, VecFld=None)

    VF, vecfld = adata.uns[uns_key]["VecFld"], adata.uns[uns_key]["VecFld2D"]
    xlim, ylim = (
        adata.uns[uns_key]["xlim"] if xlim is None else xlim,
        adata.uns[uns_key]["ylim"] if ylim is None else ylim,
    )

    if init_cells is not None:
        intersect_cell_names = list(set(init_cells).intersection(adata.obs_names))
        _init_states = (
            adata.obsm["X_" + basis][init_cells, :]
            if len(intersect_cell_names) == 0
            else adata[intersect_cell_names].obsm["X_" + basis].copy()
        )
        V = (
            adata.obsm["velocity_" + basis][init_cells, :]
            if len(intersect_cell_names) == 0
            else adata[intersect_cell_names].obsm["velocity_" + basis].copy()
        )

        if init_states is None:
            init_states = _init_states

    if quiver_source == "reconstructed" or (
        init_states is not None and init_cells is None
    ):
        from ..tools.utils import vector_field_function

        V = vector_field_function(init_states, VF, [0, 1])

    # plt.figure(facecolor=_background)
    axes_list, color_list, font_color = scatters(
        adata=adata,
        basis=basis,
        x=x,
        y=y,
        color=color,
        layer=layer,
        highlights=highlights,
        labels=labels,
        values=values,
        theme=theme,
        cmap=cmap,
        color_key=color_key,
        color_key_cmap=color_key_cmap,
        background=_background,
        ncols=ncols,
        pointsize=pointsize,
        figsize=figsize,
        show_legend=show_legend,
        use_smoothed=use_smoothed,
        aggregate=aggregate,
        show_arrowed_spines=show_arrowed_spines,
        ax=ax,
        sort=sort,
        save_show_or_return="return",
        frontier=frontier,
        **s_kwargs_dict,
        return_all=True,
    )

    if type(axes_list) != list:
        axes_list, color_list, font_color = [axes_list], [color_list], [font_color]
    for i in range(len(axes_list)):
        # ax = axes_list[i]

        axes_list[i].set_xlabel(basis + "_1")
        axes_list[i].set_ylabel(basis + "_2")
        axes_list[i].set_aspect("equal")

        # Build the plot
        axes_list[i].set_xlim(xlim)
        axes_list[i].set_ylim(ylim)

        axes_list[i].set_facecolor(background)

        if t is None:
            max_t = np.max((np.diff(xlim), np.diff(ylim))) / np.min(
                np.abs(VF["grid_V"])
            )

            t = np.linspace(0, max_t, 10 ** (np.min((int(np.log10(max_t)), 8))))

        integration_direction = (
            "both"
            if fate == "both"
            else "forward"
            if fate == "future"
            else "backward"
            if fate == "history"
            else "both"
        )

        if "streamline" in terms:
            if approx:
                axes_list[i] = plot_flow_field(
                    vecfld,
                    xlim,
                    ylim,
                    background=_background,
                    start_points=init_states,
                    integration_direction=integration_direction,
                    density=density,
                    linewidth=linewidth,
                    streamline_color=streamline_color,
                    streamline_alpha=streamline_alpha,
                    color_start_points=color_start_points,
                    ax=axes_list[i],
                    **streamline_kwargs_dict,
                )
            else:
                axes_list[i] = plot_flow_field(
                    vecfld,
                    xlim,
                    ylim,
                    background=_background,
                    density=density,
                    linewidth=linewidth,
                    streamline_color=streamline_color,
                    color_start_points=color_start_points,
                    ax=axes_list[i],
                    **streamline_kwargs_dict,
                )

        if "nullcline" in terms:
            axes_list[i] = plot_nullclines(
                vecfld, background=_background, ax=axes_list[i]
            )

        if "fixed_points" in terms:
            axes_list[i] = plot_fixed_points(
                vecfld, background=_background, ax=axes_list[i], markersize=markersize, cmap=marker_cmap,
            )

        if "separatrices" in terms:
            axes_list[i] = plot_separatrix(
                vecfld, xlim, ylim, t=t, background=_background, ax=axes_list[i]
            )

        if init_states is not None and "trajectory" in terms:
            if not approx:
                axes_list[i] = plot_traj(
                    vecfld.func,
                    init_states,
                    t,
                    background=_background,
                    integration_direction=integration_direction,
                    ax=axes_list[i],
                )

        # show quivers for the init_states cells
        if init_states is not None and "quiver" in terms:
            X = init_states
            V /= 3 * quiver_autoscaler(X, V)

            df = pd.DataFrame({"x": X[:, 0], "y": X[:, 1], "u": V[:, 0], "v": V[:, 1]})

            if quiver_size is None:
                quiver_size = 1
            if _background in ["#ffffff", "black"]:
                edgecolors = "white"
            else:
                edgecolors = "black"

            head_w, head_l, ax_l, scale = default_quiver_args(
                quiver_size, quiver_length
            )  #
            quiver_kwargs = {
                "angles": "xy",
                "scale": scale,
                "scale_units": "xy",
                "width": 0.0005,
                "headwidth": head_w,
                "headlength": head_l,
                "headaxislength": ax_l,
                "minshaft": 1,
                "minlength": 1,
                "pivot": "tail",
                "linewidth": 0.1,
                "edgecolors": edgecolors,
                "alpha": 1,
                "zorder": 10,
            }
            quiver_kwargs = update_dict(quiver_kwargs, q_kwargs_dict)
            # axes_list[i].quiver(X_grid[:, 0], X_grid[:, 1], V_grid[:, 0], V_grid[:, 1], **quiver_kwargs)
            axes_list[i].quiver(
                df.iloc[:, 0],
                df.iloc[:, 1],
                df.iloc[:, 2],
                df.iloc[:, 3],
                **quiver_kwargs
            )  # color='red',  facecolors='gray'

    if save_show_or_return == "save":
        s_kwargs = {"path": None, "prefix": 'topography', "dpi": None,
                    "ext": 'pdf', "transparent": True, "close": True, "verbose": True}
        s_kwargs = update_dict(s_kwargs, save_kwargs)

        save_fig(**s_kwargs)
    elif save_show_or_return == "show":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
        
            plt.tight_layout()
        
        plt.show()
    elif save_show_or_return == "return":
        return axes_list if len(axes_list) > 1 else axes_list[0]
