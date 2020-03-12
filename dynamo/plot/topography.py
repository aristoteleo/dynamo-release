import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt

from ..tools.topography import topography as _topology  # , compute_separatrices
from ..configuration import set_figure_params
from .scatters import scatters
from .scatters import docstrings
from .utils import _plot_traj, quiver_autoscaler


def plot_flow_field(
    vecfld,
    x_range,
    y_range,
    n_grid=100,
    lw_min=0.5,
    lw_max=3,
    start_points=None,
    integration_direction="both",
    background=None,
    ax=None,
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
    lw_min, lw_max: `float` (defaults: 0.5, 3)
        The smallest and largest linewidth allowed for the stream lines.
    start_points: np.ndarray (default: None)
        The initial points from which the streamline will be draw.
    integration_direction:  {'forward', 'backward', 'both'} (default: `both`)
        Integrate the streamline in forward, backward or both directions. default is 'both'.
    background: `str` or None (default: None)
        The background color of the plot.
    ax : Matplotlib Axis instance
        Axis on which to make the plot

    Returns
    -------
    output : Matplotlib Axis instance
        Axis with streamplot included.
    """

    from matplotlib import rcParams

    if background is not None:
        set_figure_params(background=background)
    else:
        background = rcParams.get("figure.facecolor")

    if background == "black":
        color, color_start_points = "white", "red"
    else:
        color, color_start_points = "thistle", "tomato"

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
    lw = lw_min + (lw_max - lw_min) * speed / speed.max()

    # Make stream plot
    if ax is None:
        ax = plt.gca()
    if start_points is None:
        ax.streamplot(
            uu, vv, u_vel, v_vel, linewidth=lw, arrowsize=1.2, density=1, color=color
        )
    else:
        if len(start_points.shape) == 1:
            start_points.reshape((1, 2))
        ax.scatter(*start_points, marker="*", zorder=100)

        ax.streamplot(
            uu,
            vv,
            u_vel,
            v_vel,
            linewidth=lw_max,
            arrowsize=1.2,
            start_points=start_points,
            integration_direction=integration_direction,
            density=10,
            color=color_start_points,
        )

    return ax


def plot_nullclines(vecfld, lw=3, background=None, ax=None):
    """Plot nullclines stored in the VectorField2D class.

    Arguments
    ---------
        vecfld: :class:`~VectorField2D`
            An instance of the VectorField2D class which presumably has fixed points computed and stored.
        lw: `float` (default: 3)
            The linewidth of the nullcline.
        background: `str` or None (default: None)
            The background color of the plot.
        ax: :class:`~matplotlib.axes.Axes`
            The matplotlib axes used for plotting. Default is to use the current axis.
    """
    from matplotlib import rcParams

    if background is not None:
        set_figure_params(background=background)
    else:
        background = rcParams.get("figure.facecolor")

    if background == "black":
        colors = ["#189e1a", "#1f77b4"]
    else:
        colors = ["#189e1a", "#1f77b4"]

    if ax is None:
        ax = plt.gca()
    for ncx in vecfld.NCx:
        ax.plot(*ncx.T, c=colors[0], lw=lw)
    for ncy in vecfld.NCy:
        ax.plot(*ncy.T, c=colors[1], lw=lw)

    return ax


def plot_fixed_points(
    vecfld,
    marker="o",
    markersize=10,
    filltype=["full", "top", "none"],
    background=None,
    ax=None,
):
    """Plot fixed points stored in the VectorField2D class.
    
    Arguments
    ---------
        vecfld: :class:`~VectorField2D`
            An instance of the VectorField2D class which presumably has fixed points computed and stored.
        marker: `str` (default: `o`)
            The marker type. Any string supported by matplotlib.markers.
        markersize: `float` (default: 20)
            The size of the marker.
        filltype: list
            The fill type used for stable, saddle, and unstable fixed points. Default is 'full', 'top' and 'none',
            respectively.
        background: `str` or None (default: None)
            The background color of the plot.
        ax: :class:`~matplotlib.axes.Axes`
            The matplotlib axes used for plotting. Default is to use the current axis.
    """
    from matplotlib import rcParams

    if background is not None:
        set_figure_params(background=background)
    else:
        background = rcParams.get("figure.facecolor")

    if background == "black":
        markercolor = "#ffffff"
    else:
        markercolor = "k"

    Xss, ftype = vecfld.get_fixed_points(get_types=True)
    if ax is None:
        ax = plt.gca()
    for i in range(len(Xss)):
        ax.plot(
            *Xss[i],
            marker=marker,
            markersize=markersize,
            c=markercolor,
            fillstyle=filltype[int(ftype[i] + 1)],
            linestyle="none"
        )

    return ax


def plot_traj(
    f, y0, t, args=(), lw=2, background=None, integration_direction="both", ax=None
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
    ax : Matplotlib Axis instance
        Axis on which to make the plot

    Returns
    -------
    output : Matplotlib Axis instance
        Axis with streamplot included.
    """
    from matplotlib import rcParams

    if background is not None:
        set_figure_params(background=background)
    else:
        background = rcParams.get("figure.facecolor")

    if background == "black":
        color = ["#ffffff"]
    else:
        color = "black"

    if len(y0.shape) == 1:
        ax = _plot_traj(y0, t, args, integration_direction, ax, color, lw, f)
    else:
        for i in range(y0.shape[0]):
            cur_y0 = y0[i, None]  # don't drop dimension
            ax = _plot_traj(cur_y0, t, args, integration_direction, ax, color, lw, f)

    return ax


def plot_separatrix(
    vecfld, x_range, y_range, t, noise=1e-6, lw=3, background=None, ax=None
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
        ax : Matplotlib Axis instance
            Axis on which to make the plot

        code adapted from: http://be150.caltech.edu/2017/handouts/dynamical_systems_approaches.html

    """
    from matplotlib import rcParams

    if background is not None:
        set_figure_params(background=background)
    else:
        background = rcParams.get("figure.facecolor")

    if background == "black":
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

    return ax


@docstrings.with_indent(4)
def topography(
    adata,
    basis="umap",
    x=0,
    y=1,
    color=None,
    layer="X",
    highlights=None,
    labels=None,
    values=None,
    theme=None,
    cmap=None,
    color_key=None,
    color_key_cmap=None,
    background=None,
    ncols=1,
    pointsize=None,
    figsize=(7, 5),
    show_legend=True,
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
    ax=None,
    aggregate=None,
    s_kwargs_dict={},
    q_kwargs_dict={},
    **topography_kwargs
):
    """Plot the streamline, fixed points (attractor / saddles), nullcline, separatrices of a recovered dynamic system
    for single cells. The plot is created on two dimensional space.

    Parameters
    ----------
        %(scatters.parameters.no_show_legend|kwargs)s
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
            the nuumber of data units per arrow length unit, e.g., m/s per plot width; a smaller scale parameter makes the arrow longer.
        ax: Matplotlib Axis instance
            Axis on which to make the plot
        s_kwargs_dict: `dict` (default: {})
            The dictionary of the scatter arguments.
        q_kwargs_dict:
            Additional parameters that will be passed to plt.quiver function
        topography_kwargs:
            Additional parameters that will be passed to plt.quiver function

    Returns
    -------
        Plot the streamline, fixed points (attractors / saddles), nullcline, separatrices of a recovered dynamic system
        for single cells or return the corresponding axis, depending on the plot argument.
    """
    from matplotlib import rcParams

    if background is not None:
        set_figure_params(background=background)
    else:
        background = rcParams.get("figure.facecolor")

    if approx:
        if "streamline" not in terms:
            terms.append("streamline")
        if "trajectory" in terms:
            terms = list(set(terms).difference("trajectory"))

    uns_key = "VecFld" if basis == "X" else "VecFld_" + basis

    if uns_key not in adata.uns.keys():
        _topology(adata, basis, VecFld=None)
    elif "VecFld2D" not in adata.uns[uns_key].keys():
        _topology(adata, basis, VecFld=None)
    elif (
        "VecFld2D" in adata.uns[uns_key].keys()
        and type(adata.uns[uns_key]["VecFld2D"]) == str
    ):
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

        V = vector_field_function(init_states, None, VF, [0, 1])

    axes_list, color_list, font_color = scatters(
        adata,
        basis,
        x,
        y,
        color,
        layer,
        highlights,
        labels,
        values,
        theme,
        cmap,
        color_key,
        color_key_cmap,
        background,
        ncols,
        pointsize,
        figsize,
        show_legend,
        use_smoothed,
        ax,
        "return",
        aggregate,
        **s_kwargs_dict
    )

    for i in range(len(axes_list)):
        # ax = axes_list[i]

        axes_list[i].set_xlabel(basis + "_1")
        axes_list[i].set_ylabel(basis + "_2")
        axes_list[i].set_aspect("equal")

        # Build the plot
        axes_list[i].set_xlim(xlim)
        axes_list[i].set_ylim(ylim)

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
                    background=background,
                    start_points=init_states,
                    integration_direction=integration_direction,
                    ax=axes_list[i],
                )
            else:
                axes_list[i] = plot_flow_field(
                    vecfld, xlim, ylim, background=background, ax=axes_list[i]
                )

        if "nullcline" in terms:
            axes_list[i] = plot_nullclines(
                vecfld, background=background, ax=axes_list[i]
            )

        if "fixed_points" in terms:
            axes_list[i] = plot_fixed_points(
                vecfld, background=background, ax=axes_list[i]
            )

        if "separatrices" in terms:
            axes_list[i] = plot_separatrix(
                vecfld, xlim, ylim, t=t, background=background, ax=axes_list[i]
            )

        if init_states is not None and "trajectory" in terms:
            if not approx:
                axes_list[i] = plot_traj(
                    vecfld.func,
                    init_states,
                    t,
                    background=background,
                    integration_direction=integration_direction,
                    ax=axes_list[i],
                )

        # show quivers for the init_states cells
        if init_states is not None and "quiver" in terms:
            from .utils import default_quiver_args
            from ..tools.utils import update_dict

            X = init_states
            V /= 3 * quiver_autoscaler(X, V)

            df = pd.DataFrame({"x": X[:, 0], "y": X[:, 1], "u": V[:, 0], "v": V[:, 1]})

            if quiver_size is None:
                quiver_size = 1
            if background == "black":
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

    plt.tight_layout()
    plt.show()
