import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy
from anndata import AnnData
from matplotlib.axes import Axes

from ..configuration import _themes
from ..dynamo_logger import LoggerManager
from ..tools.cell_velocities import cell_velocities
from ..tools.utils import nearest_neighbors, update_dict
from ..vectorfield.scVectorField import BaseVectorField
from ..vectorfield.topography import (  # , compute_separatrices
    VectorField,
    VectorField2D,
)
from ..vectorfield.topography import topography as _topology  # , compute_separatrices
from ..vectorfield.utils import vecfld_from_adata
from ..vectorfield.vector_calculus import curl, divergence
from .scatters import docstrings, scatters, scatters_interactive
from .utils import (
    _plot_traj,
    _select_font_color,
    default_quiver_args,
    quiver_autoscaler,
    retrieve_plot_save_path,
    save_show_ret,
    save_plotly_figure,
    save_pyvista_plotter,
    set_arrow_alpha,
    set_stream_line_alpha,
)


def plot_flow_field(
    vecfld: VectorField2D,
    x_range: npt.ArrayLike,
    y_range: npt.ArrayLike,
    n_grid: int = 100,
    start_points: Optional[np.ndarray] = None,
    integration_direction: Literal["forward", "backward", "both"] = "both",
    background: Optional[str] = None,
    density: float = 1,
    linewidth: float = 1,
    streamline_color: Optional[str] = None,
    streamline_alpha: float = 0.4,
    color_start_points: Optional[float] = None,
    save_show_or_return: Literal["save", "show", "return"] = "return",
    save_kwargs: Dict[str, Any] = {},
    ax: Optional[Axes] = None,
    **streamline_kwargs,
) -> Optional[Axes]:
    """Plots the flow field with line thickness proportional to speed.

    Code adapted from: http://be150.caltech.edu/2017/handouts/dynamical_systems_approaches.html

    Args:
        vecfld: an instance of the vector_field class.
        x_range: the range of values for x-axis.
        y_range: the range of values for y-axis.
        n_grid: the number of grid points to use in computing derivatives on phase portrait. Defaults to 100.
        start_points: the initial points from which the streamline will be drawn. Defaults to None.
        integration_direction: integrate the streamline in forward, backward or both directions. default is 'both'.
            Defaults to "both".
        background: the background color of the plot. Defaults to None.
        density: the density of the plt.streamplot function. Defaults to 1.
        linewidth: the multiplier of automatically calculated linewidth passed to the plt.streamplot function. Defaults
            to 1.
        streamline_color: the color of the vector field streamlines. Defaults to None.
        streamline_alpha: the alpha value applied to the vector field streamlines. Defaults to 0.4.
        color_start_points: the color of the starting point that will be used to predict cell fates. Defaults to None.
        save_show_or_return: whether to save, show or return the figure. Defaults to "return".
        save_kwargs: a dictionary that will be passed to the save_show_ret function. By default, it is an empty dictionary
            and the save_show_ret function will use the {"path": None, "prefix": 'plot_flow_field', "dpi": None,
            "ext": 'pdf', "transparent": True, "close": True, "verbose": True} as its parameters. Otherwise, you can
            provide a dictionary that properly modify those keys according to your needs. Defaults to {}.
        ax: the Axis on which to make the plot. Defaults to None.

    Returns:
        None would be returned by default. If `save_show_or_return` is set to be 'return', the Axes of the generated
        figure would be returned.
    """

    from matplotlib import patches, rcParams
    from matplotlib.colors import to_hex

    if background is None:
        _background = rcParams.get("figure.facecolor")
        _background = to_hex(_background) if type(_background) is tuple else _background
    else:
        _background = background

    if _background in ["#ffffff", "black"]:
        color, color_start_points = (
            "white" if streamline_color is None else streamline_color,
            "red" if color_start_points is None else color_start_points,
        )
    else:
        color, color_start_points = (
            "black" if streamline_color is None else streamline_color,
            "red" if color_start_points is None else color_start_points,
        )

    # Set up u,v space
    u = np.linspace(x_range[0], x_range[1], n_grid)
    v = np.linspace(y_range[0], y_range[1], n_grid)
    uu, vv = np.meshgrid(u, v)

    # Compute derivatives
    u_vel = np.empty_like(uu)
    v_vel = np.empty_like(vv)
    for i in range(uu.shape[0]):
        for j in range(uu.shape[1]):
            u_vel[i, j], v_vel[i, j] = vecfld(np.array([uu[i, j], vv[i, j]]))

    # Compute speed
    speed = np.sqrt(u_vel**2 + v_vel**2)

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
            uu,
            vv,
            u_vel,
            v_vel,
            color=color,
            **streamplot_kwargs,
        )
        set_arrow_alpha(ax, streamline_alpha)
        set_stream_line_alpha(s, streamline_alpha)
    else:
        if len(start_points.shape) == 1:
            start_points.reshape((1, 2))
        ax.scatter(*start_points, marker="*", zorder=4)

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

    return save_show_ret("plot_flow_field", save_show_or_return, save_kwargs, ax)


def plot_nullclines(
    vecfld: VectorField2D,
    vecfld_dict: Dict[str, Any] = None,
    lw: float = 3,
    background: Optional[float] = None,
    save_show_or_return: Literal["save", "show", "return"] = "return",
    save_kwargs: Dict[str, Any] = {},
    ax: Optional[Axes] = None,
) -> Optional[Axes]:
    """Plot nullclines stored in the VectorField2D class.

    Args:
        vecfld: an instance of the VectorField2D class which presumably has fixed points computed and stored.
        vecfld_dict: a dict with entries to create a `VectorField2D` instance. Defaults to None.
        lw: the linewidth of the nullcline. Defaults to 3.
        background: the background color of the plot. Defaults to None.
        save_show_or_return: whether to save, show, or return the figure. Defaults to "return".
        save_kwargs: a dictionary that will be passed to the save_show_ret function. By default, it is an empty dictionary
            and the save_show_ret function will use the {"path": None, "prefix": 'plot_nullclines', "dpi": None,
            "ext": 'pdf', "transparent": True, "close": True, "verbose": True} as its parameters. Otherwise, you can
            provide a dictionary that properly modify those keys according to your needs. Defaults to {}.
        ax: the matplotlib axes used for plotting. Default is to use the current axis. Defaults to None.

    Returns:
        None would be returned by default. If `save_show_or_return` is set to be 'return', the Axes of the generated
        figure would be returned.
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

    NCx, NCy = None, None

    # if nullcline is not previously calculated, calculate and plot them
    if vecfld_dict is None or "NCx" not in vecfld_dict.keys() or "NCy" not in vecfld_dict.keys():
        if vecfld_dict is not None:
            X_basis = vecfld_dict["X"][:, :2]
            min_, max_ = X_basis.min(0), X_basis.max(0)

            xlim = [
                min_[0] - (max_[0] - min_[0]) * 0.1,
                max_[0] + (max_[0] - min_[0]) * 0.1,
            ]
            ylim = [
                min_[1] - (max_[1] - min_[1]) * 0.1,
                max_[1] + (max_[1] - min_[1]) * 0.1,
            ]

            vecfld2d = VectorField2D(vecfld, X_data=vecfld_dict["X"])
            vecfld2d.find_fixed_points_by_sampling(25, xlim, ylim)

            if vecfld2d.get_num_fixed_points() > 0:
                vecfld2d.compute_nullclines(xlim, ylim, find_new_fixed_points=True)

                NCx, NCy = vecfld2d.NCx, vecfld.NCy
    else:
        NCx, NCy = (
            [vecfld_dict["NCx"][index] for index in vecfld_dict["NCx"]],
            [vecfld_dict["NCy"][index] for index in vecfld_dict["NCy"]],
        )

    if ax is None:
        ax = plt.gca()

    if NCx is not None and NCy is not None:
        for ncx in NCx:
            ax.plot(*ncx.T, c=colors[0], lw=lw)
        for ncy in NCy:
            ax.plot(*ncy.T, c=colors[1], lw=lw)

    return save_show_ret("plot_nullclines", save_show_or_return, save_kwargs, ax)


def plot_fixed_points_2d(
    vecfld: VectorField2D,
    marker: str = "o",
    markersize: float = 200,
    cmap: Optional[str] = None,
    filltype: List[str] = ["full", "top", "none"],
    background: Optional[str] = None,
    save_show_or_return: Literal["save", "show", "return"] = "return",
    save_kwargs: Dict[str, Any] = {},
    ax: Optional[Axes] = None,
) -> Optional[Axes]:
    """Plot fixed points stored in the VectorField2D class.

    Args:
        vecfld: an instance of the VectorField2D class which presumably has fixed points computed and stored.
        marker: the marker type. Any string supported by matplotlib.markers. Defaults to "o".
        markersize: the size of the marker. Defaults to 200.
        cmap: the name of a matplotlib colormap to use for coloring or shading the confidence of fixed points. If None,
            the default color map will set to be viridis (inferno) when the background is white (black). Defaults to
            None.
        filltype: the fill type used for stable, saddle, and unstable fixed points. Default is 'full', 'top' and 'none',
            respectively. Defaults to ["full", "top", "none"].
        background: the background color of the plot. Defaults to None.
        save_show_or_return: whether to save, show or return the figure. Defaults to "return".
        save_kwargs: a dictionary that will be passed to the save_show_ret function. By default, it is an empty dictionary
            and the save_show_ret function will use the {"path": None, "prefix": 'plot_fixed_points', "dpi": None,
            "ext": 'pdf', "transparent": True, "close": True, "verbose": True} as its parameters. Otherwise, you can
            provide a dictionary that properly modify those keys according to your needs. Defaults to {}.
        ax: the matplotlib axes used for plotting. Default is to use the current axis. Defaults to None.

    Returns:
        None would be returned by default. If `save_show_or_return` is set to be 'return', the Axes of the generated
        figure would be returned.
    """

    import matplotlib
    import matplotlib.patheffects as PathEffects
    from matplotlib import markers, rcParams
    from matplotlib.colors import to_hex

    if background is None:
        _background = rcParams.get("figure.facecolor")
        _background = to_hex(_background) if type(_background) is tuple else _background
    else:
        _background = background

    if _background in ["#ffffff", "black"]:
        _theme_ = "inferno"
    else:
        _theme_ = "viridis"
    _cmap = _themes[_theme_]["cmap"] if cmap is None else cmap

    Xss, ftype = vecfld.get_fixed_points(get_types=True)
    confidence = vecfld.get_Xss_confidence()

    if ax is None:
        ax = plt.gca()

    cm = matplotlib.cm.get_cmap(_cmap) if type(_cmap) is str else _cmap
    for i in range(len(Xss)):
        cur_ftype = ftype[i]
        marker_ = markers.MarkerStyle(marker=marker, fillstyle=filltype[int(cur_ftype + 1)])
        ax.scatter(
            *Xss[i],
            marker=marker_,
            s=markersize,
            c=np.array(cm(confidence[i])).reshape(1, -1),
            edgecolor=_select_font_color(_background),
            linewidths=1,
            cmap=_cmap,
            vmin=0,
            zorder=5,
        )
        txt = ax.text(
            *Xss[i],
            repr(i),
            c=("black" if cur_ftype == -1 else "blue" if cur_ftype == 0 else "red"),
            horizontalalignment="center",
            verticalalignment="center",
            zorder=6,
            weight="bold",
        )
        txt.set_path_effects(
            [
                PathEffects.Stroke(linewidth=1.5, foreground=_background, alpha=0.8),
                PathEffects.Normal(),
            ]
        )

    return save_show_ret("plot_fixed_points", save_show_or_return, save_kwargs, ax)


def plot_fixed_points(
    vecfld: VectorField2D,
    vecfld_dict: Dict[str, Any] = None,
    marker: str = "o",
    markersize: int = 200,
    c: str = "w",
    cmap: Optional[str] = None,
    filltype: List[str] = ["full", "top", "none"],
    background: Optional[str] = None,
    save_show_or_return: Literal["save", "show", "return"] = "return",
    save_kwargs: Dict[str, Any] = {},
    plot_method: Literal["pv", "matplotlib"] = "matplotlib",
    ax: Optional[Axes] = None,
    **kwargs,
) -> Optional[Axes]:
    """Plot fixed points stored in the VectorField class.

    Args:
        vecfld: an instance of the vector_field class.
        vecfld_dict: a dict with entries to create a `VectorField2D` instance. Defaults to None.
        marker: the marker type. Any string supported by matplotlib.markers. Defaults to "o".
        markersize: the size of the marker. Defaults to 200.
        c: the marker colors. Defaults to "w".
        cmap: the name of a matplotlib colormap to use for coloring or shading the confidence of fixed points. If None,
            the default color map will set to be viridis (inferno) when the background is white (black). Defaults to
            None.
        filltype: the fill type used for stable, saddle, and unstable fixed points. Default is 'full', 'top' and 'none',
            respectively. Defaults to ["full", "top", "none"].
        background: the background color of the plot. Defaults to None.
        save_show_or_return: whether to save, show or return the figure. Defaults to "return".
        save_kwargs: a dictionary that will be passed to the save_show_ret function. By default, it is an empty dictionary
            and the save_show_ret function will use the {"path": None, "prefix": 'plot_fixed_points', "dpi": None,
            "ext": 'pdf', "transparent": True, "close": True, "verbose": True} as its parameters. Otherwise, you can
            provide a dictionary that properly modify those keys according to your needs. Defaults to {}.
        plot_method: the method to plot 3D points. Options include `pv` (pyvista) and `matplotlib`.
        ax: the matplotlib axes or pyvista plotter used for plotting. Default is to use the current axis. Defaults to
            None.

    Returns:
        None would be returned by default. If `save_show_or_return` is set to be 'return', the Axes of the generated
        figure would be returned.
    """

    import matplotlib
    import matplotlib.patheffects as PathEffects
    from matplotlib import markers, rcParams
    from matplotlib.colors import to_hex

    if background is None:
        _background = rcParams.get("figure.facecolor")
        _background = to_hex(_background) if type(_background) is tuple else _background
    else:
        _background = background

    if _background in ["#ffffff", "black"]:
        _theme_ = "inferno"
    else:
        _theme_ = "viridis"
    _cmap = _themes[_theme_]["cmap"] if cmap is None else cmap

    if vecfld_dict is None or any(("Xss" not in vecfld_dict.keys(), "ftype" not in vecfld_dict.keys())):
        if vecfld_dict is not None:
            if vecfld_dict["X"].shape[1] == 2:
                min_, max_ = vecfld_dict["X"].min(0), vecfld_dict["X"].max(0)

                xlim = [
                    min_[0] - (max_[0] - min_[0]) * 0.1,
                    max_[0] + (max_[0] - min_[0]) * 0.1,
                ]
                ylim = [
                    min_[1] - (max_[1] - min_[1]) * 0.1,
                    max_[1] + (max_[1] - min_[1]) * 0.1,
                ]

                vecfld = VectorField2D(vecfld, X_data=vecfld_dict["X"])
                vecfld.find_fixed_points_by_sampling(25, xlim, ylim)
                if vecfld.get_num_fixed_points() > 0:
                    vecfld.compute_nullclines(xlim, ylim, find_new_fixed_points=True)

                Xss, ftype = vecfld.get_fixed_points(get_types=True)
                confidence = vecfld.get_Xss_confidence()
            else:
                confidence = None
                vecfld = BaseVectorField(
                    X=vecfld_dict["X"][vecfld_dict["valid_ind"], :],
                    V=vecfld_dict["Y"][vecfld_dict["valid_ind"], :],
                    func=vecfld,
                )

                Xss, ftype = vecfld.get_fixed_points(**kwargs)
                if Xss.ndim > 1 and Xss.shape[1] > 2:
                    fp_ind = nearest_neighbors(Xss, vecfld.data["X"], 1).flatten()
                    # need to use "X_basis" to plot on the scatter point space
                    Xss = vecfld_dict["X_basis"][fp_ind]

    else:
        Xss, ftype, confidence = (
            vecfld_dict["Xss"],
            vecfld_dict["ftype"],
            vecfld_dict["confidence"],
        )

    cm = matplotlib.cm.get_cmap(_cmap) if type(_cmap) is str else _cmap
    colors = [c if confidence is None else np.array(cm(confidence[i])) for i in range(len(confidence))]
    text_colors = ["black" if cur_ftype == -1 else "blue" if cur_ftype == 0 else "red" for cur_ftype in ftype]

    if plot_method == "pv":
        try:
            import pyvista as pv
        except ImportError:
            raise ImportError("Please install pyvista first.")

        emitting_indices = [index for index, color in enumerate(text_colors) if color == "red"]
        unstable_indices = [index for index, color in enumerate(text_colors) if color == "blue"]
        absorbing_indices = [index for index, color in enumerate(text_colors) if color == "black"]
        fps_type_indices = [emitting_indices, unstable_indices, absorbing_indices]

        r, c = ax.shape[0], ax.shape[1]
        subplot_indices = [[i, j] for i in range(r) for j in range(c)]
        cur_subplot = 0

        for i in range(r * c):

            if r * c != 1:
                ax.subplot(subplot_indices[cur_subplot][0], subplot_indices[cur_subplot][1])
                cur_subplot += 1

            for indices in fps_type_indices:
                points = pv.PolyData(Xss[indices])
                points.point_data["colors"] = np.array(colors)[indices]
                points["Labels"] = [str(idx) for idx in indices]

                ax.add_points(points, render_points_as_spheres=True, rgba=True, point_size=15)
                ax.add_point_labels(
                    points,
                    "Labels",
                    text_color=text_colors[indices[0]],
                    font_size=24,
                    shape_opacity=0,
                    show_points=False,
                    always_visible=True,
                )

        return save_pyvista_plotter(
            pl=ax,
            save_show_or_return=save_show_or_return,
            save_kwargs=save_kwargs,
        )
    elif plot_method == "plotly":
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError("Please install plotly first.")

        r, c = ax._get_subplot_rows_columns()
        r, c = list(r)[-1], list(c)[-1]
        subplot_indices = [[i, j] for i in range(r) for j in range(c)]
        cur_subplot = 0

        for i in range(r * c):
            ax.add_trace(
                go.Scatter3d(
                    x=Xss[:, 0],
                    y=Xss[:, 1],
                    z=Xss[:, 2],
                    mode="markers+text",
                    marker=dict(
                        color=colors,
                        size=15,
                    ),
                    text=[str(i) for i in range(len(Xss))],
                    textfont=dict(
                        color=text_colors,
                        size=15,
                    ),
                    **kwargs,
                ),
                row=subplot_indices[cur_subplot][0] + 1, col=subplot_indices[cur_subplot][1] + 1,
            )

        return save_plotly_figure(
            pl=ax,
            save_show_or_return=save_show_or_return,
            save_kwargs=save_kwargs,
        )
    else:
        if ax is None:
            ax = plt.gca()

        for i in range(len(Xss)):
            cur_ftype = ftype[i]
            marker_ = markers.MarkerStyle(marker=marker, fillstyle=filltype[int(cur_ftype + 1)])
            ax.scatter(
                *Xss[i],
                marker=marker_,
                s=markersize,
                c=c if confidence is None else np.array(cm(confidence[i])).reshape(1, -1),
                edgecolor=_select_font_color(_background),
                linewidths=1,
                cmap=_cmap,
                vmin=0,
                zorder=5,
            )  # TODO: Figure out the user warning that no data for colormapping provided via 'c'.
            txt = ax.text(
                *Xss[i],
                repr(i),
                c=("black" if cur_ftype == -1 else "blue" if cur_ftype == 0 else "red"),
                horizontalalignment="center",
                verticalalignment="center",
                zorder=6,
                weight="bold",
            )
            txt.set_path_effects(
                [
                    PathEffects.Stroke(linewidth=1.5, foreground=_background, alpha=0.8),
                    PathEffects.Normal(),
                ]
            )

        return save_show_ret("plot_fixed_points", save_show_or_return, save_kwargs, ax)


def plot_traj(
    f: Callable,
    y0: npt.ArrayLike,
    t: npt.ArrayLike,
    args: Sequence[Any] = (),
    lw: float = 2,
    background: Optional[str] = None,
    integration_direction: Literal["forward", "backward", "both"] = "both",
    save_show_or_return: Literal["save", "show", "return"] = "return",
    save_kwargs: Dict[str, Any] = {},
    ax: Optional[Axes] = None,
) -> Optional[Axes]:
    """Plots a trajectory on a phase portrait.
    
    Code adapted from: http://be150.caltech.edu/2017/handouts/dynamical_systems_approaches.html

    Args:
        f: the function for form f(y, t, *args). It would work as the right-hand-side of the dynamical system. Must
            return a 2-array.
        y0: the initial condition.
        t: the time points for trajectory.
        args: additional arguments to be passed to f. Defaults to ().
        lw: the line width of the trajectory. Defaults to 2.
        background: the background color of the plot. Defaults to None.
        integration_direction: Determines whether to integrate the trajectory in the forward, backward, or both
            direction. Default to "both".
        save_show_or_return: whether to save, show or return the figure. Defaults to "return".
        save_kwargs: a dictionary that will be passed to the save_show_ret function. By default, it is an empty dictionary
            and the save_show_ret function will use the {"path": None, "prefix": 'plot_traj', "dpi": None, "ext": 'pdf',
            "transparent": True, "close": True, "verbose": True} as its parameters. Otherwise, you can provide a
            dictionary that properly modify those keys according to your needs. Defaults to {}.
        ax: the axis on which to make the plot. If None, new axis would be created. Defaults to None.

    Returns:
        None would be returned by default. If `save_show_or_return` is set to be 'return', the Axes of the generated
        figure would be returned.
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

    return save_show_ret("plot_traj", save_show_or_return, save_kwargs, ax)


def plot_separatrix(
    vecfld: VectorField2D,
    x_range: npt.ArrayLike,
    y_range: npt.ArrayLike,
    t: npt.ArrayLike,
    noise: float = 1e-6,
    lw: float = 3,
    vecfld_dict: Dict[str, Any] = None,
    background: Optional[str] = None,
    save_show_or_return: Literal["save", "show", "return"] = "return",
    save_kwargs: Dict[str, Any] = {},
    ax: Optional[Axes] = None,
) -> Optional[Axes]:
    """Plot separatrix on phase portrait.

    Args:
        vecfld: an instance of the VectorField2D class which presumably has fixed points computed and stored.
        x_range: the range of values for x-axis.
        y_range: the range of values for y-axis.
        t: the time points for trajectory.
        noise: a small noise added to steady states for drawing the separatrix. Defaults to 1e-6.
        lw: the line width of the trajectory. Defaults to 3.
        vecfld_dict: a dict with entries to create a `VectorField2D` instance. Defaults to None.
        background: the background color of the plot. Defaults to None.
        save_show_or_return: whether to save, show, or return the generated figure. Defaults to "return".
        save_kwargs: a dictionary that will be passed to the save_show_ret function. By default, it is an empty dictionary
            and the save_show_ret function will use the {"path": None, "prefix": 'plot_separatrix', "dpi": None,
            "ext": 'pdf', "transparent": True, "close": True, "verbose": True} as its parameters. Otherwise, you can
            provide a dictionary that properly modify those keys according to your needs. Defaults to {}.
        ax: the axis on which to make the plot. Defaults to None.

    Returns:
        None would be returned by default. If `save_show_or_return` is set to be 'return', the Axes of the generated
        figure would get returned.
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
    if vecfld_dict is None or "separatrix" not in vecfld_dict.keys():
        if vecfld_dict is not None:
            X_basis = vecfld_dict["X"][:, :2]
            min_, max_ = X_basis.min(0), X_basis.max(0)

            xlim = [
                min_[0] - (max_[0] - min_[0]) * 0.1,
                max_[0] + (max_[0] - min_[0]) * 0.1,
            ]
            ylim = [
                min_[1] - (max_[1] - min_[1]) * 0.1,
                max_[1] + (max_[1] - min_[1]) * 0.1,
            ]

            vecfld2d = VectorField2D(vecfld, X_data=vecfld_dict["X"])
            vecfld2d.find_fixed_points_by_sampling(25, xlim, ylim)

            fps, ftypes = vecfld2d.get_fixed_points(get_types=True)
            J = vecfld2d.Xss.get_J()
            saddle = fps[ftypes == 0]
            Jacobian = J[[ftypes == 0]]
            if len(saddle) > 0:
                # Negative time function to integrate to compute separatrix
                def rhs(ab, t):
                    # Unpack variables
                    a, b = ab
                    # Stop integrating if we get the edge of where we want to integrate
                    if x_range[0] < a < x_range[1] and y_range[0] < b < y_range[1]:
                        return -vecfld2d(ab)
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

                    all_sep_a = sep_a if all_sep_a is None else np.concatenate((all_sep_a, sep_a))
                    all_sep_b = sep_b if all_sep_b is None else np.concatenate((all_sep_b, sep_b))

    return save_show_ret("plot_separatrix", save_show_or_return, save_kwargs, ax)


@docstrings.with_indent(4)
def topography(
    adata: AnnData,
    basis: str = "umap",
    fps_basis: str = "umap",
    x: int = 0,
    y: int = 1,
    color: str = "ntr",
    layer: str = "X",
    highlights: Optional[list] = None,
    labels: Optional[list] = None,
    values: Optional[list] = None,
    theme: Optional[
        Literal[
            "blue",
            "red",
            "green",
            "inferno",
            "fire",
            "viridis",
            "darkblue",
            "darkred",
            "darkgreen",
        ]
    ] = None,
    cmap: Optional[str] = None,
    color_key: Union[Dict[str, str], List[str], None] = None,
    color_key_cmap: Optional[str] = None,
    background: Optional[str] = "white",
    ncols: int = 4,
    pointsize: Optional[float] = None,
    figsize: Tuple[float, float] = (6, 4),
    show_legend: str = "on data",
    use_smoothed: bool = True,
    xlim: np.ndarray = None,
    ylim: np.ndarray = None,
    t: Optional[npt.ArrayLike] = None,
    terms: List[str] = ["streamline", "fixed_points"],
    init_cells: List[int] = None,
    init_states: np.ndarray = None,
    quiver_source: Literal["raw", "reconstructed"] = "raw",
    fate: Literal["history", "future", "both"] = "both",
    approx: bool = False,
    quiver_size: Optional[float] = None,
    quiver_length: Optional[float] = None,
    density: float = 1,
    linewidth: float = 1,
    streamline_color: Optional[str] = None,
    streamline_alpha: float = 0.4,
    color_start_points: Optional[str] = None,
    markersize: float = 200,
    marker_cmap: Optional[str] = None,
    save_show_or_return: Literal["save", "show", "return"] = "show",
    save_kwargs: Dict[str, Any] = {},
    aggregate: Optional[str] = None,
    show_arrowed_spines: bool = False,
    ax: Optional[Axes] = None,
    sort: Literal["raw", "abs", "neg"] = "raw",
    frontier: bool = False,
    s_kwargs_dict: Dict[str, Any] = {},
    q_kwargs_dict: Dict[str, Any] = {},
    n: int = 25,
    **streamline_kwargs_dict,
) -> Union[Axes, List[Axes], None]:
    """Plot the streamline, fixed points (attractor / saddles), nullcline, separatrices of a recovered dynamic system
    for single cells. The plot is created on two dimensional space.

    Topography function plots the full vector field topology including streamline, fixed points, characteristic lines. A
    key difference between dynamo and Velocyto or scVelo is that we learn a functional form of a vector field which can
    be used to predict cell fate over arbitrary time and space. On states near observed cells, it retrieves the key
    kinetics dynamics from the data observed and smoothes them. When time and state is far from your observed single
    cell RNA-seq datasets, the accuracy of prediction will decay. Vector field can be efficiently reconstructed in high
    dimension or lower pca/umap space. Since we learn a vector field function, we can plot the full vector via
    streamline on the entire domain as well as predicts cell fates by providing a set of initial cell states (via
    `init_cells`, `init_states`). The nullcline and separatrix provide topological information about the reconstructed
    vector field. By definition, the x/y-nullcline is a set of points in the phase plane so that dx/dt = 0 or dy/dt=0.
    Geometrically, these are the points where the vectors are either straight up or straight down. Algebraically, we
    find the x-nullcline by solving f(x,y) = 0. The boundary different attractor basis is the separatrix
    because it separates the regions into different subregions with a specific behavior. To find them is a very
    difficult problem and separatrix calculated by dynamo requires manual inspection.

    Here is more details on the fixed points drawn on the vector field: Fixed points are concepts introduced in dynamic
    systems theory. There are three types of fixed points: 1) repeller: a repelling state that only has outflows, which
    may correspond to a pluripotent cell state (ESC) that tends to differentiate into other cell states automatically or
    under small perturbation; 2) unstable fixed points or saddle points. Those states have attraction on some dimension
    (genes or reduced dimensions) but diverge in at least one other dimension. Saddle may correspond to progenitors,
    which are differentiated from ESC/pluripotent cells and relatively stable, but can further differentiate into
    multiple terminal cell types / states; 3) lastly, stable fixed points / cell type or attractors, which only have
    inflows and attract all cell states nearby, which may correspond to stable cell types and can only be kicked out of
    its cell state under extreme perturbation or in very rare situation. Fixed points are numbered with each number
    color coded. The mapping of the color of the number to the type of fixed point are: red: repellers; blue: saddle
    points; black: attractors. The scatter point itself also has filled color, which corresponds to confidence of the
    estimated fixed point. The lighter, the more confident or the fixed points are closer to the sequenced single
    cells. Confidence of each fixed points can be used in conjunction with the Jacobian analysis for investigating
    regulatory network with spatiotemporal resolution.

    By default, we plot a figure with three subplots, each colors cells either with `potential`, `curl` or `divergence`.
    `potential` is related to the intrinsic time, where a small potential is related to smaller intrinsic time and vice
    versa. Divergence can be used to indicate the state of each cell is in. Negative values correspond to potential sink
    while positive corresponds to potential source. https://en.wikipedia.org/wiki/Divergence. Curl may be related to
    cell cycle or other cycling cell dynamics. On 2d, negative values correspond to clockwise rotation while positive
    corresponds to anticlockwise rotation.
    https://www.khanacademy.org/math/multivariable-calculus/greens-theorem-and-stokes-theorem/formal-definitions-of-divergence-and-curl/a/defining-curl
    In conjunction with cell cycle score (dyn.pp.cell_cycle_scores), curl can be used to identify cells under active
    cell cycle progression.

    Args:
        adata: an AnnData object.
        basis: the reduced dimension stored in adata.obsm. The specific basis key will be constructed in the following
            priority if exits: 1) specific layer input +  basis 2) X_ + basis 3) basis. E.g. if basis is PCA, `scatters`
            is going to look for 1) if specific layer is spliced, `spliced_pca` 2) `X_pca` (dynamo convention) 3) `pca`.
            Defaults to "umap".
        fps_basis: the basis that will be used for identifying or retrieving fixed points. Note that if `fps_basis` is
            different from `basis`, the nearest cells of the fixed point from the `fps_basis` will be found and used to
            visualize the position of the fixed point on `basis` embedding. Defaults to "umap".
        x: the column index of the low dimensional embedding for the x-axis. Defaults to 0.
        y: the column index of the low dimensional embedding for the y-axis. Defaults to 1.
        color: any column names or gene expression, etc. that will be used for coloring cells. Defaults to "ntr".
        layer: the layer of data to use for the scatter plot. Defaults to "X".
        highlights: the color group that will be highlighted. If highligts is a list of lists, each list is relate to
            each color element. Defaults to None.
        labels: an array of labels (assumed integer or categorical), one for each data sample. This will be used for
            coloring the points in the plot according to their label. Note that this option is mutually exclusive to the
            `values` option. Defaults to None.
        values: an array of values (assumed float or continuous), one for each sample. This will be used for coloring
            the points in the plot according to a colorscale associated to the total range of values. Note that this
            option is mutually exclusive to the `labels` option. Defaults to None.
        theme: A color theme to use for plotting. A small set of predefined themes are provided which have relatively
            good aesthetics. Available themes are: {'blue', 'red', 'green', 'inferno', 'fire', 'viridis', 'darkblue',
            'darkred', 'darkgreen'}. Defaults to None.
        cmap: The name of a matplotlib colormap to use for coloring or shading points. If no labels or values are passed
            this will be used for shading points according to density (largely only of relevance for very large
            datasets). If values are passed this will be used for shading according the value. Note that if theme is
            passed then this value will be overridden by the corresponding option of the theme. Defaults to None.
        color_key: the method to assign colors to categoricals. This can either be an explicit dict mapping labels to
            colors (as strings of form '#RRGGBB'), or an array like object providing one color for each distinct
            category being provided in `labels`. Either way this mapping will be used to color points according to the
            label. Note that if theme is passed then this value will be overridden by the corresponding option of the
            theme. Defaults to None.
        color_key_cmap: the name of a matplotlib colormap to use for categorical coloring. If an explicit `color_key` is
            not given a color mapping for categories can be generated from the label list and selecting a matching list
            of colors from the given colormap. Note that if theme is passed then this value will be overridden by the
            corresponding option of the theme. Defaults to None.
        background: the color of the background. Usually this will be either 'white' or 'black', but any color name will
            work. Ideally one wants to match this appropriately to the colors being used for points etc. This is one of
            the things that themes handle for you. Note that if theme is passed then this value will be overridden by
            the corresponding option of the theme. Defaults to None.
        ncols: the number of columns for the figure. Defaults to 4.
        pointsize: the scale of the point size. Actual point cell size is calculated as
            `500.0 / np.sqrt(adata.shape[0]) * pointsize`. Defaults to None.
        figsize: the width and height of a figure. Defaults to (6, 4).
        show_legend: whether to display a legend of the labels. Defaults to "on data".
        use_smoothed: whether to use smoothed values (i.e. M_s / M_u instead of spliced / unspliced, etc.). Defaults to
            True.
        xlim: the range of x-coordinate. Defaults to None.
        ylim: the range of y-coordinate. Defaults to None.
        t: the length of the time period from which to predict cell state forward or backward over time. This is used by
            the odeint function. Defaults to None.
        terms: a tuple of plotting items to include in the final topography figure. ('streamline', 'nullcline',
            'fixed_points', 'separatrix', 'trajectory', 'quiver') are all the items that we can support. Defaults to
            ["streamline", "fixed_points"].
        init_cells: cell name or indices of the initial cell states for the historical or future cell state prediction
            with numerical integration. If the names in init_cells are not find in the adata.obs_name, it will be
            treated as cell indices and must be integers. Defaults to None.
        init_states: the initial cell states for the historical or future cell state prediction with numerical
            integration. It can be either a one-dimensional array or N x 2 dimension array. The `init_state` will be
            replaced to that defined by init_cells if init_cells are not None. Defaults to None.
        quiver_source: the data source that will be used to draw the quiver plot. If `init_cells` is provided, this will
            set to be the projected RNA velocity before vector field reconstruction automatically. If `init_cells` is
            not provided, this will set to be the velocity vectors calculated from the reconstructed vector field
            function automatically. If quiver_source is `reconstructed`, the velocity vectors calculated from the
            reconstructed vector field function will be used. Defaults to "raw".
        fate: predict the historial, future or both cell fates. This corresponds to integrating the trajectory in
            forward, backward or both directions defined by the reconstructed vector field function. Defaults to "both".
        approx: whether to use streamplot to draw the integration line from the init_state. Defaults to False.
        quiver_size: the size of quiver. If None, we will use set quiver_size to be 1. Note that quiver quiver_size is
            used to calculate the head_width (10 x quiver_size), head_length (12 x quiver_size) and headaxislength
            (8 x quiver_size) of the quiver. This is done via the `default_quiver_args` function which also calculate
            the scale of the quiver (1 / quiver_length). Defaults to None.
        quiver_length: the length of quiver. The quiver length which will be used to calculate scale of quiver. Note
            that befoe applying `default_quiver_args` velocity values are first rescaled via the quiver_autoscaler
            function. Scale of quiver indicates the number of data units per arrow length unit, e.g., m/s per plot
            width; a smaller scale parameter makes the arrow longer. Defaults to None.
        density: the density of the plt.streamplot function. Defaults to 1.
        linewidth: the multiplier of automatically calculated linewidth passed to the plt.streamplot function. Defaults
            to 1.
        streamline_color: the color of the vector field streamlines. Defaults to None.
        streamline_alpha: the alpha value applied to the vector field streamlines. Defaults to 0.4.
        color_start_points: the color of the starting point that will be used to predict cell fates. Defaults to None.
        markersize: the size of the marker. Defaults to 200.
        marker_cmap: the name of a matplotlib colormap to use for coloring or shading the confidence of fixed points. If
            None, the default color map will set to be viridis (inferno) when the background is white (black). Defaults
            to None.
        save_show_or_return: whether to save, show or return the figure. Defaults to "show".
        save_kwargs: a dictionary that will be passed to the save_show_ret function. By default, it is an empty dictionary
            and the save_show_ret function will use the {"path": None, "prefix": 'topography', "dpi": None, "ext": 'pdf',
            "transparent": True, "close": True, "verbose": True} as its parameters. Otherwise, you can provide a
            dictionary that properly modify those keys according to your needs. Defaults to {}.
        aggregate: the column in adata.obs that will be used to aggregate data points. Defaults to None.
        show_arrowed_spines: whether to show a pair of arrowed spines representing the basis of the scatter is currently
            using. Defaults to False.
        ax: the axis on which to make the plot. Defaults to None.
        sort: the method to reorder data so that high values points will be on top of background points. Can be one of
            {'raw', 'abs', 'neg'}, i.e. sorted by raw data, sort by absolute values or sort by negative values. Defaults
            to "raw". Defaults to "raw".
        frontier: whether to add the frontier. Scatter plots can be enhanced by using transparency (alpha) in order to
            show area of high density and multiple scatter plots can be used to delineate a frontier. See matplotlib
            tips & tricks cheatsheet (https://github.com/matplotlib/cheatsheets). Originally inspired by figures from
            scEU-seq paper: https://science.sciencemag.org/content/367/6482/1151. Defaults to False.
        s_kwargs_dict: the dictionary of the scatter arguments. Defaults to {}.
        q_kwargs_dict: additional parameters that will be passed to plt.quiver function. Defaults to {}.
        n: Number of samples for calculating the fixed points.
        **streamline_kwargs_dict: any other kwargs that would be passed to `pyplot.streamline`.

    Returns:
        None would be returned by default. If `save_show_or_return` is set to be 'return', the Axes of the generated
        subplots would be returned.
    """

    from ..external.hodge import ddhodge

    logger = LoggerManager.gen_logger("dynamo-topography-plot")
    logger.log_time()

    from matplotlib import rcParams
    from matplotlib.colors import to_hex

    if type(color) == str:
        color = [color]
    elif color is None:
        obs_keys = adata.obs.keys()
        if np.array([key.endswith("potential") for key in obs_keys]).sum() == 0:
            ddhodge(adata, basis=basis)
            prefix = "" if basis is None else basis + "_"
            color = [prefix + "ddhodge_potential"]
        else:
            color = [np.array(obs_keys)[[key.endswith("potential") for key in obs_keys]][0]]
        if np.array([key.endswith("curl") for key in obs_keys]).sum() == 0:
            curl(adata, basis=basis)
            color.append("curl_" + basis)
        else:
            color.append(np.array(obs_keys)[[key.endswith("curl") for key in obs_keys]][0])
        if np.array([key.endswith("divergence") for key in obs_keys]).sum() == 0:
            divergence(adata, basis=basis)
            color.append("divergence_" + basis)
        else:
            color.append(np.array(obs_keys)[[key.endswith("divergence") for key in obs_keys]][0])

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
        terms.append("trajectory")

    uns_key = "VecFld" if basis == "X" else "VecFld_" + basis
    fps_uns_key = "VecFld" if fps_basis == "X" else "VecFld_" + fps_basis

    if uns_key not in adata.uns.keys():

        if "velocity_" + basis not in adata.obsm_keys():
            logger.info(
                f"velocity_{basis} is computed yet. " f"Projecting the velocity vector to {basis} basis now ...",
                indent_level=1,
            )
            cell_velocities(adata, basis=basis)

        logger.info(
            f"Vector field for {basis} is not constructed. Constructing it now ...",
            indent_level=1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if basis == fps_basis:
                logger.info(
                    f"`basis` and `fps_basis` are all {basis}. Will also map topography ...",
                    indent_level=2,
                )
                VectorField(adata, basis, map_topography=True, n=n)
            else:
                VectorField(adata, basis)
    if fps_uns_key not in adata.uns.keys():
        if "velocity_" + basis not in adata.obsm_keys():
            logger.info(
                f"velocity_{basis} is computed yet. " f"Projecting the velocity vector to {basis} basis now ...",
                indent_level=1,
            )
            cell_velocities(adata, basis=basis)

        logger.info(
            f"Vector field for {fps_basis} is not constructed. " f"Constructing it and mapping its topography now ...",
            indent_level=1,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            VectorField(adata, fps_basis, map_topography=True, n=n)
    # elif "VecFld2D" not in adata.uns[uns_key].keys():
    #     with warnings.catch_warnings():
    #         warnings.simplefilter("ignore")
    #
    #         _topology(adata, basis, VecFld=None)
    # elif "VecFld2D" in adata.uns[uns_key].keys() and type(adata.uns[uns_key]["VecFld2D"]) == str:
    #     with warnings.catch_warnings():
    #         warnings.simplefilter("ignore")
    #
    #         _topology(adata, basis, VecFld=None)

    vecfld_dict, vecfld = vecfld_from_adata(adata, basis)
    if vecfld_dict["Y"].shape[1] > 2:
        logger.info(
            f"Vector field for {fps_basis} is but its topography is not mapped. " f"Mapping topography now ...",
            indent_level=1,
        )
        new_basis = f"{basis}_{x}_{y}"
        adata.obsm["X_" + new_basis], adata.obsm["velocity_" + new_basis] = (
            adata.obsm["X_" + basis][:, [x, y]],
            adata.obsm["velocity_" + basis][:, [x, y]],
        )
        VectorField(adata, new_basis, dims=[x, y])
        vecfld_dict, vecfld = vecfld_from_adata(adata, new_basis)

    fps_vecfld_dict, fps_vecfld = vecfld_from_adata(adata, fps_basis)

    # need to use "X_basis" to plot on the scatter point space
    if "Xss" not in fps_vecfld_dict:
        # if topology is not mapped for this basis, calculate it now.
        logger.info(
            f"Vector field for {fps_basis} is but its topography is not mapped. " f"Mapping topography now ...",
            indent_level=1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _topology(adata, fps_basis, VecFld=None, n=n)
    else:
        if fps_vecfld_dict["Xss"].size > 0 and fps_vecfld_dict["Xss"].shape[1] > 2:
            fps_vecfld_dict["X_basis"], fps_vecfld_dict["Xss"] = (
                vecfld_dict["X"][:, :2],
                vecfld_dict["X"][fps_vecfld_dict["fp_ind"], :2],
            )

    xlim, ylim = (
        adata.uns[fps_uns_key]["xlim"] if xlim is None else xlim,
        adata.uns[fps_uns_key]["ylim"] if ylim is None else ylim,
    )

    if xlim is None or ylim is None:
        X_basis = vecfld_dict["X"][:, :2]
        min_, max_ = X_basis.min(0), X_basis.max(0)

        xlim = [
            min_[0] - (max_[0] - min_[0]) * 0.1,
            max_[0] + (max_[0] - min_[0]) * 0.1,
        ]
        ylim = [
            min_[1] - (max_[1] - min_[1]) * 0.1,
            max_[1] + (max_[1] - min_[1]) * 0.1,
        ]

    if init_cells is not None:
        if init_states is None:
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

            init_states = _init_states

    if quiver_source == "reconstructed" or (init_states is not None and init_cells is None):
        from ..tools.utils import vector_field_function

        V = vector_field_function(init_states, vecfld_dict, [0, 1])

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
        axes_list, color_list, font_color = (
            [axes_list],
            [color_list],
            [font_color],
        )
    for i in range(len(axes_list)):
        # ax = axes_list[i]

        axes_list[i].set_xlabel(basis + "_1")
        axes_list[i].set_ylabel(basis + "_2")
        # axes_list[i].set_aspect("equal")

        # Build the plot
        axes_list[i].set_xlim(xlim)
        axes_list[i].set_ylim(ylim)

        axes_list[i].set_facecolor(background)

        if t is None:
            if vecfld_dict["grid_V"] is None:
                max_t = np.max((np.diff(xlim), np.diff(ylim))) / np.min(np.abs(vecfld_dict["V"][:, :2]))
            else:
                max_t = np.max((np.diff(xlim), np.diff(ylim))) / np.min(np.abs(vecfld_dict["grid_V"]))

            t = np.linspace(0, max_t, 10 ** (np.min((int(np.log10(max_t)), 8))))

        integration_direction = (
            "both" if fate == "both" else "forward" if fate == "future" else "backward" if fate == "history" else "both"
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
                    streamline_alpha=streamline_alpha,
                    color_start_points=color_start_points,
                    ax=axes_list[i],
                    **streamline_kwargs_dict,
                )

        if "nullcline" in terms:
            axes_list[i] = plot_nullclines(vecfld, vecfld_dict, background=_background, ax=axes_list[i])

        if "fixed_points" in terms:
            axes_list[i] = plot_fixed_points(
                fps_vecfld,
                fps_vecfld_dict,
                background=_background,
                ax=axes_list[i],
                markersize=markersize,
                cmap=marker_cmap,
            )

        if "separatrices" in terms:
            axes_list[i] = plot_separatrix(vecfld, xlim, ylim, t=t, background=_background, ax=axes_list[i])

        if init_states is not None and "trajectory" in terms:
            if not approx:
                axes_list[i] = plot_traj(
                    vecfld,
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

            head_w, head_l, ax_l, scale = default_quiver_args(quiver_size, quiver_length)  #
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
                "zorder": 7,
            }
            quiver_kwargs = update_dict(quiver_kwargs, q_kwargs_dict)
            # axes_list[i].quiver(X_grid[:, 0], X_grid[:, 1], V_grid[:, 0], V_grid[:, 1], **quiver_kwargs)
            axes_list[i].quiver(
                df.iloc[:, 0],
                df.iloc[:, 1],
                df.iloc[:, 2],
                df.iloc[:, 3],
                **quiver_kwargs,
            )  # color='red',  facecolors='gray'

    return save_show_ret("topography", save_show_or_return, save_kwargs, axes_list if len(axes_list) > 1 else axes_list[0])


# TODO: Implement more `terms` like streamline and trajectory for 3D topography
@docstrings.with_indent(4)
def topography_3D(
    adata: AnnData,
    basis: str = "umap",
    fps_basis: str = "umap",
    x: int = 0,
    y: int = 1,
    z: int = 2,
    color: str = "ntr",
    layer: str = "X",
    plot_method: Literal["pv", "matplotlib"] = "matplotlib",
    highlights: Optional[list] = None,
    labels: Optional[list] = None,
    values: Optional[list] = None,
    theme: Optional[
        Literal[
            "blue",
            "red",
            "green",
            "inferno",
            "fire",
            "viridis",
            "darkblue",
            "darkred",
            "darkgreen",
        ]
    ] = None,
    cmap: Optional[str] = None,
    color_key: Union[Dict[str, str], List[str], None] = None,
    color_key_cmap: Optional[str] = None,
    alpha: Optional[float] = None,
    background: Optional[str] = "white",
    ncols: int = 4,
    pointsize: Optional[float] = None,
    figsize: Tuple[float, float] = (6, 4),
    show_legend: str = True,
    use_smoothed: bool = True,
    xlim: np.ndarray = None,
    ylim: np.ndarray = None,
    zlim: np.ndarray = None,
    t: Optional[npt.ArrayLike] = None,
    terms: List[str] = ["fixed_points"],
    init_cells: List[int] = None,
    init_states: np.ndarray = None,
    quiver_source: Literal["raw", "reconstructed"] = "raw",
    approx: bool = False,
    markersize: float = 200,
    marker_cmap: Optional[str] = None,
    save_show_or_return: Literal["save", "show", "return"] = "show",
    save_kwargs: Dict[str, Any] = {},
    aggregate: Optional[str] = None,
    show_arrowed_spines: bool = False,
    ax: Optional[Axes] = None,
    sort: Literal["raw", "abs", "neg"] = "raw",
    frontier: bool = False,
    s_kwargs_dict: Dict[str, Any] = {},
    n: int = 25,
) -> Union[Axes, List[Axes], None]:
    """Plot the topography of the reconstructed vector field in 3D space.

    Args:
        adata: AnnData object that contains the reconstructed vector field.
        basis: The embedding data space that will be used to plot the topography. Defaults to `umap`.
        fps_basis: The embedding data space that will be used to plot the fixed points. Defaults to `umap`.
        x: The index of the first dimension of the embedding data space that will be used to plot the topography.
            Defaults to 0.
        y: The index of the second dimension of the embedding data space that will be used to plot the topography.
            Defaults to 1.
        z: The index of the third dimension of the embedding data space that will be used to plot the topography.
            Defaults to 2.
        color: The color of the topography. Defaults to `ntr`.
        layer: The layer of the data that will be used to plot the topography. Defaults to `X`.
        plot_method: The method that will be used to plot the topography. Defaults to `matplotlib`.
        highlights: The list of gene names that will be used to highlight the gene expression on the topography.
            Defaults to None.
        labels: The list of gene names that will be used to label the gene expression on the topography. Defaults to
            None.
        values: The list of gene names that will be used to color the gene expression on the topography. Defaults to
            None.
        theme: The color theme that will be used to plot the topography. Defaults to None.
        cmap: The name of a matplotlib colormap that will be used to color the topography. Defaults to None.
        color_key: The color dictionary that will be used to color the topography. Defaults to None.
        color_key_cmap: The name of a matplotlib colormap that will be used to color the color key. Defaults to None.
        alpha: The transparency of the topography. Defaults to None.
        background: The background color of the topography. Defaults to `white`.
        ncols: The number of columns for the figure. Defaults to 4.
        pointsize: The scale of the point size. Actual point cell size is calculated as
            `500.0 / np.sqrt(adata.shape[0]) * pointsize`. Defaults to None.
        figsize: The width and height of a figure. Defaults to (6, 4).
        show_legend: Whether to display a legend of the labels. Defaults to `on data`.
        use_smoothed: Whether to use smoothed values (i.e. M_s / M_u instead of spliced / unspliced, etc.). Defaults to
            True.
        xlim: The range of x-coordinate. Defaults to None.
        ylim: The range of y-coordinate. Defaults to None.
        zlim: The range of z-coordinate. Defaults to None.
        t: The length of the time period from which to predict cell state forward or backward over time. This is used
            by the odeint function. Defaults to None.
        terms: A list of plotting items to include in the final topography figure. ('streamline', 'nullcline',
            'fixed_points', 'separatrix', 'trajectory', 'quiver') are all the items that we can support. Defaults to
            ["streamline", "fixed_points"].
        init_cells: cell name or indices of the initial cell states for the historical or future cell state prediction
            with numerical integration. If the names in init_cells are not find in the adata.obs_name, it will be
            treated as cell indices and must be integers. Defaults to None.
        init_states: the initial cell states for the historical or future cell state prediction with numerical
            integration. It can be either a one-dimensional array or N x 2 dimension array. The `init_state` will be
            replaced to that defined by init_cells if init_cells are not None. Defaults to None.
        quiver_source: the data source that will be used to draw the quiver plot. If `init_cells` is provided, this will
            set to be the projected RNA velocity before vector field reconstruction automatically. If `init_cells` is
            not provided, this will set to be the velocity vectors calculated from the reconstructed vector field
            function automatically. If quiver_source is `reconstructed`, the velocity vectors calculated from the
            reconstructed vector field function will be used. Defaults to "raw".
        approx: whether to use streamplot to draw the integration line from the init_state. Defaults to False.
        markersize: the size of the marker. Defaults to 200.
        marker_cmap: the name of a matplotlib colormap to use for coloring or shading the confidence of fixed points. If
            None, the default color map will set to be viridis (inferno) when the background is white (black). Defaults
            to None.
        save_show_or_return: Whether to save, show or return the figure. Defaults to `show`.
        save_kwargs: A dictionary that will be passed to the save_show_ret function. By default, it is an empty dictionary
            and the save_show_ret function will use the {"path": None, "prefix": 'topography', "dpi": None, "ext": 'pdf',
            "transparent": True, "close": True, "verbose": True} as its parameters. Otherwise, you can provide a
            dictionary that properly modify those keys according to your needs. Defaults to {}.
        aggregate: The column in adata.obs that will be used to aggregate data points. Defaults to None.
        show_arrowed_spines: Whether to show a pair of arrowed spines representing the basis of the scatter is currently
            using. Defaults to False.
        ax: The axis on which to make the plot. Defaults to None.
        sort: The method to reorder data so that high values points will be on top of background points. Can be one of
            {'raw', 'abs', 'neg'}, i.e. sorted by raw data, sort by absolute values or sort by negative values. Defaults
            to "raw". Defaults to "raw".
        frontier: whether to add the frontier. Scatter plots can be enhanced by using transparency (alpha) in order to
            show area of high density and multiple scatter plots can be used to delineate a frontier. See matplotlib
            tips & tricks cheatsheet (https://github.com/matplotlib/cheatsheets). Originally inspired by figures from
            scEU-seq paper: https://science.sciencemag.org/content/367/6482/1151. Defaults to False.
        s_kwargs_dict: The dictionary of the scatter arguments. Defaults to {}.
        n: Number of samples for calculating the fixed points.

    Returns:
        None would be returned by default. If `save_show_or_return` is set to be 'return', the Axes of the generated
        subplots would be returned.
    """

    logger = LoggerManager.gen_logger("dynamo-topography-plot")
    logger.log_time()

    from matplotlib import rcParams
    from matplotlib.colors import to_hex

    if type(color) == str:
        color = [color]

    if alpha is None:
        alpha = 0.8 if plot_method in ["pv", "plotly"] else 0.1

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
        terms.append("trajectory")

    uns_key = "VecFld" if basis == "X" else "VecFld_" + basis
    fps_uns_key = "VecFld" if fps_basis == "X" else "VecFld_" + fps_basis

    if uns_key not in adata.uns.keys():

        if "velocity_" + basis not in adata.obsm_keys():
            logger.info(
                f"velocity_{basis} is computed yet. " f"Projecting the velocity vector to {basis} basis now ...",
                indent_level=1,
            )
            cell_velocities(adata, basis=basis)

        logger.info(
            f"Vector field for {basis} is not constructed. Constructing it now ...",
            indent_level=1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if basis == fps_basis:
                logger.info(
                    f"`basis` and `fps_basis` are all {basis}. Will also map topography ...",
                    indent_level=2,
                )
                VectorField(adata, basis, map_topography=True, n=n)
            else:
                VectorField(adata, basis)
    if fps_uns_key not in adata.uns.keys():
        if "velocity_" + basis not in adata.obsm_keys():
            logger.info(
                f"velocity_{basis} is computed yet. " f"Projecting the velocity vector to {basis} basis now ...",
                indent_level=1,
            )
            cell_velocities(adata, basis=basis)

        logger.info(
            f"Vector field for {fps_basis} is not constructed. " f"Constructing it and mapping its topography now ...",
            indent_level=1,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            VectorField(adata, fps_basis, map_topography=True, n=n)

    vecfld_dict, vecfld = vecfld_from_adata(adata, basis)

    fps_vecfld_dict, fps_vecfld = vecfld_from_adata(adata, fps_basis)

    # need to use "X_basis" to plot on the scatter point space
    if "Xss" not in fps_vecfld_dict:
        # if topology is not mapped for this basis, calculate it now.
        logger.info(
            f"Vector field for {fps_basis} is but its topography is not mapped. " f"Mapping topography now ...",
            indent_level=1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _topology(adata, fps_basis, VecFld=None, n=n)
    else:
        if fps_vecfld_dict["Xss"].size > 0 and fps_vecfld_dict["Xss"].shape[1] > 3:
            fps_vecfld_dict["X_basis"], fps_vecfld_dict["Xss"] = (
                vecfld_dict["X"][:, :3],
                vecfld_dict["X"][fps_vecfld_dict["fp_ind"], :3],
            )

    xlim, ylim, zlim = (
        adata.uns[fps_uns_key]["xlim"] if xlim is None else xlim,
        adata.uns[fps_uns_key]["ylim"] if ylim is None else ylim,
        adata.uns[fps_uns_key]["zlim"] if zlim is None else zlim,
    )

    if xlim is None or ylim is None or zlim is None:
        X_basis = vecfld_dict["X"][:, :3]
        min_, max_ = X_basis.min(0), X_basis.max(0)

        xlim = [
            min_[0] - (max_[0] - min_[0]) * 0.1,
            max_[0] + (max_[0] - min_[0]) * 0.1,
        ]
        ylim = [
            min_[1] - (max_[1] - min_[1]) * 0.1,
            max_[1] + (max_[1] - min_[1]) * 0.1,
        ]
        zlim = [
            min_[2] - (max_[2] - min_[2]) * 0.1,
            max_[2] + (max_[2] - min_[2]) * 0.1,
        ]


    if init_cells is not None:
        if init_states is None:
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

            init_states = _init_states

    if quiver_source == "reconstructed" or (init_states is not None and init_cells is None):
        from ..tools.utils import vector_field_function

        V = vector_field_function(init_states, vecfld_dict, [0, 1])

    if plot_method == "pv":
        try:
            import pyvista as pv
        except ImportError:
            raise ImportError("Please install pyvista first.")

        pl, colors_list = scatters_interactive(
            adata=adata,
            basis=basis,
            x=x,
            y=y,
            z=z,
            color=color,
            layer=layer,
            labels=labels,
            values=values,
            cmap=cmap,
            theme=theme,
            background=background,
            color_key=color_key,
            color_key_cmap=color_key_cmap,
            use_smoothed=use_smoothed,
            save_show_or_return="return",
            # style='points_gaussian',
            opacity=alpha,
        )

        if "fixed_points" in terms:
            pl = plot_fixed_points(
                fps_vecfld,
                fps_vecfld_dict,
                background=_background,
                ax=pl,
                markersize=markersize,
                cmap=marker_cmap,
                plot_method="pv",
            )

        return save_pyvista_plotter(
            pl=pl,
            colors_list=colors_list,
            save_show_or_return=save_show_or_return,
            save_kwargs=save_kwargs,
        )
    elif plot_method == "plotly":
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError("Please install plotly first.")

        pl, colors_list = scatters_interactive(
            adata=adata,
            basis=basis,
            x=x,
            y=y,
            z=z,
            color=color,
            layer=layer,
            plot_method="plotly",
            labels=labels,
            values=values,
            cmap=cmap,
            theme=theme,
            background=background,
            color_key=color_key,
            color_key_cmap=color_key_cmap,
            use_smoothed=use_smoothed,
            save_show_or_return="return",
            opacity=alpha,
        )

        if "fixed_points" in terms:
            pl = plot_fixed_points(
                fps_vecfld,
                fps_vecfld_dict,
                background=_background,
                ax=pl,
                markersize=markersize,
                cmap=marker_cmap,
                plot_method="plotly",
            )

        return save_plotly_figure(
            pl=pl,
            colors_list=colors_list,
            save_show_or_return=save_show_or_return,
            save_kwargs=save_kwargs,
        )
    else:
        # plt.figure(facecolor=_background)
        axes_list, color_list, font_color = scatters(
            adata=adata,
            basis=basis,
            x=x,
            y=y,
            z=z,
            color=color,
            layer=layer,
            highlights=highlights,
            labels=labels,
            values=values,
            theme=theme,
            cmap=cmap,
            color_key=color_key,
            color_key_cmap=color_key_cmap,
            alpha=alpha,
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
            projection="3d",
            **s_kwargs_dict,
            return_all=True,
        )

        if type(axes_list) != list:
            axes_list, color_list, font_color = (
                [axes_list],
                [color_list],
                [font_color],
            )
        for i in range(len(axes_list)):
            # ax = axes_list[i]

            axes_list[i].set_xlabel(basis + "_1")
            axes_list[i].set_ylabel(basis + "_2")
            axes_list[i].set_zlabel(basis + "_3")
            # axes_list[i].set_aspect("equal")

            # Build the plot
            axes_list[i].set_xlim(xlim)
            axes_list[i].set_ylim(ylim)
            axes_list[i].set_zlim(zlim)

            axes_list[i].set_facecolor(background)

            if t is None:
                if vecfld_dict["grid_V"] is None:
                    max_t = np.max((np.diff(xlim), np.diff(ylim))) / np.min(np.abs(vecfld_dict["V"][:, :2]))
                else:
                    max_t = np.max((np.diff(xlim), np.diff(ylim))) / np.min(np.abs(vecfld_dict["grid_V"]))

                t = np.linspace(0, max_t, 10 ** (np.min((int(np.log10(max_t)), 8))))

            if "fixed_points" in terms:
                axes_list[i] = plot_fixed_points(
                    fps_vecfld,
                    fps_vecfld_dict,
                    background=_background,
                    ax=axes_list[i],
                    markersize=markersize,
                    cmap=marker_cmap,
                )

        return save_show_ret("topography", save_show_or_return, save_kwargs, axes_list if len(axes_list) > 1 else axes_list[0])
