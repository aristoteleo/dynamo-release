import numpy as np
import scipy
import matplotlib.pyplot as plt

from ..tools.topography import topography as _topology # , compute_separatrices
from ..configuration import set_figure_params
from .scatters import scatters
from .scatters import docstrings
from .utils import _plot_traj

def plot_flow_field(vecfld, x_range, y_range, n_grid=100, lw_min=0.5, lw_max=3,
                    start_points=None, integration_direction='both', background=None, ax=None):
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
        background = rcParams.get('figure.facecolor')

    if background == 'black':
        color, color_start_points = 'white', 'red'
    else:
        color, color_start_points = 'thistle', 'tomato'

    # Set up u,v space
    u = np.linspace(x_range[0], x_range[1], n_grid)
    v = np.linspace(y_range[0], y_range[1], n_grid)
    uu, vv = np.meshgrid(u, v)

    # Compute derivatives
    u_vel = np.empty_like(uu)
    v_vel = np.empty_like(vv)
    for i in range(uu.shape[0]):
        for j in range(uu.shape[1]):
            u_vel[i,j], v_vel[i,j] = vecfld.func(np.array([uu[i,j], vv[i,j]]))

    # Compute speed
    speed = np.sqrt(u_vel**2 + v_vel**2)

    # Make linewidths proportional to speed,
    # with minimal line width of 0.5 and max of 3
    lw = lw_min + (lw_max - lw_min) * speed / speed.max()

    # Make stream plot
    if ax is None:
        ax = plt.gca()
    if start_points is None:
        ax.streamplot(uu, vv, u_vel, v_vel, linewidth=lw, arrowsize=1.2,
                      density=1, color=color)
    else:
        if len(start_points.shape) == 1: start_points.reshape((1, 2))
        ax.scatter(*start_points, c=color, marker="*")

        ax.streamplot(uu, vv, u_vel, v_vel, linewidth=lw_max, arrowsize=1.2, start_points=start_points,
                      integration_direction=integration_direction, density=1, color=color_start_points)

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
        background = rcParams.get('figure.facecolor')

    if background == 'black':
        colors = ['#189e1a', '#1f77b4']
    else:
        colors = ['#189e1a', '#1f77b4']

    if ax is None:
        ax = plt.gca()
    for ncx in vecfld.NCx:
        ax.plot(*ncx.T, c=colors[0], lw=lw)
    for ncy in vecfld.NCy:
        ax.plot(*ncy.T, c=colors[1], lw=lw)

    return ax

def plot_fixed_points(vecfld, marker='o', markersize=10, filltype=['full', 'top', 'none'], background=None, ax=None):
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
        background = rcParams.get('figure.facecolor')

    if background == 'black':
        markercolor = '#ffffff'
    else:
        markercolor = 'k'

    Xss, ftype = vecfld.get_fixed_points(get_types=True)
    if ax is None:
        ax = plt.gca()
    for i in range(len(Xss)):
        ax.plot(*Xss[i], marker=marker, markersize=markersize, c=markercolor, fillstyle=filltype[int(ftype[i] + 1)], linestyle='none')

    return ax

def plot_traj(f, y0, t, args=(), lw=2, background=None, integration_direction='both', ax=None):
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
        background = rcParams.get('figure.facecolor')

    if background == 'black':
        color = ['#ffffff']
    else:
        color = 'black'

    if len(y0.shape) == 1:
        ax = _plot_traj(y0, t, args, integration_direction, ax, color, lw, f)
    else:
        for i in range(y0.shape[0]):
            cur_y0 = y0[i, :]
            ax = _plot_traj(cur_y0, t, args, integration_direction, ax, color, lw, f)

    return ax


def plot_separatrix(vecfld, x_range, y_range, t, noise=1e-6, lw=3, background=None, ax=None):
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
        background = rcParams.get('figure.facecolor')

    if background == 'black':
        color = ['#ffffff']
    else:
        color = 'tomato'

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
            ax.plot(sep_a, sep_b, '-', color=color, lw=lw)

            all_sep_a = sep_a if all_sep_a is None else np.concatenate((all_sep_a, sep_a))
            all_sep_b = sep_b if all_sep_b is None else np.concatenate((all_sep_b, sep_b))

    return ax


@docstrings.with_indent(4)
def topography(
        adata,
        basis='umap',
        x=0,
        y=1,
        color=None,
        layer='X',
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
        figsize=(7,5),
        show_legend=True,
        use_smoothed=True,
        xlim=None,
        ylim=None,
        t=None,
        terms=('streamline', 'fixed_points'),
        init_state=None,
        integration_direction='both',
        approx=False,
        ax=None,
        s_kwargs_dict={},
        **topography_kwargs):
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
             'separatrix', 'trajectory') are all the items that we can support.
        init_state: `numpy.ndarray` (default: None)
            Initial cell states for the historical or future cell state prediction with numerical integration. It can be
            either a one-dimensional array or N x 2 dimension array.
        integration_direction: `str` (default: `forward`)
            Integrate the trajectory in forward, backward or both directions. default is 'both'.
        approx: `bool` (default: False)
            Whether to use streamplot to draw the integration line from the init_state.
        s_kwargs_dict: `dict` (default: {})
            The dictionary of the scatter arguments.
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
        background = rcParams.get('figure.facecolor')

    if approx:
        if 'streamline' not in terms: terms.append('streamline')
        if 'trajectory' in terms: terms = list(set(terms).difference('trajectory'))

    uns_key = 'VecFld' if basis == 'X' else 'VecFld_' + basis

    if uns_key not in adata.uns.keys():
        _topology(adata, basis, VecFld=None)
    elif 'VecFld2D' not in adata.uns[uns_key].keys():
        _topology(adata, basis, VecFld=None)
    else:
        VF, vecfld = adata.uns[uns_key]["VecFld"], adata.uns[uns_key]["VecFld2D"]
        xlim, ylim = adata.uns[uns_key]["xlim"] if xlim is None else xlim, \
                     adata.uns[uns_key]["ylim"] if ylim is None else ylim

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
        'return',
        **s_kwargs_dict)

    for i in range(len(axes_list)):
        ax = axes_list[i]

        ax.set_xlabel(basis + '_1')
        ax.set_ylabel(basis + '_2')
        ax.set_aspect('equal')

        # Build the plot
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        if t is None:
            t = np.linspace(0, max(max(np.diff(xlim), np.diff(ylim)) / np.min(VF['grid_V']), 1), 10000)

        if 'streamline' in terms:
            if approx:
                ax = plot_flow_field(vecfld, xlim, ylim, background=background, start_points=init_state, integration_direction=integration_direction, ax=ax)
            else:
                ax = plot_flow_field(vecfld, xlim, ylim, background=background, ax=ax)

        if 'nullcline' in terms:
            ax = plot_nullclines(vecfld, background=background, ax=ax)

        if 'fixed_points' in terms:
            ax = plot_fixed_points(vecfld, background=background, ax=ax)

        if 'separatrices' in terms:
            ax = plot_separatrix(vecfld, xlim, ylim, t=t, background=background, ax=ax)

        if init_state is not None and 'trajectory' in terms:
            if not approx:
                ax = plot_traj(vecfld.func, init_state, t, background=background, integration_direction=integration_direction, ax=ax)

    plt.tight_layout()
    plt.show()
