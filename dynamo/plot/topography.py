import numpy as np
import scipy
import matplotlib.pyplot as plt

from ..tools.topography import topography as _topology # , compute_separatrices

def plot_flow_field(vecfld, x_range, y_range, start_points=None, n_grid=100, lw_min=0.5, lw_max=3, color='thistle', color_start_points='tomato', ax=None):
    """Plots the flow field with line thickness proportional to speed.
    code adapted from: http://be150.caltech.edu/2017/handouts/dynamical_systems_approaches.html

    Parameters
    ----------
    ax : Matplotlib Axis instance
        Axis on which to make the plot
    vecfld: :class:`~VectorField2D`
        An instance of the VectorField2D class which presumably has fixed points computed and stored.
    x_range : array_like, shape (2,)
        Range of values for x-axis.
    y_range : array_like, shape (2,)
        Range of values for y-axis.
    n_grid : int, default 100
        Number of grid points to use in computing
        derivatives on phase portrait.
        
    Returns
    -------
    output : Matplotlib Axis instance
        Axis with streamplot included.
    """
    
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
        ax.streamplot(uu, vv, u_vel, v_vel, linewidth=lw_max, arrowsize=1.2, start_points = start_points,
                      density=1, color=color_start_points)


def plot_nullclines(vecfld, colors=['#189e1a', '#1f77b4'], lw=3, ax=None):
    """Plot nullclines stored in the VectorField2D class.

    Arguments
    ---------
        vecfld: :class:`~VectorField2D`
            An instance of the VectorField2D class which presumably has fixed points computed and stored.
        ax: :class:`~matplotlib.axes.Axes`
            The matplotlib axes used for plotting. Default is to use the current axis.
    """
    if ax is None:
        ax = plt.gca()
    for ncx in vecfld.NCx:
        plt.plot(*ncx.T, c=colors[0], lw=lw)
    for ncy in vecfld.NCy:
        plt.plot(*ncy.T, c=colors[1], lw=lw)


def plot_fixed_points(vecfld, marker='o', markersize=20, markercolor='k', filltype=['full', 'top', 'none'], ax=None):
    """Plot fixed points stored in the VectorField2D class.
    
    Arguments
    ---------
        vecfld: :class:`~VectorField2D`
            An instance of the VectorField2D class which presumably has fixed points computed and stored.
        ax: :class:`~matplotlib.axes.Axes`
            The matplotlib axes used for plotting. Default is to use the current axis.
        filltype: list
            The fill type used for stable, saddle, and unstable fixed points. Default is 'full', 'top' and 'none', respectively.
    """
    Xss, ftype = vecfld.get_fixed_points(get_types=True)
    if ax is None:
        ax = plt.gca()
    for i in range(len(Xss)):
        ax.plot(*Xss[i], marker=marker, markersize=markersize, c=markercolor, fillstyle=filltype[int(ftype[i] + 1)], linestyle='none')


def plot_traj(f, y0, t, args=(), color='black', lw=2, ax=None):
    """Plots a trajectory on a phase portrait.
    code adapted from: http://be150.caltech.edu/2017/handouts/dynamical_systems_approaches.html

    Parameters
    ----------
    ax : Matplotlib Axis instance
        Axis on which to make the plot
    f : function for form f(y, t, *args)
        The right-hand-side of the dynamical system.
        Must return a 2-array.
    y0 : array_like, shape (2,)
        Initial condition.
    t : array_like
        Time points for trajectory.
    args : tuple, default ()
        Additional arguments to be passed to f
    n_grid : int, default 100
        Number of grid points to use in computing
        derivatives on phase portrait.
        
    Returns
    -------
    output : Matplotlib Axis instance
        Axis with streamplot included.
    """
    
    y = scipy.integrate.odeint(f, y0, t, args=args)
    ax.plot(*y.transpose(), color=color, lw=lw)
    return ax


def plot_separatrix(vecfld, x_range, y_range, t, eps=1e-6,
                           color='tomato', lw=3, ax=None):
    """Plot separatrix on phase portrait.
        code adapted from: http://be150.caltech.edu/2017/handouts/dynamical_systems_approaches.html

    """
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
            ab0 = fps + eps
            ab_upper = scipy.integrate.odeint(rhs, ab0, t)

            # Build lower left branch of separatrix
            ab0 = fps - eps
            ab_lower = scipy.integrate.odeint(rhs, ab0, t)

            # Concatenate, reversing lower so points are sequential
            sep_a = np.concatenate((ab_lower[::-1, 0], ab_upper[:, 0]))
            sep_b = np.concatenate((ab_lower[::-1, 1], ab_upper[:, 1]))

            # Plot
            ax.plot(sep_a, sep_b, '-', color=color, lw=lw)

            all_sep_a = sep_a if all_sep_a is None else np.concatenate((all_sep_a, sep_a))
            all_sep_b = sep_b if all_sep_b is None else np.concatenate((all_sep_b, sep_b))


def topography(adata, basis, xlim=None, ylim=None, t=None, terms=['streamline', 'nullcline', 'fixed_points', 'separatrices', 'trajectory'],
               init_state=None, figsize=(5, 5), plot=True):
    """Plot the streamline, fixed points (attractor / saddles), nullcline, separatrices of a recovered dynamic system
    for single cells. The plot is created on two dimensional space.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object.
        basis: `str` (default: `trimap`)
            The reduced dimension embedding of cells to visualize.
        t:  t_end: `float` (default 1)
            The length of the time period from which to predict cell state forward or backward over time. This is used
            by the odeint function.
        xlim: `numpy.ndarray`
            The range of x-coordinate
        ylim: `numpy.ndarray`
            The range of y-coordinate
        terms: `list` (default: ['streamline', 'nullcline', 'fixed_points', 'separatrices', 'trajectory'])
            A list of plotting items to include in the final topography figure.
        init_state: `numpy.ndarray` (default: None)
            Initial cell states for the historical or future cell state prediction with numerical integration.
        figsize: `tuple` (default: (5, 5)
            The size of figure.
        plot: `bool` (default: True)
            Whether or not to plot the topography plot or just return the axis object.

    Returns
    -------
        Plot the streamline, fixed points (attractors / saddles), nullcline, separatrices of a recovered dynamic system
        for single cells or return the corresponding axis, depending on the plot argument.
    """

    uns_key = 'VecFld' if basis == 'X' else 'VecFld_' + basis

    if uns_key not in adata.uns.keys():
        _topology(adata, basis, VecFld=None)
    elif 'VecFld2D' not in adata.uns[uns_key].keys():
        _topology(adata, basis, VecFld=None)
    else:
        VF, vecfld = adata.uns[uns_key]["VecFld"], adata.uns[uns_key]["VecFld2D"]
        xlim, ylim = adata.uns[uns_key]["xlim"] if xlim is None else xlim, \
                     adata.uns[uns_key]["ylim"] if ylim is None else ylim

    # Set up the figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_xlabel(basis + '_1')
    ax.set_ylabel(basis + '_2')
    ax.set_aspect('equal')

    # Build the plot
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    if t is None:
        t = np.linspace(0, max(max(np.diff(xlim), np.diff(ylim)) / np.percentile(np.abs(VF['grid_V']), 5)), 10000)

    if 'streamline' in terms:
        ax = plot_flow_field(vecfld, xlim, ylim, ax=ax)

    if 'nullcline' in terms:
        ax = plot_nullclines(vecfld, ax=ax)

    if 'fixed_points' in terms:
        ax = plot_fixed_points(vecfld, ax=ax)

    if 'separatrices' in terms:
        ax = plot_separatrix(vecfld, xlim, ylim, t=t, ax=ax)

    if init_state is not None and 'trajectory' in terms:
        ax = plot_traj(vecfld.func, init_state, t, ax=ax)

    if plot:
        plt.show()
    else:
        return ax

